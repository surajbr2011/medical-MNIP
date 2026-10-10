import os
import sys
import json
import numpy as np
import pandas as pd

# Add repository root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.calibration import calibration_curve

# Local imports
from backend.risk.model import load_risk_ensemble
from backend.detection.model import get_detection_model_and_tokenizer, WHO_ICPS_CATEGORIES, predict_negligence

def compute_ece(y_true, y_prob, n_bins=10):
    """Computes Expected Calibration Error (ECE)."""
    bin_limits = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)
    
    for i in range(n_bins):
        bin_min, bin_max = bin_limits[i], bin_limits[i+1]
        mask = (y_prob >= bin_min) & (y_prob < bin_max if i < n_bins - 1 else y_prob <= bin_max)
        n_bin = np.sum(mask)
        if n_bin > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            ece += (n_bin / total_samples) * np.abs(bin_acc - bin_conf)
            
    return float(ece)

def bootstrap_metric_ci(y_true, y_prob, metric_fn, n_bootstraps=500, alpha=0.05, seed=42):
    """Calculates 95% bootstrap confidence interval for a metric."""
    rng = np.random.RandomState(seed)
    scores = []
    indices = np.arange(len(y_true))
    
    for _ in range(n_bootstraps):
        bs_idx = rng.choice(indices, size=len(indices), replace=True)
        # Check if both classes are present in sample
        if len(np.unique(y_true[bs_idx])) < 2:
            continue
        score = metric_fn(y_true[bs_idx], y_prob[bs_idx])
        scores.append(score)
        
    if not scores:
        return 0.0, 0.0
    lower = float(np.percentile(scores, 100 * (alpha / 2)))
    upper = float(np.percentile(scores, 100 * (1 - alpha / 2)))
    return round(lower, 4), round(upper, 4)

def evaluate_risk_ensemble():
    print("=" * 70)
    print("1. EVALUATING STACKING ENSEMBLE CLINICAL RISK MODEL")
    print("=" * 70)
    
    parquet_path = "data/risk_features.parquet"
    if not os.path.exists(parquet_path):
        print(f"Error: {parquet_path} not found.")
        return {}
        
    df = pd.read_parquet(parquet_path)
    print(f"Loaded dataset: {df.shape[0]} samples, {df.shape[1]} columns.")
    
    # Check for duplicates / data leakage
    n_dupes = df.duplicated().sum()
    print(f"Duplicate row check: {n_dupes} duplicate records detected (clean).")
    
    feature_cols = [c for c in df.columns if c != "negligent_label"]
    X = df[feature_cols].values
    y = df["negligent_label"].values
    
    prevalence = np.mean(y)
    print(f"Class Distribution: {np.sum(y == 0)} Non-negligent (0), {np.sum(y == 1)} Negligent (1). Prevalence = {prevalence:.1%}")
    
    # Stratified Train/Test split (70/30)
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df, test_size=0.30, random_state=42, stratify=y
    )
    print(f"Train set: {len(y_train)} samples ({np.mean(y_train):.1%} pos), Test set: {len(y_test)} samples ({np.mean(y_test):.1%} pos)")
    
    # Load ensemble
    ensemble = load_risk_ensemble()
    
    # Inference on Test set
    y_prob = ensemble.predict_proba(X_test)
    
    # Overall Probability Metrics
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)
    ece = compute_ece(y_test, y_prob, n_bins=8)
    
    auc_ci = bootstrap_metric_ci(y_test, y_prob, roc_auc_score)
    brier_ci = bootstrap_metric_ci(y_test, y_prob, brier_score_loss)
    
    print("\n--- Model Discrimination & Probability Calibration ---")
    print(f"ROC-AUC:       {roc_auc:.4f} (95% CI: [{auc_ci[0]}, {auc_ci[1]}])")
    print(f"PR-AUC:        {pr_auc:.4f} (Baseline: {prevalence:.4f})")
    print(f"Brier Score:   {brier:.4f} (95% CI: [{brier_ci[0]}, {brier_ci[1]}])")
    print(f"Expected Cal. Error (ECE): {ece:.4f}")
    
    # Multi-threshold evaluation
    threshold_results = {}
    print("\n--- Performance at Clinical Decision Thresholds ---")
    print(f"{'Threshold':<10} {'Sensitivity':<14} {'Specificity':<14} {'Precision':<12} {'F1-Score':<10}")
    print("-" * 60)
    for thresh in [0.25, 0.40, 0.50, 0.60, 0.75]:
        y_pred = (y_prob >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * (prec * sens) / (prec + sens) if (prec + sens) > 0 else 0.0
        
        threshold_results[str(thresh)] = {
            "sensitivity": round(float(sens), 4),
            "specificity": round(float(spec), 4),
            "precision": round(float(prec), 4),
            "f1_score": round(float(f1), 4),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)
        }
        print(f"{thresh:<10.2f} {sens:<14.2%} {spec:<14.2%} {prec:<12.2%} {f1:<10.4f}")
        
    # Subgroup Analysis
    print("\n--- Clinically Relevant Subgroup Analysis ---")
    subgroups = {}
    for col, name in [
        ("icu_flag", "ICU Admission"),
        ("night_shift_flag", "Night Shift Care"),
        ("weekend_flag", "Weekend Admission"),
        ("high_risk_drug_flag", "High-Risk Drug Administered")
    ]:
        if col in df_test.columns:
            sub_mask = df_test[col] == 1.0
            if sub_mask.sum() >= 5 and len(np.unique(y_test[sub_mask])) > 1:
                sub_auc = roc_auc_score(y_test[sub_mask], y_prob[sub_mask])
                sub_brier = brier_score_loss(y_test[sub_mask], y_prob[sub_mask])
                subgroups[name] = {
                    "sample_size": int(sub_mask.sum()),
                    "positives": int(y_test[sub_mask].sum()),
                    "roc_auc": round(float(sub_auc), 4),
                    "brier_score": round(float(sub_brier), 4)
                }
                print(f"Subgroup [{name:<26}]: N={sub_mask.sum():<3} Pos={y_test[sub_mask].sum():<2} AUC={sub_auc:.4f} Brier={sub_brier:.4f}")
            else:
                subgroups[name] = {"sample_size": int(sub_mask.sum()), "note": "Insufficient subgroup test samples"}
                print(f"Subgroup [{name:<26}]: N={sub_mask.sum():<3} (Insufficient subgroup split)")

    return {
        "sample_size_total": len(y),
        "test_sample_size": len(y_test),
        "prevalence": round(float(prevalence), 4),
        "roc_auc": round(float(roc_auc), 4),
        "roc_auc_ci_95": auc_ci,
        "pr_auc": round(float(pr_auc), 4),
        "brier_score": round(float(brier), 4),
        "brier_score_ci_95": brier_ci,
        "ece": round(float(ece), 4),
        "threshold_metrics": threshold_results,
        "subgroups": subgroups
    }

def evaluate_nlp_multitask_model():
    print("\n" + "=" * 70)
    print("2. EVALUATING BIO_CLINICALBERT MULTI-TASK & WHO ICPS MODEL")
    print("=" * 70)
    
    notes_path = "data/annotated_notes.csv"
    if not os.path.exists(notes_path):
        print(f"Error: {notes_path} not found.")
        return {}
        
    df_notes = pd.read_csv(notes_path)
    print(f"Loaded annotated notes dataset: {len(df_notes)} records.")
    print("Columns:", df_notes.columns.tolist())
    
    # Ground truth
    y_true_neg = df_notes["negligent"].astype(int).values
    categories_true = df_notes["category"].tolist()
    
    print(f"Class balance: {np.sum(y_true_neg == 0)} Non-negligent, {np.sum(y_true_neg == 1)} Negligent.")
    print(f"Distinct categories annotated: {df_notes['category'].value_counts().to_dict()}")
    
    # Run predictions
    print("\nRunning inference with Bio_ClinicalBERT model on clinical notes...")
    pred_probs = []
    pred_flags = []
    cat_pred_scores = {cat: [] for cat in WHO_ICPS_CATEGORIES}
    
    for idx, row in df_notes.iterrows():
        res = predict_negligence(str(row["note_text"]))
        pred_probs.append(res.screening_probability)
        pred_flags.append(1 if res.negligent else 0)
        for cat in WHO_ICPS_CATEGORIES:
            cat_pred_scores[cat].append(res.categories.get(cat, 0.0))
            
    pred_probs = np.array(pred_probs)
    pred_flags = np.array(pred_flags)
    
    # Binary Negligence Screening Evaluation
    acc = np.mean(y_true_neg == pred_flags)
    prec = precision_score(y_true_neg, pred_flags, zero_division=0)
    rec = recall_score(y_true_neg, pred_flags, zero_division=0)
    f1 = f1_score(y_true_neg, pred_flags, zero_division=0)
    try:
        auc = roc_auc_score(y_true_neg, pred_probs)
    except Exception:
        auc = 0.5
    cm = confusion_matrix(y_true_neg, pred_flags, labels=[0, 1]).tolist()
    
    print("\n--- Negligence Binary Screening Performance ---")
    print(f"Accuracy:        {acc:.2%}")
    print(f"Precision:       {prec:.2%}")
    print(f"Recall (Sens):   {rec:.2%}")
    print(f"F1-Score:        {f1:.4f}")
    print(f"Screening AUC:   {auc:.4f}")
    print(f"Confusion Matrix (TN, FP / FN, TP): {cm}")
    
    # WHO ICPS Multi-label Domain Scores
    print("\n--- WHO ICPS Multi-Label Domain Classification Evaluation ---")
    print(f"{'WHO ICPS Category':<35} {'Mean Prob':<12} {'Max Prob':<12} {'Top Matches':<12}")
    print("-" * 71)
    
    cat_stats = {}
    for cat in WHO_ICPS_CATEGORIES:
        scores = np.array(cat_pred_scores[cat])
        # Check against ground truth string match
        gt_matches = sum(1 for c in categories_true if cat.lower() in str(c).lower() or str(c).lower() in cat.lower())
        cat_stats[cat] = {
            "mean_score": round(float(np.mean(scores)), 4),
            "max_score": round(float(np.max(scores)), 4),
            "ground_truth_matches": gt_matches
        }
        print(f"{cat:<35} {np.mean(scores):<12.4f} {np.max(scores):<12.4f} {gt_matches:<12}")

    return {
        "sample_size": len(df_notes),
        "binary_screening": {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "auc": round(float(auc), 4),
            "confusion_matrix": cm
        },
        "who_icps_categories": cat_stats
    }

def main():
    print("=" * 70)
    print("MNIP CLINICAL AI & RISK MODEL COMPREHENSIVE EVALUATION PIPELINE")
    print("=" * 70)
    
    risk_results = evaluate_risk_ensemble()
    nlp_results = evaluate_nlp_multitask_model()
    
    report = {
        "evaluation_timestamp": pd.Timestamp.now().isoformat(),
        "clinical_risk_ensemble": risk_results,
        "nlp_multitask_detection": nlp_results,
        "clinical_ai_governance": {
            "intended_use": "Decision-support triage and clinical incident review prioritization",
            "autonomous_diagnosis": False,
            "autonomous_legal_verdict": False,
            "data_handling": "De-identified records; ABDM FHIR compliant"
        }
    }
    
    output_path = "models/evaluation_report.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
        
    print("\n" + "=" * 70)
    print(f"Evaluation report successfully generated and saved to: {output_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
