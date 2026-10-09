import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import shap
import mlflow

from mnip.config import settings
from mnip.risk.features import FEATURE_NAMES

# Set MLflow experiment
mlflow.set_experiment("mnip_risk")

def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Calculates the Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Select samples in this bin
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
            
    return ece

def generate_mock_risk_dataset():
    """Generates a mock data/risk_features.parquet dataset if not exists."""
    os.makedirs("data", exist_ok=True)
    parquet_path = "data/risk_features.parquet"
    if not os.path.exists(parquet_path):
        print("Mocking data/risk_features.parquet...")
        np.random.seed(42)
        n_samples = 200
        
        data = {}
        # Create 31 features
        for name in FEATURE_NAMES:
            if "flag" in name:
                data[name] = np.random.choice([0.0, 1.0], size=n_samples, p=[0.8, 0.2])
            else:
                data[name] = np.random.normal(loc=5.0, scale=2.0, size=n_samples)
                
            # Create 31 missingness indicators
            data[f"{name}_missing"] = np.random.choice([0.0, 1.0], size=n_samples, p=[0.9, 0.1])
            
        # Target column: negligent_label
        # Derived from a combination of treatment delay, high risk drug, and vital signs
        score = (data["treatment_delay_hours"] * 0.15 + 
                 data["high_risk_drug_flag"] * 2.0 + 
                 data["vital_sign_deterioration_flag"] * 3.0 + 
                 np.random.normal(0, 1, size=n_samples))
                 
        data["negligent_label"] = (score > 2.5).astype(int)
        
        df = pd.DataFrame(data)
        df.to_parquet(parquet_path)
        print(f"Created mock parquet dataset with {n_samples} records.")

def train_ensemble():
    generate_mock_risk_dataset()
    parquet_path = "data/risk_features.parquet"
    df = pd.read_parquet(parquet_path)
    
    # Separate features and target
    # Feature columns: 31 features + 31 missingness flags in the exact correct order
    feature_cols = []
    for name in FEATURE_NAMES:
        feature_cols.append(name)
    for name in FEATURE_NAMES:
        feature_cols.append(f"{name}_missing")
        
    X = df[feature_cols].values
    y = df["negligent_label"].values
    
    # 5-fold CV split
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_metrics = []
    
    # Store out-of-fold predictions for meta-learner and calibrator
    oof_lgb_probs = np.zeros(len(y))
    oof_rf_probs = np.zeros(len(y))
    oof_meta_probs = np.zeros(len(y))
    
    print("Training base models via Stratified 5-Fold Cross Validation...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # 1. Base Model 1: LightGBM
        lgb_model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            random_state=42 + fold,
            verbose=-1
        )
        lgb_model.fit(X_train, y_train)
        
        # 2. Base Model 2: RandomForest
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_features="sqrt",
            random_state=42 + fold
        )
        rf_model.fit(X_train, y_train)
        
        # Out of fold predictions
        oof_lgb_probs[val_idx] = lgb_model.predict_proba(X_val)[:, 1]
        oof_rf_probs[val_idx] = rf_model.predict_proba(X_val)[:, 1]
        
        # Train a local fold-meta learner to evaluate uncalibrated meta-probs
        X_train_meta_local = np.column_stack((
            lgb_model.predict_proba(X_train)[:, 1],
            rf_model.predict_proba(X_train)[:, 1]
        ))
        meta_local = LogisticRegression(C=0.5, random_state=42)
        meta_local.fit(X_train_meta_local, y_train)
        
        X_val_meta_local = np.column_stack((oof_lgb_probs[val_idx], oof_rf_probs[val_idx]))
        oof_meta_probs[val_idx] = meta_local.predict_proba(X_val_meta_local)[:, 1]

    # Now train the final ensemble on the FULL dataset
    print("Training final stacking base learners on full dataset...")
    final_lgb = lgb.LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42, verbose=-1)
    final_lgb.fit(X, y)
    
    final_rf = RandomForestClassifier(n_estimators=200, max_features="sqrt", random_state=42)
    final_rf.fit(X, y)
    
    # Train the meta-learner on the Out-Of-Fold predictions of base models
    print("Training final Stacking Meta-learner...")
    X_meta = np.column_stack((oof_lgb_probs, oof_rf_probs))
    final_meta = LogisticRegression(C=0.5, random_state=42)
    final_meta.fit(X_meta, y)
    
    # Train Isotonic Calibrator on the meta-learner's out-of-fold probabilities
    print("Calibrating stacking outputs with Isotonic Regression...")
    final_calibrator = IsotonicRegression(out_of_bounds="clip")
    final_calibrator.fit(oof_meta_probs, y.astype(float))
    
    # Compute metrics for calibrated probabilities
    calibrated_probs = final_calibrator.transform(oof_meta_probs)
    
    # Track overall performance metrics
    auroc = roc_auc_score(y, calibrated_probs)
    auprc = average_precision_score(y, calibrated_probs)
    brier = brier_score_loss(y, calibrated_probs)
    ece = expected_calibration_error(y, calibrated_probs)
    
    print(f"\nFinal Calibrated Ensemble Performance:")
    print(f"  AUROC: {auroc:.4f}")
    print(f"  AUPRC: {auprc:.4f}")
    print(f"  Brier Score: {brier:.4f}")
    print(f"  ECE: {ece:.4f}")
    
    # 3. Log to MLflow
    with mlflow.start_run():
        mlflow.log_param("lgb_n_estimators", 300)
        mlflow.log_param("lgb_max_depth", 6)
        mlflow.log_param("rf_n_estimators", 200)
        mlflow.log_param("meta_C", 0.5)
        
        mlflow.log_metric("auroc", auroc)
        mlflow.log_metric("auprc", auprc)
        mlflow.log_metric("brier_score", brier)
        mlflow.log_metric("ece", ece)
        
        # Save models to artifact directory
        model_dir = settings.RISK_ENSEMBLE_PATH
        os.makedirs(model_dir, exist_ok=True)
        
        joblib.dump(final_lgb, os.path.join(model_dir, "lgb_model.joblib"))
        joblib.dump(final_rf, os.path.join(model_dir, "rf_model.joblib"))
        joblib.dump(final_meta, os.path.join(model_dir, "meta_learner.joblib"))
        joblib.dump(final_calibrator, os.path.join(model_dir, "calibrator.joblib"))
        
        print(f"Saved ensemble components to {model_dir}")
        
        # 4. Generate SHAP feature importance plot
        print("Generating SHAP summary plot...")
        explainer = shap.TreeExplainer(final_lgb)
        shap_values = explainer.shap_values(X)
        
        # Handle binary classification SHAP shapes
        if isinstance(shap_values, list):
            # Take target class (index 1)
            shap_vals_to_plot = shap_values[1]
        elif len(shap_values.shape) == 3:
            shap_vals_to_plot = shap_values[1]
        else:
            shap_vals_to_plot = shap_values
            
        reports_dir = "reports"
        os.makedirs(reports_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_vals_to_plot, X, feature_names=feature_cols, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(reports_dir, "shap_summary.png"))
        plt.close()
        
        print(f"Saved SHAP feature importance plot to {os.path.join(reports_dir, 'shap_summary.png')}")
        mlflow.log_artifact(os.path.join(reports_dir, "shap_summary.png"))

if __name__ == "__main__":
    train_ensemble()
