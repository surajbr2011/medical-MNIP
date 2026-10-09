import os
import joblib
import numpy as np
from functools import lru_cache
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

from mnip.config import settings
from mnip.api.schemas import RiskScoreResponse, FHIREpisode
from mnip.risk.features import extract_all_features, to_numpy, FEATURE_NAMES

class StackingEnsemble:
    """Stacking Ensemble combining LightGBM and RandomForest with Isotonic Regression calibration."""
    def __init__(self, lgb_model, rf_model, meta_learner, calibrator):
        self.lgb_model = lgb_model
        self.rf_model = rf_model
        self.meta_learner = meta_learner
        self.calibrator = calibrator

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Base models return probability for class 1
        lgb_prob = self.lgb_model.predict_proba(X)[:, 1]
        rf_prob = self.rf_model.predict_proba(X)[:, 1]
        
        # Stacking features
        X_meta = np.column_stack((lgb_prob, rf_prob))
        
        # Meta model prediction
        meta_prob = self.meta_learner.predict_proba(X_meta)[:, 1]
        
        # Isotonic Calibration
        calibrated_prob = self.calibrator.transform(meta_prob)
        return calibrated_prob

def create_fallback_ensemble() -> StackingEnsemble:
    """Creates, trains, and returns a dummy StackingEnsemble for bootstrapping."""
    print("Creating bootstrap fallback risk ensemble model...")
    # Generate dummy data: 100 samples, 62 features (31 features + 31 indicators)
    np.random.seed(42)
    X = np.random.rand(100, 62)
    # Simple rule: if feature index 0 (treatment_delay) > 0.5 and index 8 (negligence_probability) > 0.5, then 1 else 0
    y = ((X[:, 0] > 0.5) & (X[:, 8] > 0.5)).astype(int)
    
    # Base learners
    lgb_model = lgb.LGBMClassifier(n_estimators=10, max_depth=3, learning_rate=0.1, random_state=42, verbose=-1)
    lgb_model.fit(X, y)
    
    rf_model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    rf_model.fit(X, y)
    
    # Meta learner training
    lgb_train_prob = lgb_model.predict_proba(X)[:, 1]
    rf_train_prob = rf_model.predict_proba(X)[:, 1]
    X_meta = np.column_stack((lgb_train_prob, rf_train_prob))
    
    meta_learner = LogisticRegression(C=0.5, random_state=42)
    meta_learner.fit(X_meta, y)
    
    # Calibration training
    meta_train_prob = meta_learner.predict_proba(X_meta)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(meta_train_prob, y.astype(float))
    
    # Save fallback model
    model_dir = settings.RISK_ENSEMBLE_PATH
    os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(lgb_model, os.path.join(model_dir, "lgb_model.joblib"))
    joblib.dump(rf_model, os.path.join(model_dir, "rf_model.joblib"))
    joblib.dump(meta_learner, os.path.join(model_dir, "meta_learner.joblib"))
    joblib.dump(calibrator, os.path.join(model_dir, "calibrator.joblib"))
    
    return StackingEnsemble(lgb_model, rf_model, meta_learner, calibrator)

@lru_cache(maxsize=1)
def load_risk_ensemble() -> StackingEnsemble:
    """Loads stacking ensemble components once (singleton pattern)."""
    model_dir = settings.RISK_ENSEMBLE_PATH
    lgb_path = os.path.join(model_dir, "lgb_model.joblib")
    rf_path = os.path.join(model_dir, "rf_model.joblib")
    meta_path = os.path.join(model_dir, "meta_learner.joblib")
    cal_path = os.path.join(model_dir, "calibrator.joblib")
    
    if all(os.path.exists(p) for p in [lgb_path, rf_path, meta_path, cal_path]):
        try:
            lgb_model = joblib.load(lgb_path)
            rf_model = joblib.load(rf_path)
            meta_learner = joblib.load(meta_path)
            calibrator = joblib.load(cal_path)
            return StackingEnsemble(lgb_model, rf_model, meta_learner, calibrator)
        except Exception as e:
            print(f"Error loading saved ensemble: {str(e)}. Rebuilding bootstrap fallback.")
            
    return create_fallback_ensemble()

def predict_risk(episode: FHIREpisode) -> RiskScoreResponse:
    """Predicts medical negligence risk score and level from FHIREpisode data."""
    # 1. Extract feature vector
    fv = extract_all_features(episode)
    X = to_numpy(fv).reshape(1, -1) # Shape [1, 62]
    
    # 2. Get model prediction
    ensemble = load_risk_ensemble()
    risk_score = float(ensemble.predict_proba(X)[0])
    
    # 3. Determine level
    # Risk levels: Minimal(<0.25), Moderate(0.25-0.60), High(0.60-0.80), Critical(>0.80)
    if risk_score < 0.25:
        risk_level = "Minimal"
    elif risk_score <= 0.60:
        risk_level = "Moderate"
    elif risk_score <= 0.80:
        risk_level = "High"
    else:
        risk_level = "Critical"
        
    # 4. Generate SHAP explanations
    # Pre-import shap_explainer to avoid circular dependencies
    from mnip.risk.shap_explainer import explain_risk
    shap_values, narrative, top_drivers = explain_risk(ensemble, fv, risk_score)
    
    return RiskScoreResponse(
        risk_score=risk_score,
        risk_level=risk_level,
        shap_values=shap_values,
        narrative=narrative,
        top_drivers=top_drivers
    )
