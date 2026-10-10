import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
from collections import Counter

from backend.detection.model import WHO_ICPS_CATEGORIES

logger = logging.getLogger("backend.analytics")

DATA_DIR = "data"
HISTORY_FILE = os.path.join(DATA_DIR, "incidents_history.json")

# In-memory store
_INCIDENTS_HISTORY: List[Dict[str, Any]] = []

def _load_history():
    global _INCIDENTS_HISTORY
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                _INCIDENTS_HISTORY = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load incidents history from {HISTORY_FILE}: {str(e)}")
            _INCIDENTS_HISTORY = []
    else:
        _INCIDENTS_HISTORY = []

def _save_history():
    os.makedirs(DATA_DIR, exist_ok=True)
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(_INCIDENTS_HISTORY, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to persist incidents history to {HISTORY_FILE}: {str(e)}")

# Initialize on import
_load_history()

def record_incident(
    episode_id: str,
    is_negligent: bool,
    risk_level: str,
    risk_score: float,
    domain_probabilities: Dict[str, float] = None,
    top_drivers: List[str] = None,
    latency: float = 0.0
):
    """Records a processed incident into the live analytics log."""
    record = {
        "episode_id": episode_id,
        "timestamp": datetime.utcnow().isoformat(),
        "is_negligent": is_negligent,
        "risk_level": risk_level,
        "risk_score": risk_score,
        "domain_probabilities": domain_probabilities or {},
        "top_drivers": top_drivers or [],
        "latency": round(latency, 3)
    }
    _INCIDENTS_HISTORY.append(record)
    _save_history()
    logger.info(f"Recorded live incident for episode '{episode_id}' (Total logged: {len(_INCIDENTS_HISTORY)})")

def get_live_metrics() -> Dict[str, Any]:
    """Computes real production analytics without synthetic mock data."""
    total_incidents = len(_INCIDENTS_HISTORY)
    
    if total_incidents == 0:
        return {
            "kpis": {
                "total_incidents": 0,
                "negligence_rate": 0.0,
                "critical_alerts": 0,
                "latency": 0.0
            },
            "risk_values": [0, 0, 0, 0],
            "cat_counts": [0] * len(WHO_ICPS_CATEGORIES),
            "time_series": [0] * 12,
            "top_drivers": [
                {"name": "treatment_delay_hours", "value": 0.0},
                {"name": "vital_sign_deterioration_flag", "value": 0.0},
                {"name": "consent_documented_flag_missing", "value": 0.0},
                {"name": "high_risk_drug_flag", "value": 0.0},
                {"name": "staff_patient_ratio", "value": 0.0}
            ]
        }
        
    critical_alerts = sum(1 for inc in _INCIDENTS_HISTORY if inc.get("risk_level") == "Critical")
    negligent_count = sum(1 for inc in _INCIDENTS_HISTORY if inc.get("is_negligent"))
    negligence_rate = round((negligent_count / total_incidents) * 100, 1)
    
    avg_latency = round(sum(inc.get("latency", 0.0) for inc in _INCIDENTS_HISTORY) / total_incidents, 2)
    
    # Risk Distribution: Minimal, Moderate, High, Critical
    risk_values = [
        sum(1 for inc in _INCIDENTS_HISTORY if inc.get("risk_level") == "Minimal"),
        sum(1 for inc in _INCIDENTS_HISTORY if inc.get("risk_level") == "Moderate"),
        sum(1 for inc in _INCIDENTS_HISTORY if inc.get("risk_level") == "High"),
        sum(1 for inc in _INCIDENTS_HISTORY if inc.get("risk_level") == "Critical")
    ]
    
    # WHO ICPS Domain Distribution
    cat_counts = [0] * len(WHO_ICPS_CATEGORIES)
    for inc in _INCIDENTS_HISTORY:
        probs = inc.get("domain_probabilities", {})
        if probs:
            top_cat = max(probs.items(), key=lambda x: x[1])[0]
            if top_cat in WHO_ICPS_CATEGORIES:
                idx = WHO_ICPS_CATEGORIES.index(top_cat)
                cat_counts[idx] += 1
        elif inc.get("is_negligent"):
            cat_counts[0] += 1
            
    # Time Series: Past 12 weeks
    now = datetime.utcnow()
    weekly_counts = [0] * 12
    for inc in _INCIDENTS_HISTORY:
        try:
            ts = datetime.fromisoformat(inc.get("timestamp", ""))
            weeks_ago = (now - ts).days // 7
            if 0 <= weeks_ago < 12:
                # 11 - weeks_ago maps the oldest week to index 0, current week to index 11
                weekly_counts[11 - weeks_ago] += 1
        except Exception:
            weekly_counts[-1] += 1
            
    # Top Drivers
    driver_counter = Counter()
    for inc in _INCIDENTS_HISTORY:
        for driver in inc.get("top_drivers", []):
            driver_counter[driver] += 1
            
    top_driver_list = []
    if driver_counter:
        for d, count in driver_counter.most_common(5):
            top_driver_list.append({"name": d, "value": round(count / total_incidents, 2)})
    else:
        top_driver_list = [
            {"name": "treatment_delay_hours", "value": 0.0},
            {"name": "vital_sign_deterioration_flag", "value": 0.0},
            {"name": "consent_documented_flag_missing", "value": 0.0},
            {"name": "high_risk_drug_flag", "value": 0.0},
            {"name": "staff_patient_ratio", "value": 0.0}
        ]
        
    return {
        "kpis": {
            "total_incidents": total_incidents,
            "negligence_rate": negligence_rate,
            "critical_alerts": critical_alerts,
            "latency": avg_latency
        },
        "risk_values": risk_values,
        "cat_counts": cat_counts,
        "time_series": weekly_counts,
        "top_drivers": top_driver_list
    }
