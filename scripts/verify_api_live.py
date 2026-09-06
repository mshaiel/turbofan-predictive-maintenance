"""
Verification script for Antigravity REST API live inference.
"""

import pprint
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.app import app
from src.data_loader import load_dataset

def main():
    print("=" * 60)
    print("VERIFYING LIVE ANTIGRAVITY REST API")
    print("=" * 60)

    client = app.test_client()

    # 1. Health check
    res_health = client.get("/health")
    print("\n[GET /health]")
    print(f"Status Code: {res_health.status_code}")
    print("Response:")
    pprint.pprint(res_health.get_json())
    assert res_health.status_code == 200
    assert res_health.get_json()["model_loaded"] is True

    # 2. Extract 20 cycles of telemetry for Test Engine 1
    raw_test = load_dataset("FD001", "test")
    test_e1 = raw_test[raw_test["engine_id"] == 1].head(25)
    cycle_records = test_e1.to_dict(orient="records")

    payload = {
        "engine_id": 1,
        "cycles": cycle_records
    }

    print(f"\n[POST /predict] (Sending {len(cycle_records)} operational cycles for Engine 1)")
    res_pred = client.post("/predict", json=payload)
    print(f"Status Code: {res_pred.status_code}")
    print("Inference Response:")
    pred_data = res_pred.get_json()
    pprint.pprint(pred_data)

    assert res_pred.status_code == 200
    assert "predicted_rul" in pred_data
    assert "alert_level" in pred_data
    assert "top_shap_contributors" in pred_data
    assert len(pred_data["top_shap_contributors"]) == 3
    print("\n[SUCCESS] Antigravity REST API verified end-to-end with live SHAP attribution!")

if __name__ == "__main__":
    main()
