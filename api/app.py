"""
Antigravity REST API application for Industrial Predictive Maintenance RUL Forecaster.

Endpoints:
- GET  /health   : System health status and model availability
- POST /predict  : Live Remaining Useful Life estimation with SHAP attributions
"""

import logging
from flask import Flask, jsonify, request
from api.predict import inference_engine

# Initialize logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("antigravity_api")

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health_check():
    """
    Health check endpoint verifying API uptime and model readiness.

    Returns
    -------
    Response
        JSON with 'status' and 'model_loaded' boolean flag.
    """
    return jsonify({
        "status": "ok",
        "model_loaded": inference_engine.is_ready,
        "framework": "Antigravity REST"
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict Remaining Useful Life (RUL) and return industrial alerts and SHAP factors.

    Request Body (JSON):
    {
        "engine_id": 42,
        "cycles": [
            {
                "cycle": 150,
                "op_setting_1": -0.0007,
                "op_setting_2": -0.0004,
                "op_setting_3": 100.0,
                "s1": 518.67, "s2": 641.82, ...
            }
        ]
    }

    Returns
    -------
    Response
        JSON containing 'engine_id', 'predicted_rul', 'alert_level',
        'top_shap_contributors', and 'confidence_note'.
    """
    if not request.is_json:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    payload = request.get_json()

    if not isinstance(payload, dict):
        return jsonify({"error": "Payload must be a JSON object."}), 400

    if "cycles" not in payload or not isinstance(payload["cycles"], list):
        return jsonify({"error": "Missing or invalid 'cycles' list in request payload."}), 400

    if len(payload["cycles"]) == 0:
        return jsonify({"error": "'cycles' list cannot be empty."}), 400

    try:
        result = inference_engine.predict(payload)
        return jsonify(result), 200
    except RuntimeError as re:
        logger.error("Runtime error during inference: %s", re)
        return jsonify({"error": str(re)}), 503
    except Exception as e:
        logger.exception("Inference failed unexpectedly: %s", e)
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500


if __name__ == "__main__":
    logger.info("Starting Antigravity REST API server on port 5000...")
    app.run(host="0.0.0.0", port=5000, debug=False)
