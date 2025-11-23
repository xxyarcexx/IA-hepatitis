from flask import Flask, request, jsonify
import numpy as np
from .model_loader import get_model, get_scaler, get_features


app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Espera un JSON con las features del modelo, por ejemplo:

    {
      "Age": 30,
      "Sex": 1,
      "Estado_Civil": 0,
      "Ciudad": 2,
      "Steroid": 0,
      "Antivirals": 1,
      "Fatigue": 0,
      "Malaise": 0,
      "Anorexia": 0,
      "Liver_Big": 1,
      "Liver_Firm": 0,
      "Spleen_Palpable": 0,
      "Spiders": 0,
      "Ascites": 0,
      "Varices": 0,
      "Bilirubin": 1.2,
      "Alk_Phosphate": 85,
      "Sgot": 40,
      "Albumin": 3.5,
      "Protime": 60,
      "Histology": 1
    }

    Los valores deben estar en el MISMO formato numérico
    que usaste para entrenar el modelo.
    """
    data = request.get_json()

    if not data:
        return jsonify({"error": "Debes enviar un JSON en el body"}), 400

    features = get_features()

    # Verificamos que vengan todas las features
    missing = [f for f in features if f not in data]
    if missing:
        return jsonify({
            "error": "Faltan features en el JSON",
            "missing_features": missing
        }), 400

    # Construimos el vector de entrada en el orden correcto
    x = [data[f] for f in features]  # lista de 21 valores
    X = np.array([x])  # shape (1, n_features)

    scaler = get_scaler()
    model = get_model()

    # Escalamos
    X_scaled = scaler.transform(X)

    # Predicción
    y_pred = model.predict(X_scaled)[0]

    # Probabilidad (si el modelo lo soporta)
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = float(model.predict_proba(X_scaled)[0][1])  # clase positiva

    return jsonify({
        "input_order": features,
        "input_values": x,
        "prediction": int(y_pred),
        "positive_probability": y_proba
    })


if __name__ == "__main__":
    # Servidor de desarrollo
    app.run(host="0.0.0.0", port=5000, debug=True)
