import os
import joblib
import json

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "modelo_regresion_logistica.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
INFO_PATH = os.path.join(MODELS_DIR, "modelo_regresion_logistica_info.json")

_model = None
_scaler = None
_features = None


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def get_scaler():
    global _scaler
    if _scaler is None:
        _scaler = joblib.load(SCALER_PATH)
    return _scaler


def get_features():
    """
    Lista de features en el orden correcto.
    La sacamos del JSON para no hardcodear a mano.
    """
    global _features
    if _features is None:
        with open(INFO_PATH, "r", encoding="utf-8") as f:
            info = json.load(f)
        _features = info["features"]
    return _features
