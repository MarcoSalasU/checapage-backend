from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware

# Cargar el modelo
model = load_model("phishing_model.keras")

# Inicializar la app
app = FastAPI()

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Puedes restringirlo si deseas
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo para recibir el input
class FeaturesInput(BaseModel):
    features: list

# Ruta para predicción
@app.post("/predict")
def predict(data: FeaturesInput):
    try:
        features = np.array(data.features).reshape(1, -1)
        pred_prob = model.predict(features)[0][0]
        resultado = "phishing" if pred_prob > 0.5 else "seguro"
        return {
            "resultado": resultado,
            "probabilidad": round(float(pred_prob), 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
