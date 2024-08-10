# api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from occupancy_predictor import OccupancyPredictor

app = FastAPI(title="Occupancy Predictor API", description="API for predicting room occupancy")

# Initialize the predictor
predictor = OccupancyPredictor()


# Load the models at startup
@app.on_event("startup")
async def startup_event():
    try:
        predictor.load_models('models/')
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Failed to load models: {str(e)}")
        raise


class PredictionInput(BaseModel):
    Temperature: float
    Humidity: float
    Light: float
    CO2: float
    HumidityRatio: float


class PredictionOutput(BaseModel):
    prediction: int
    probability: float


@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    try:
        prediction, _, _ = predictor.predict(input_data.dict())
        probability, _, _ = predictor.predict_proba(input_data.dict())

        return PredictionOutput(
            prediction=prediction,
            probability=probability
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/")
async def root():
    return {"message": "Welcome to the Occupancy Predictor API. Send POST requests to /predict endpoint."}