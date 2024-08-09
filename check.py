from occupancy_predictor import OccupancyPredictor

# Initialize the predictor
predictor = OccupancyPredictor()

# Load the models and get metrics
model_path = 'models/'

load_time, load_memory = predictor.load_models(model_path)
print(f"Total model loading time: {load_time:.4f} seconds")
print(f"Total memory used for model loading: {load_memory:.2f} MB")

# Make a prediction and get metrics
new_data = {
    'Temperature': 22.5,
    'Humidity': 27.2,
    'Light': 400,
    'CO2': 700,
    'HumidityRatio': 0.0048
}
prediction, pred_time, pred_memory = predictor.predict(new_data)
print(f"Prediction: {prediction}")
print(f"Prediction time: {pred_time:.4f} seconds")
print(f"Memory used for prediction: {pred_memory:.2f} MB")

# Get probability prediction and metrics
prob, prob_time, prob_memory = predictor.predict_proba(new_data)
print(f"Probability: {prob:.4f}")
print(f"Probability prediction time: {prob_time:.4f} seconds")
print(f"Memory used for probability prediction: {prob_memory:.2f} MB")