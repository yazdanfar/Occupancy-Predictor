# OccupancyPredictor

## Overview

OccupancyPredictor is a Python class for predicting room occupancy based on environmental sensor data. It uses machine learning models to make predictions with or without light sensor data, and includes performance monitoring for time and memory usage.

## Features

- Predict room occupancy using environmental sensor data
- Option to use or exclude light sensor data in predictions
- Probability predictions for occupancy
- Performance monitoring (time and memory usage) for model operations
- Easy-to-use API for making predictions and loading models

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/OccupancyPredictor.git
   cd OccupancyPredictor
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Here's a basic example of how to use the OccupancyPredictor:

```python
from occupancy_predictor import OccupancyPredictor

# Initialize the predictor
predictor = OccupancyPredictor()

# Load the models
load_time, load_memory = predictor.load_models('path/to/models')
print(f"Model loading time: {load_time:.4f} seconds")
print(f"Memory used for model loading: {load_memory:.2f} MB")

# Prepare input data
new_data = {
    'Temperature': 22.5,
    'Humidity': 27.2,
    'Light': 400,
    'CO2': 700,
    'HumidityRatio': 0.0048
}

# Make a prediction
prediction, pred_time, pred_memory = predictor.predict(new_data)
print(f"Prediction: {prediction}")
print(f"Prediction time: {pred_time:.4f} seconds")
print(f"Memory used for prediction: {pred_memory:.2f} MB")

# Get probability prediction
prob, prob_time, prob_memory = predictor.predict_proba(new_data)
print(f"Probability: {prob:.4f}")
print(f"Probability prediction time: {prob_time:.4f} seconds")
print(f"Memory used for probability prediction: {prob_memory:.2f} MB")
```

## API Reference

### `OccupancyPredictor`

#### Methods:

- `load_models(models_dir: str = 'models') -> Tuple[float, float]`
  Loads the trained models from the specified directory. Returns the time taken and memory used for loading.

- `predict(new_data: Dict[str, float], use_light: bool = True) -> Tuple[int, float, float]`
  Makes a prediction based on the input data. Returns the prediction (0 or 1), time taken, and memory used.

- `predict_proba(new_data: Dict[str, float], use_light: bool = True) -> Tuple[float, float, float]`
  Predicts the probability of occupancy. Returns the probability, time taken, and memory used.

## Testing

To run the tests, use the following command from the project root directory:

```
pytest tests/
```

## Contributing

Contributions to OccupancyPredictor are welcome! Please feel free to submit a Pull Request.

## License

