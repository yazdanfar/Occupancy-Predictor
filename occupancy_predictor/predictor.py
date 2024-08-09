import numpy as np
from typing import List, Dict, Optional, Tuple
import joblib
import os
import time
import psutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.exceptions import InconsistentVersionWarning


class OccupancyPredictor:
    def __init__(self):
        self.features_with_light: List[str] = ['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']
        self.features_without_light: List[str] = ['Temperature', 'Humidity', 'CO2', 'HumidityRatio']
        self.model_with_light: Optional[RandomForestClassifier] = None
        self.model_without_light: Optional[RandomForestClassifier] = None
        self.scaler_with_light: Optional[StandardScaler] = None
        self.scaler_without_light: Optional[StandardScaler] = None

    def load_models(self, models_dir: str = 'models') -> Tuple[float, float]:
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # Memory in MB

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
                warnings.filterwarnings("ignore", category=UserWarning)

                self.model_with_light = joblib.load(os.path.join(models_dir, 'model_with_light.joblib'))
                self.model_without_light = joblib.load(os.path.join(models_dir, 'model_without_light.joblib'))
                self.scaler_with_light = joblib.load(os.path.join(models_dir, 'scaler_with_light.joblib'))
                self.scaler_without_light = joblib.load(os.path.join(models_dir, 'scaler_without_light.joblib'))
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Unable to load model files from {models_dir}. Error: {e}")

        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # Memory in MB

        duration = end_time - start_time
        memory_used = end_memory - start_memory

        print(f"Model loading time: {duration:.4f} seconds")
        print(f"Memory used for model loading: {memory_used:.2f} MB")

        return duration, memory_used

    def predict(self, new_data: Dict[str, float], use_light: bool = True) -> Tuple[int, float, float]:
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # Memory in MB

        if use_light:
            if self.model_with_light is None or self.scaler_with_light is None:
                raise ValueError("Model with light feature not loaded. Call load_models() first.")
            model = self.model_with_light
            scaler = self.scaler_with_light
            features = self.features_with_light
        else:
            if self.model_without_light is None or self.scaler_without_light is None:
                raise ValueError("Model without light feature not loaded. Call load_models() first.")
            model = self.model_without_light
            scaler = self.scaler_without_light
            features = self.features_without_light

        try:
            input_data = [[new_data[feature] for feature in features]]
            input_data_scaled = scaler.transform(input_data)
            prediction = int(model.predict(input_data_scaled)[0])
        except KeyError as e:
            raise ValueError(f"Input data is missing required features. Error: {e}")
        except ValueError as e:
            raise ValueError(f"Error in scaling or predicting. Ensure input data is valid. Error: {e}")

        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # Memory in MB

        duration = end_time - start_time
        memory_used = end_memory - start_memory

        print(f"Prediction time: {duration:.4f} seconds")
        print(f"Memory used for prediction: {memory_used:.2f} MB")

        return prediction, duration, memory_used

    def predict_proba(self, new_data: Dict[str, float], use_light: bool = True) -> Tuple[float, float, float]:
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # Memory in MB

        if use_light:
            if self.model_with_light is None or self.scaler_with_light is None:
                raise ValueError("Model with light feature not loaded. Call load_models() first.")
            model = self.model_with_light
            scaler = self.scaler_with_light
            features = self.features_with_light
        else:
            if self.model_without_light is None or self.scaler_without_light is None:
                raise ValueError("Model without light feature not loaded. Call load_models() first.")
            model = self.model_without_light
            scaler = self.scaler_without_light
            features = self.features_without_light

        try:
            input_data = [[new_data[feature] for feature in features]]
            input_data_scaled = scaler.transform(input_data)
            probability = float(model.predict_proba(input_data_scaled)[0, 1])
        except KeyError as e:
            raise ValueError(f"Input data is missing required features. Error: {e}")
        except ValueError as e:
            raise ValueError(f"Error in scaling or predicting. Ensure input data is valid. Error: {e}")

        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # Memory in MB

        duration = end_time - start_time
        memory_used = end_memory - start_memory

        print(f"Probability prediction time: {duration:.4f} seconds")
        print(f"Memory used for probability prediction: {memory_used:.2f} MB")

        return probability, duration, memory_used