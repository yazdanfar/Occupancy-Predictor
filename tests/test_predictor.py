# tests/test_predictor.py

import pytest
from occupancy_predictor import OccupancyPredictor
import os

@pytest.fixture
def predictor():
    return OccupancyPredictor()

@pytest.fixture
def sample_data():
    return {
        'Temperature': 22.5,
        'Humidity': 27.2,
        'Light': 400,
        'CO2': 700,
        'HumidityRatio': 0.0048
    }

@pytest.fixture
def models_path():
    # Adjust this path to where your models are actually stored
    return os.path.join(os.path.dirname(__file__), '..', 'models')

def test_predictor_initialization(predictor):
    assert predictor.features_with_light == ['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']
    assert predictor.features_without_light == ['Temperature', 'Humidity', 'CO2', 'HumidityRatio']
    assert predictor.model_with_light is None
    assert predictor.model_without_light is None
    assert predictor.scaler_with_light is None
    assert predictor.scaler_without_light is None

def test_load_models(predictor, models_path):
    predictor.load_models(models_path)
    assert predictor.model_with_light is not None
    assert predictor.model_without_light is not None
    assert predictor.scaler_with_light is not None
    assert predictor.scaler_without_light is not None

def test_predict_with_light(predictor, sample_data, models_path):
    predictor.load_models(models_path)
    prediction = predictor.predict(sample_data, use_light=True)
    assert isinstance(prediction, int)
    assert prediction in [0, 1]

def test_predict_without_light(predictor, sample_data, models_path):
    predictor.load_models(models_path)
    prediction = predictor.predict(sample_data, use_light=False)
    assert isinstance(prediction, int)
    assert prediction in [0, 1]

def test_predict_proba_with_light(predictor, sample_data, models_path):
    predictor.load_models(models_path)
    probability = predictor.predict_proba(sample_data, use_light=True)
    assert isinstance(probability, float)
    assert 0 <= probability <= 1

def test_predict_proba_without_light(predictor, sample_data, models_path):
    predictor.load_models(models_path)
    probability = predictor.predict_proba(sample_data, use_light=False)
    assert isinstance(probability, float)
    assert 0 <= probability <= 1

def test_predict_without_loading_models(predictor, sample_data):
    with pytest.raises(ValueError):
        predictor.predict(sample_data)

def test_predict_proba_without_loading_models(predictor, sample_data):
    with pytest.raises(ValueError):
        predictor.predict_proba(sample_data)

def test_predict_with_missing_features(predictor, sample_data, models_path):
    predictor.load_models(models_path)
    incomplete_data = sample_data.copy()
    del incomplete_data['Temperature']
    with pytest.raises(ValueError):
        predictor.predict(incomplete_data)

def test_load_models_with_missing_files(predictor):
    with pytest.raises(FileNotFoundError):
        predictor.load_models('non_existent_directory')