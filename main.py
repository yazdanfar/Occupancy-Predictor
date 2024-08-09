# main.py

import sys
import logging
from typing import Dict
from occupancy_predictor import OccupancyPredictor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_input_data() -> Dict[str, float]:
    data = {}
    features = ['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']
    for feature in features:
        while True:
            try:
                value = float(input(f"{feature}: "))
                data[feature] = value
                break
            except ValueError:
                print(f"Please enter a valid number for {feature}.")
    return data


def main():
    predictor = OccupancyPredictor()

    try:
        load_duration, load_memory = predictor.load_models()
        logging.info(
            f"Models loaded successfully in {load_duration:.4f} seconds, using {load_memory:.2f} MB of memory.")
    except FileNotFoundError as e:
        logging.error(f"Error loading models: {e}")
        sys.exit(1)

    while True:
        print("\nEnter data for prediction (or 'q' to quit):")
        user_input = input()
        if user_input.lower() == 'q':
            break

        data = get_input_data()
        use_light = 'Light' in data

        try:
            prediction, pred_duration, pred_memory = predictor.predict(data, use_light)
            probability, prob_duration, prob_memory = predictor.predict_proba(data, use_light)

            logging.info(f"Prediction: {'Occupied' if prediction == 1 else 'Not Occupied'}")
            logging.info(f"Probability of occupancy: {probability:.4f}")
            logging.info(f"Prediction time: {pred_duration:.4f} seconds")
            logging.info(f"Prediction memory usage: {pred_memory:.2f} MB")
            logging.info(f"Probability calculation time: {prob_duration:.4f} seconds")
            logging.info(f"Probability calculation memory usage: {prob_memory:.2f} MB")

            # Only print the final prediction
            print(f"Prediction: {'Occupied' if prediction == 1 else 'Not Occupied'}")
        except ValueError as e:
            logging.error(f"Error during prediction: {e}")


if __name__ == "__main__":
    main()