# app.py

import streamlit as st
from occupancy_predictor import OccupancyPredictor

# Initialize the predictor
predictor = OccupancyPredictor()

# Load the models
predictor.load_models('models/')

st.title('Occupancy Predictor')

st.write('Enter the following data to predict room occupancy:')

temperature = st.number_input('Temperature', value=22.5, step=0.1)
humidity = st.number_input('Humidity', value=27.2, step=0.1)
light = st.number_input('Light', value=400, step=1)
co2 = st.number_input('CO2', value=700, step=1)
humidity_ratio = st.number_input('Humidity Ratio', value=0.0048, step=0.0001, format="%.4f")

if st.button('Predict'):
    input_data = {
        'Temperature': temperature,
        'Humidity': humidity,
        'Light': light,
        'CO2': co2,
        'HumidityRatio': humidity_ratio
    }

    prediction, pred_time, pred_memory = predictor.predict(input_data)
    probability, prob_time, prob_memory = predictor.predict_proba(input_data)

    st.write(f"Prediction: {'Occupied' if prediction == 1 else 'Not Occupied'}")
    st.write(f"Probability of occupancy: {probability:.4f}")

    st.write("\nPerformance Metrics:")
    st.write(f"Prediction time: {pred_time:.4f} seconds")
    st.write(f"Prediction memory usage: {pred_memory:.2f} MB")
    st.write(f"Probability calculation time: {prob_time:.4f} seconds")
    st.write(f"Probability calculation memory usage: {prob_memory:.2f} MB")