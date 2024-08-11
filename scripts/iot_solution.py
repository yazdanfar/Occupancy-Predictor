import paho.mqtt.client as mqtt
import time
import json
import random
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# MQTT broker settings
BROKER_ADDRESS = "localhost"
BROKER_PORT = 1883

# Simulated sensor data
def generate_sensor_data():
    return {
        "temperature": round(random.uniform(20, 30), 2),
        "humidity": round(random.uniform(30, 70), 2),
        "light": round(random.uniform(0, 1000), 2),
        "co2": round(random.uniform(400, 1000), 2)
    }

# Simulated ML model (Random Forest Classifier)
class OccupancyPredictor:
    def __init__(self):
        self.model = RandomForestClassifier()
        # In a real scenario, you would load a pre-trained model
        # For this example, we'll use a dummy model
        X = np.random.rand(100, 4)
        y = np.random.randint(2, size=100)
        self.model.fit(X, y)

    def predict(self, data):
        features = np.array([[
            data['temperature'],
            data['humidity'],
            data['light'],
            data['co2']
        ]])
        return self.model.predict(features)[0]

# MQTT client for the sensor
class SensorClient:
    def __init__(self, client_id, room):
        self.client = mqtt.Client(client_id)
        self.client.connect(BROKER_ADDRESS, BROKER_PORT)
        self.room = room

    def publish_data(self):
        data = generate_sensor_data()
        topic = f"building/floor1/{self.room}/sensors"
        self.client.publish(topic, json.dumps(data))
        print(f"Published to {topic}: {data}")

# MQTT client for the data processing unit
class DataProcessingUnit:
    def __init__(self):
        self.client = mqtt.Client("data_processor")
        self.client.on_message = self.on_message
        self.client.connect(BROKER_ADDRESS, BROKER_PORT)
        self.client.subscribe("building/floor1/+/sensors")
        self.predictor = OccupancyPredictor()

    def on_message(self, client, userdata, message):
        data = json.loads(message.payload.decode())
        room = message.topic.split('/')[2]
        occupancy = self.predictor.predict(data)
        occupancy_topic = f"building/floor1/{room}/occupancy"
        self.client.publish(occupancy_topic, str(occupancy))
        print(f"Predicted occupancy for {room}: {occupancy}")

    def start(self):
        self.client.loop_start()

# MQTT client for the HVAC control unit
class HVACControlUnit:
    def __init__(self):
        self.client = mqtt.Client("hvac_controller")
        self.client.on_message = self.on_message
        self.client.connect(BROKER_ADDRESS, BROKER_PORT)
        self.client.subscribe("building/floor1/+/occupancy")

    def on_message(self, client, userdata, message):
        occupancy = int(message.payload.decode())
        room = message.topic.split('/')[2]
        hvac_command = "ON" if occupancy == 1 else "OFF"
        command_topic = f"building/floor1/{room}/hvac/command"
        self.client.publish(command_topic, hvac_command)
        print(f"HVAC command for {room}: {hvac_command}")

    def start(self):
        self.client.loop_start()

# Main execution
if __name__ == "__main__":
    # Start data processing unit
    dpu = DataProcessingUnit()
    dpu.start()

    # Start HVAC control unit
    hvac = HVACControlUnit()
    hvac.start()

    # Simulate sensors in two rooms
    sensor1 = SensorClient("sensor1", "room101")
    sensor2 = SensorClient("sensor2", "room102")

    # Simulate sensor data publication
    try:
        while True:
            sensor1.publish_data()
            sensor2.publish_data()
            time.sleep(5)  # Publish every 5 seconds
    except KeyboardInterrupt:
        print("Simulation stopped")
