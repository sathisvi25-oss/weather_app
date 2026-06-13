import numpy as np
import pandas as pd
import joblib

from tensorflow.keras.models import load_model

# Load model
model = load_model(
    "weather_forecast_lstm.h5",
    compile=False
)

scaler = joblib.load(
    "weather_scaler.pkl"
)

encoder = joblib.load(
    "weather_label_encoder.pkl"
)

# Load dataset
df = pd.read_csv(
    "final_weather_dataset.csv"
)

features = [
    "temperature_2m",
    "relative_humidity_2m",
    "pressure_msl",
    "cloud_cover",
    "wind_speed_10m",
    "precipitation"
]

# Last 24 hours
sample = df[features].tail(24)

sample_scaled = scaler.transform(
    sample
)

X = sample_scaled.reshape(
    1,
    24,
    6
)

prediction = model.predict(X)

predicted_class = np.argmax(
    prediction
)

weather = encoder.inverse_transform(
    [predicted_class]
)[0]

print("Predicted Weather:")
print(weather)

print(
    "Confidence:",
    round(np.max(prediction)*100,2),
    "%"
)