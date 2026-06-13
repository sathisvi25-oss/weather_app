import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# ====================================
# LOAD DATASET
# ====================================

df = pd.read_csv("final_weather_dataset.csv")

# ====================================
# FEATURES
# ====================================

features = [
    "temperature_2m",
    "relative_humidity_2m",
    "pressure_msl",
    "cloud_cover",
    "wind_speed_10m",
    "precipitation"
]

# ====================================
# CLEAN DATA
# ====================================

df = df.dropna()

# ====================================
# SCALE FEATURES
# ====================================

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(
    df[features]
)

joblib.dump(
    scaler,
    "weather_scaler.pkl"
)

# ====================================
# ENCODE LABELS
# ====================================

encoder = LabelEncoder()

labels = encoder.fit_transform(
    df["WeatherType"]
)

joblib.dump(
    encoder,
    "weather_label_encoder.pkl"
)

y = to_categorical(labels)

# ====================================
# CREATE SEQUENCES
# ====================================

look_back = 24

X = []
Y = []

for i in range(
    len(X_scaled) - look_back
):

    X.append(
        X_scaled[
            i:i+look_back
        ]
    )

    # NEXT HOUR WEATHER
    Y.append(
        y[
            i+look_back
        ]
    )

X = np.array(X)
Y = np.array(Y)

print("X Shape:", X.shape)
print("Y Shape:", Y.shape)

# ====================================
# SPLIT
# ====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

# ====================================
# MODEL
# ====================================

model = Sequential()

model.add(
    LSTM(
        64,
        return_sequences=True,
        input_shape=(24,6)
    )
)

model.add(
    Dropout(0.2)
)

model.add(
    LSTM(32)
)

model.add(
    Dropout(0.2)
)

model.add(
    Dense(
        32,
        activation="relu"
    )
)

model.add(
    Dense(
        Y.shape[1],
        activation="softmax"
    )
)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ====================================
# TRAIN
# ====================================

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=64,
    validation_data=(
        X_test,
        y_test
    )
)

# ====================================
# SAVE
# ====================================

model.save(
    "weather_forecast_lstm.h5"
)

print("Model Saved")