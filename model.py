import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

def train_model():
    # Load the dataset
    data = pd.read_csv("data/ships.csv")

    # Prepare training data
    X = data[["Latitude", "Longitude"]]
    y_lat = data["Next Latitude"]
    y_lon = data["Next Longitude"]

    # Train separate models for latitude and longitude
    lat_model = LinearRegression()
    lon_model = LinearRegression()

    lat_model.fit(X, y_lat)
    lon_model.fit(X, y_lon)

    # Save the models to files
    joblib.dump(lat_model, "ml/lat_model.pkl")
    joblib.dump(lon_model, "ml/lon_model.pkl")
    print("Models trained and saved successfully!")
