# sarimax_model.py

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pickle

def train_sarimax_model(data, order, seasonal_order):
    model = SARIMAX(data, order=order, seasonal_order=seasonal_order)
    fitted_model = model.fit()
    return fitted_model

def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def forecast_sarimax(model, steps):
    forecast = model.get_forecast(steps=steps)
    return forecast.predicted_mean

if __name__ == "__main__":
    # Example usage
    data_path = 'data/data.csv'
    data = pd.read_csv(data_path, parse_dates=['sate'], index_col='Date')
    data = data['sales']
