import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.optim as optim
import torch.nn as nn
import datetime

# Load the saved model
model = torch.load('trained_model_final_save.pt', map_location=torch.device('cpu'))

# Load the dataset
df = pd.read_csv('AAPL.csv')

# Preprocess the dataset
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Weekday'] = df['Date'].dt.weekday
df['Close'] = df['Close'].fillna(0)  # Handle missing values (if any)

# Normalize the features
feature_cols = ['Year', 'Month', 'Day', 'Weekday', 'Close']
df[feature_cols] = (df[feature_cols] - df[feature_cols].mean()) / df[feature_cols].std()

# Define the function to predict the stock price given a specific date
def predict_price(target_data):
    # Convert the target data to PyTorch tensors
    target_features = torch.tensor(target_data[:, :-1], dtype=torch.float)
    target_edge_index = torch.tensor([[0, 0]], dtype=torch.long).t()
    target_data = Data(x=target_features, edge_index=target_edge_index)

    # Pass the data through the model to get the predicted price
    output = model(target_data)
    predicted_price = output.item()

    return predicted_price

# Create the Streamlit app
def main():
    st.title("Stock Price Prediction App")

    # Add user input for date
    date_input = st.date_input("Enter a date for stock price prediction:")

    if st.button("Predict"):
        # Convert the date input to the desired format
        target_date = np.datetime64(date_input)

        if target_date <= np.max(df['Date'].values):
            # If the target date is within the available data, make a prediction
            target_data = df[df['Date'] <= target_date].tail(1)[feature_cols].values
            predicted_price = predict_price(target_data)
            st.write("Predicted Stock Price on {}: ${:.2f}".format(target_date, predicted_price))
        else:
            # If the target date is beyond the available data, generate future predictions
            st.write("Predicting future prices...")

            # Generate future dates
            num_days = (target_date - np.max(df['Date'].values)).astype(int)
            future_dates = [np.max(df['Date'].values) + np.timedelta64(i, 'D') for i in range(1, num_days + 1)]

            # Make predictions for future dates
            future_predictions = []
            for date in future_dates:
                target_data = df[df['Date'] <= date].tail(1)[feature_cols].values
                predicted_price = predict_price(target_data)
                future_predictions.append(predicted_price)

            # Combine the future dates and predictions into a DataFrame
            future_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_predictions})

            # Display the future predictions
            st.write(future_df)

if __name__ == '__main__':
    main()
