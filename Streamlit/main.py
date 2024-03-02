import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import plotly.graph_objects as go

# Load the dataset
df = pd.read_csv('AAPL.csv')

# Define the GCNModel class
class GCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.lin = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.dropout3(x)
        x = self.lin(x)
        return x.view(-1)


# Function to perform prediction
def predict(x, edge_index, edge_attr):
    model.eval()
    with torch.no_grad():
        output = model(x, edge_index)
    return output


# Load the trained model
model = GCNModel(input_dim=1, hidden_dim=256, output_dim=1, dropout_rate=0.5)
checkpoint = torch.load('trained_model_final_save.pt', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)
model.eval()

# Streamlit app configuration
st.title('Stock Price Prediction')
st.sidebar.title('Settings')

# Input section
st.sidebar.header('Input')
selected_range = st.sidebar.slider('Select the range of data', min_value=0, max_value=len(df) - 1,
                                   value=(len(df) - 50, len(df) - 1), step=1)
start_index, end_index = selected_range
selected_data = df.iloc[start_index:end_index + 1]

# Display selected data
st.subheader('Selected Data')
st.write(selected_data)

# Prediction section
st.sidebar.header('Prediction')
button = st.sidebar.button('Predict')

# Perform prediction using the loaded model
if button:
    # Prepare data for prediction
    x = torch.tensor(selected_data['Close'].values, dtype=torch.float).view(-1, 1)
    edge_index = torch.tensor([[i, i+1] for i in range(len(selected_data) - 1)], dtype=torch.long).t().contiguous()
    data_selected = Data(x=x, edge_index=edge_index)

    # Perform prediction
    predictions = predict(data_selected.x, data_selected.edge_index, data_selected.edge_attr)

    # Convert predictions to numpy array and reshape
    predictions = predictions.detach().numpy().reshape(-1)

    # Denormalize predictions
    train_mean = torch.tensor([df['Close'].mean()], dtype=torch.float).numpy()
    train_std = torch.tensor([df['Close'].std()], dtype=torch.float).numpy()
    predictions = predictions * train_std + train_mean

    # Display predicted prices
    st.subheader('Predicted Prices')
    predicted_df = selected_data.copy()
    predicted_df['Predicted Close'] = predictions
    st.write(predicted_df[['Date', 'Close', 'Predicted Close']])

    # Plot the line graph with actual prices
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=selected_data['Date'], y=selected_data['Close'], name='Actual Close'))
    fig.update_layout(title='Stock Price Chart', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig)
