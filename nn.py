import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os

# Get the directory of the script
script_directory = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the CSV file in the script's directory
csv_file_path = os.path.join(script_directory, 'prices.csv')

# Load the CSV file into a DataFrame
data = pd.read_csv(csv_file_path)

# Drop unnecessary columns like 'symbol' and 'date'
data = data.drop(['symbol', 'date'], axis=1)

# Normalize the data using Min-Max scaling
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Define the input (X) and target (y) variables
X = data_scaled[:, :-1]  # Features (open, close, low, high, volume)
y = data_scaled[:, -1]   # Target variable (e.g., close price)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the Neural Network
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1)  # Output layer for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the Neural Network
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the Model
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

# Make Predictions
# Example prediction (you can use new data here)
new_data = np.array([[0.5, 0.6, 0.4, 0.7, 0.8]])  # Example input data
scaled_new_data = scaler.transform(new_data)
predicted_price = model.predict(scaled_new_data)
print(f'Predicted Price: {predicted_price[0][0]}')
