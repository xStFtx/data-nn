import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression
from tensorflow.keras.layers import (
    Dense,
    Dropout,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    LearningRateScheduler,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('prices.csv')

# Handle missing values by filling them with the mean of the column
data.fillna(data.mean(), inplace=True)

# Feature Engineering: Calculate additional technical indicators
data['10_day_ma'] = data['close'].rolling(window=10).mean()
data['30_day_std'] = data['close'].rolling(window=30).std()

# Extract features and target
X = data[['open', 'low', 'high', 'volume', '10_day_ma', '30_day_std']].values
y = data['close'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features using Min-Max scaling
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply Feature Selection using SelectKBest with f_regression
selector = SelectKBest(score_func=f_regression, k=4)
X_train = selector.fit_transform(X_train, y_train)
X_test = selector.transform(X_test)

# Build a more advanced model with a deep neural network
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),  # Input layer with selected features
    Dense(128, activation='relu'),  # Dense layer with more neurons
    Dropout(0.4),  # Dropout layer for regularization
    Dense(64, activation='relu'),  # Additional dense layer
    Dense(1)  # Output layer with 1 unit (regression)
])

# Compile the model with Adam optimizer and Mean Absolute Error loss
optimizer = Adam(learning_rate=1e-3)
model.compile(optimizer=optimizer, loss='mean_absolute_error')

# Implement a learning rate scheduler and ReduceLROnPlateau
def lr_schedule(epoch):
    if epoch < 50:
        return 1e-3
    elif epoch < 75:
        return 5e-4
    else:
        return 1e-4

lr_scheduler = LearningRateScheduler(lr_schedule)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6, verbose=1)

# Implement early stopping and model checkpoint for saving the best model
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)

# Implement TensorBoard for visualization
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)

# Train the model
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[lr_scheduler, reduce_lr, early_stopping, model_checkpoint, tensorboard]
)

# Evaluate the model on the test set
test_predictions = model.predict(X_test)

# Calculate Mean Absolute Error (MAE)
test_mae = mean_absolute_error(y_test, test_predictions)
print(f'Test Mean Absolute Error: {test_mae}')

# Calculate Mean Squared Error (MSE)
test_mse = mean_squared_error(y_test, test_predictions)
print(f'Test Mean Squared Error: {test_mse}')

# Plot the predictions vs. actual values
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual Close Price', color='blue')
plt.plot(test_predictions, label='Predicted Close Price', color='red')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.legend()
plt.show()
