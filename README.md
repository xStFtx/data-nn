## Stock Price Prediction with Neural Networks

This Python script demonstrates how to build a neural network for stock price prediction using historical price data from a CSV file. The code performs the following tasks:

1. **Data Loading**: It loads historical stock price data from a CSV file named `prices.csv`.

2. **Data Preprocessing**: The script drops unnecessary columns like 'symbol' and 'date' and then normalizes the data using Min-Max scaling.

3. **Data Splitting**: It splits the data into training and testing sets for model training and evaluation.

4. **Neural Network Model**: The neural network model is constructed using TensorFlow and Keras, consisting of input, hidden, and output layers.

5. **Model Compilation**: The model is compiled with the 'adam' optimizer and 'mean_squared_error' loss function.

6. **Model Training**: The script trains the neural network on the training data with a specified number of epochs and batch size.

7. **Model Evaluation**: It evaluates the model's performance on the testing data and prints the test loss.

8. **Making Predictions**: An example prediction is demonstrated using new data.

You can adapt this code for your own stock price prediction tasks by replacing the example data with your dataset and modifying the model architecture or training parameters as needed.
