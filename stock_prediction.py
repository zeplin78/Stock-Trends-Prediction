import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io
import base64
from flask import Flask, render_template, request
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping

# Load and preprocess the stock data
def load_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def preprocess_data(data):
    feature_columns = ['Close', 'High', 'Low', 'Volume']
    features = data[feature_columns].values
    close_prices = features[:, 0].reshape(-1, 1)  # Close prices, reshaped correctly

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(features)
    scaled_close = scaler.fit_transform(close_prices)

    # Create sequences for LSTM model
    X, y = [], []
    for i in range(60, len(scaled_close)):  # Using 60 timesteps for training
        X.append(scaled_features[i - 60:i])  # 60 timesteps of all features
        y.append(scaled_close[i, 0])  # Target is the next close price
    X, y = np.array(X), np.array(y)

    # Reshape X to (samples, timesteps, features)
    X = X.reshape((X.shape[0], X.shape[1], 4))  # 60 timesteps and 4 features (Close, High, Low, Volume)

    return X, y, scaler

# Build and train the LSTM model
def build_lstm_model(input_shape):
    model = Sequential([ 
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)  # Output layer
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Flask application
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ticker = request.form['ticker']
        start_date = "2024-01-01"
        end_date = "2025-12-31"

        try:
            # Load and preprocess stock data
            stock_data = load_data(ticker, start_date, end_date)
            X, y, scaler = preprocess_data(stock_data)

            # Split data into training and testing sets
            split_idx = int(0.8 * len(X))
            X_train, y_train = X[:split_idx], y[:split_idx]
            X_test, y_test = X[split_idx:], y[split_idx:]

            # Build and train the LSTM model
            model = build_lstm_model(X_train.shape[1:])
            early_stopping = EarlyStopping(monitor='loss', patience=5)
            model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1, callbacks=[early_stopping])

            # Make predictions
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            # Scale back the predictions
            train_pred = scaler.inverse_transform(train_pred)
            test_pred = scaler.inverse_transform(test_pred)
            actual_prices = scaler.inverse_transform(y.reshape(-1, 1))

            # Prepare dates
            dates = stock_data.index
            train_dates = dates[60:60 + len(train_pred)]
            test_dates = dates[-len(test_pred):]

            # Plot the results
            plt.figure(figsize=(14, 6))
            plt.plot(dates, stock_data['Close'].values, label='Actual Prices', color='blue', linewidth=2)
            plt.plot(train_dates, train_pred, label='Train Predictions', color='red', linestyle='--')
            plt.plot(test_dates, test_pred, label='Test Predictions', color='green', linestyle='--')
            plt.title(f'Stock Price Prediction for {ticker}')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()

            # Save the plot as an image and convert to base64
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()
            plt.close()

            # Plot Original vs Predicted
            plt.figure(figsize=(14, 6))
            plt.plot(actual_prices, label='Actual Prices', color='blue', linewidth=2)
            plt.plot(np.concatenate([train_pred, test_pred]), label='Predicted Prices', color='orange', linestyle='--')
            plt.title(f'Original vs Predicted Prices for {ticker}')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.legend()

            # Save this plot as an image
            img2 = io.BytesIO()
            plt.savefig(img2, format='png')
            img2.seek(0)
            plot_url2 = base64.b64encode(img2.getvalue()).decode()
            plt.close()

            # Plot Closing Price vs Time
            plt.figure(figsize=(14, 6))
            plt.plot(dates, stock_data['Close'], label='Closing Prices', color='purple')
            plt.title(f'Closing Price vs Time for {ticker}')
            plt.xlabel('Date')
            plt.ylabel('Closing Price')
            plt.legend()

            # Save this plot as an image
            img3 = io.BytesIO()
            plt.savefig(img3, format='png')
            img3.seek(0)
            plot_url3 = base64.b64encode(img3.getvalue()).decode()
            plt.close()

            # Summary statistics with transposed data including 'Open' and 'Adj Close'
            summary = stock_data[['Close', 'High', 'Low', 'Volume', 'Open']].describe().transpose()

            # Round the values in the summary to 3 decimal places
            summary = summary.round(3)

            # Now transpose the DataFrame so that the axis are swapped
            summary_transposed = summary.T

            # Adding the currency label to the side as part of the table
            summary_transposed['Currency'] = 'USD'

            # Render the transposed table
            stock_table = summary_transposed.to_html(classes='table table-striped')

            return render_template('index.html', plot_url=plot_url, plot_url2=plot_url2, plot_url3=plot_url3, stock_table=stock_table)
        except Exception as e:
            return render_template('index.html', error=str(e), plot_url=None, plot_url2=None, plot_url3=None, stock_table=None)

    return render_template('index.html', plot_url=None, plot_url2=None, plot_url3=None, stock_table=None)

if __name__ == "__main__":
    app.run(debug=True)