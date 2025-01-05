# Stock Trend Predictor Using LSTM

This project predicts stock trends using an LSTM (Long Short-Term Memory) model. The model is built with Keras and TensorFlow to analyze historical stock data and predict future trends in stock prices. Users can interact with the system through a Flask web application to input stock tickers, view predictions, and visualize trends.

## Features

- **Stock Data Fetching**: The app allows users to input a stock ticker (e.g., AAPL, TSLA), and it will automatically fetch historical data from Yahoo Finance using the `yfinance` library.
  
- **Data Preprocessing**: The stock data is processed to focus on key features like 'Close', 'High', 'Low', and 'Volume'. The data is then normalized using the `MinMaxScaler` to prepare it for the LSTM model.

- **LSTM Model**: The deep learning model is constructed using Keras, with LSTM layers designed to learn stock price patterns. The model is trained on 60 days of stock data to predict the closing price for the next day.

- **Trend Prediction and Visualization**: The model predicts stock price trends, which are compared to actual prices. Users can view:
  - Actual vs Predicted Stock Prices
  - Original vs Predicted Prices
  - Closing Prices over Time

- **Web Interface**: The app is powered by Flask and provides an easy-to-use interface for entering stock tickers and viewing the model's predictions.

## Technologies Used

- **Python Libraries**:
  - `numpy`, `pandas` – For data manipulation and analysis.
  - `matplotlib` – For plotting and visualizing stock data and predictions.
  - `yfinance` – To fetch historical stock data from Yahoo Finance.
  - `sklearn.preprocessing.MinMaxScaler` – For normalizing the data.
  - `keras` – For building and training the LSTM model.
  - `flask` – For creating the web interface.

- **Model Architecture**: The LSTM model has two LSTM layers with Dropout for regularization and Dense layers for final prediction.

## Installation

To run this project locally, follow these steps:

1. Clone the repository:

   ```
   git clone https://github.com/yourusername/stock-trend-predictor.git
   ```
2. Navigate to the project folder:
   ```
    cd stock-trends-predictor
   ```
   
3. Create and activate a virtual environment:
- On Windows:
  ```
  python -m venv venv
  venv\Scripts\activate
  ```
- On macOS/Linux:
  ```
  python3 -m venv venv
  source venv/bin/activate
  ```

4. Install the required Python libraries:
 ```
   pip install numpy pandas matplotlib yfinance scikit-learn keras flask
```
5. Run the Flask application:
 ```
python stock_prediction.py
```
6. Open your browser and go to:
   ```
   http://127.0.0.1:5000/
   ```

   
## Usage

1. Enter a stock ticker (e.g., AAPL, TSLA) in the input field on the homepage.
2. Click "Submit" to load the stock data, train the LSTM model, and generate predictions.
3. The page will display:
- Actual vs Predicted Stock Prices
- A comparison of Original vs Predicted Prices
- The stock’s closing prices over time
- A summary of key statistics (Open, Close, High, Low, Volume)

## Future Enhancements

- **Model Improvements**: Enhance prediction accuracy by incorporating additional features like moving averages or sentiment analysis.
- **Deployment**: Host the application on cloud platforms such as Heroku or AWS for public access.
- **Error Handling**: Improve error messages and validation for incorrect tickers or data issues.

## Acknowledgements

- This project uses the `yfinance` library to retrieve stock data and `keras` for building and training the LSTM model.
- Thanks to the contributors of Flask, Matplotlib, and other Python libraries used in this project.

   
