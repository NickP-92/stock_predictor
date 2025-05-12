import argparse
import datetime
from dataclasses import dataclass
from matplotlib import pyplot as plt
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

@dataclass
class TrainingResult:
    model: LinearRegression
    df: any
    X_train: any
    X_test: any
    y_train: any
    y_test: any
    y_pred: any

"""
Downloads historical stock data and trains a Linear Regression model
on past data to predict the next day's closing price.
Returns a TrainingResult dataclass containing all relevant data.
"""
def train_model(symbol: str, start_date: datetime.date, end_date: datetime.date) -> TrainingResult:
    df = yf.download(symbol, start=start_date, end=end_date)
    if df.empty:
        print('No data found. Please check the stock symbol and the date range.')
        return None

    df = df[['Close']].copy()
    df['Target'] = df['Close'].shift(-1)
    df.dropna(inplace=True)

    X = df[['Close']]
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return TrainingResult(model, df, X_train, X_test, y_train, y_test, y_pred)

"""
Generates predictions for the test set period (simulating recursive prediction).
"""
def predict_recursive_from_start(model: LinearRegression, df) -> tuple[list, list]:
    predictions = []
    dates = []

    current_price = float(df['Close'].iloc[0])
    for i in range(1, len(df)):
        next_price = model.predict([[current_price]])[0]
        predictions.append(next_price)
        dates.append(df.index[i])
        current_price = next_price

    return dates, predictions

"""
Plots historical price data, training and test sets, test predictions,
and recursive predictions over the test range.
"""
def plot_predictions(result: TrainingResult, validation_dates, validation_prices):
    df = result.df
    plt.figure(figsize=(14, 6))

    plt.plot(df.index, df['Close'], label='Full History', color='lightgray')
    plt.plot(result.X_train.index, result.y_train, label='Train Actual', color='blue')
    plt.plot(result.X_test.index, result.y_test, label='Test Actual', color='green')
    plt.plot(result.X_test.index, result.y_pred, label='Test Predicted', linestyle='--', color='orange')
    plt.plot(validation_dates, validation_prices, label='Recursive Prediction', linestyle='--', color='red')

    plt.title("Stock Price Prediction: History, Backtest, and Validation")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate stock price prediction using linear regression.")
    parser.add_argument("symbol", help="Stock ticker symbol (e.g., AAPL)")
    parser.add_argument("train_years", type=int, help="Number of years of data to use for training")
    args = parser.parse_args()

    today = datetime.date.today()
    start_date = today - datetime.timedelta(days=args.train_years * 365)

    result = train_model(args.symbol, start_date, today)

    if result:
        validation_dates, validation_prices = predict_recursive_from_start(result.model, result.df)
        plot_predictions(result, validation_dates, validation_prices)
