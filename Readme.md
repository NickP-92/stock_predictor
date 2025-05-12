# 📈 Stock Price Predictor (Recursive Linear Regression)

This project trains a simple linear regression model to predict stock prices using historical closing data.  
It simulates how the model would behave in a real-world, recursive scenario — where each prediction is used to generate the next — and compares this output to real historical performance.

---

## 🚀 Features

- Fetches stock data from Yahoo Finance
- Trains on user-defined number of years of data
- Validates prediction accuracy using:
  - A traditional test split (20%)
  - A recursive forecast over the full historical period
- Visualizes:
  - Full historical data
  - Training data
  - Test set and model predictions
  - Recursive prediction timeline

---

## 🧠 How It Works

1. Downloads X years of historical `Close` prices for the specified stock.
2. Creates a target column (`Target`) by shifting the `Close` column by -1 (next day).
3. Splits the dataset (80% train / 20% test).
4. Trains a `LinearRegression` model on the training set.
5. Predicts the test set to evaluate accuracy.
6. Starts from the first historical price and recursively predicts all remaining days.
7. Plots everything for easy comparison.

---

## 📦 Requirements

Install with:

```bash
pip install -r requirements.txt
```

## 🛠 Usage
```bash
py stock_predictor.py SYMBOL YEARS
py stock_predictor.py AAPL 5
```

## 📌 Notes
This is a very simple model using only the previous day's closing price.

Accuracy drops off over time in recursive mode due to error accumulation.

Intended for educational use, not actual investment guidance.