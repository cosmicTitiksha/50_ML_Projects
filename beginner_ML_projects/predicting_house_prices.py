# imports
from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

# load dataset
data = fetch_california_housing()

# create DataFrame and add target column
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Price'] = data.target

# ---- Quick EDA (set to False if you don’t want blocking plots) ----
DO_PLOTS = False
if DO_PLOTS:
    sns.pairplot(df.sample(500, random_state=42), diag_kind="kde")
    plt.show()

# features and target
X = df.drop('Price', axis=1)
y = df['Price']

# train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train an unscaled linear regression (baseline)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate unscaled model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f"Unscaled LinearRegression -> MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

# Visualize actual vs predicted (optional)
if DO_PLOTS:
    plt.figure(figsize=(8,6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle='--')  # y = x reference
    plt.xlabel("Actual Price (100k USD)")
    plt.ylabel("Predicted Price (100k USD)")
    plt.title("Actual vs Predicted — Unscaled LinearRegression")
    plt.show()

# Build and train a pipeline (StandardScaler + LinearRegression) — recommended
pipeline = make_pipeline(StandardScaler(), LinearRegression())
pipeline.fit(X_train, y_train)

# Evaluate pipeline
y_pred_pipe = pipeline.predict(X_test)
mse_pipe = mean_squared_error(y_test, y_pred_pipe)
rmse_pipe = np.sqrt(mse_pipe)
r2_pipe = r2_score(y_test, y_pred_pipe)
print(f"Scaled Pipeline -> MSE: {mse_pipe:.4f}, RMSE: {rmse_pipe:.4f}, R2: {r2_pipe:.4f}")

# Helper: predict from interactive prompt
def ask_and_predict(pipeline, X_train_local=None):
    """
    Interactive prompt: asks user to enter each feature value.
    If you press Enter without typing, it will use the training mean.
    """
    if X_train_local is None:
        X_train_local = X_train

    sample = {}
    print("\nEnter values for each feature (press Enter to use dataset mean):\n")
    for f in data.feature_names:
        s = input(f"{f}: ").strip()
        if s == "":
            sample[f] = float(X_train_local[f].mean())
        else:
            sample[f] = float(s)

    # build DataFrame
    x_custom = pd.DataFrame([sample], columns=data.feature_names)

    # predict
    pred_100k = pipeline.predict(x_custom)[0]
    pred_usd = pred_100k * 100_000
    print(f"\nPredicted median house value ≈ ${pred_usd:,.0f}\n")

# ---- Run interactive predictor ----
ask_and_predict(pipeline)
