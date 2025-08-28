# Using california_housing built-in dataset in scikit-learn
from sklearn.datasets import fetch_california_housing
# Exploratory data analysis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Preprocessing
from sklearn.model_selection import train_test_split
# Build Model
from sklearn.linear_model import LinearRegression
# Evaluate Model
from sklearn.metrics import mean_squared_error, r2_score

# Using california_housing built-in dataset in scikit-learn
data = fetch_california_housing()

# Data analysis
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Price'] = data.target
sns.pairplot(df.sample(500), diag_kind="kde")
plt.show()

# preprocessing
x = df.drop('Price', axis=1)
y = df['Price']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Build Model
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# Evaluate Model
print("MSE", mean_squared_error(y_test, y_pred))
print("R squared", r2_score(y_test, y_pred))

# visualize predictions
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("House Price Prediction")
plt.show()