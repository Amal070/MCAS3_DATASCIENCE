import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
# Load dataset
housing = pd.read_csv("Housing.csv")
# Define target variable and features
y = housing['price']
X = housing[['area','bedrooms','bathrooms','stories']]
# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=42)
# Create Linear Regression model
model = LinearRegression()
# Train the model
model.fit(X_train, y_train)
# Predictions
y_pred = model.predict(X_test)
# Model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nMean Squared Error:", mse)
print("R-squared:", r2)
# Display coefficients with feature names
coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient':
model.coef_})
print("\nCoefficients:")
print(coefficients)
print("\nIntercept:", model.intercept_)
# Select the features and target for pairplot
sns.pairplot(
housing,
x_vars=['area', 'bedrooms', 'stories'],
y_vars='price',
height=5,
aspect=1,
kind='scatter'
)
plt.suptitle("Scatter plots of Features vs Price", y=1.02)
plt.show()