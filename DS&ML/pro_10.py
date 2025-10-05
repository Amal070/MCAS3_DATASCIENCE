import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Step 1: Load dataset
student = pd.read_csv('student_scores.csv')

# Step 2: Separate features and target
X = student.iloc[:, 0].values.reshape(-1, 1)  # Hours studied
y = student.iloc[:, 1].values                 # Scores

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Initialize the Linear Regression model
model = LinearRegression()

# Step 5: Train the model on the training data
model.fit(X_train, y_train)

# Step 6: Predict the target values for the test data
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")

# Step 8: Plot the regression line with actual data points
plt.figure(figsize=(8,5))
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.title('Simple Linear Regression')
plt.xlabel('Hours Studied')
plt.ylabel('Dependent Variable (Scores)')
plt.legend()
plt.show()

# Step 9: Plot the actual vs predicted values using bar chart
X_axis = np.arange(len(y_test))
plt.figure(figsize=(8,5))
plt.bar(X_axis - 0.2, y_test, width=0.4, label='Actual')
plt.bar(X_axis + 0.2, y_pred, width=0.4, label='Predicted')
plt.xlabel("Test Records")
plt.ylabel("Marks")
plt.title("Student Score Prediction")
plt.legend()
plt.show()
