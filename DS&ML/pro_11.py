import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load dataset
housing = pd.read_csv("Housing.csv")

# Step 2: Define target variable and features
y = housing['price']
X = housing[['area', 'bedrooms', 'bathrooms', 'stories']]

# Step 3: Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create Linear Regression model
model = LinearRegression()

# Step 5: Train the model
model.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Squared Error:", mse)
print("R-squared:", r2)

# Step 8: Display coefficients with feature names
coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
print("\nCoefficients:")
print(coefficients)
print("\nIntercept:", model.intercept_)

# Step 9: Visualize features vs price using scatter plots
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
