# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Load dataset
dataset = pd.read_csv("iris.csv")

# Step 2: Separate features and target
X = dataset.iloc[:, :-1].values  # Features: sepal/petal measurements
y = dataset.iloc[:, -1].values   # Target: variety

# Step 3: Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create KNN classifier and train
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

# Step 5: Predict on test set
y_pred = classifier.predict(X_test)

# Step 6: Evaluate model
print("Classification Report:\n")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Step 7: Compare real vs predicted values
df_results = pd.DataFrame({'Real Values': y_test, 'Predicted Values': y_pred})
print("\nComparison of Real vs Predicted values:\n", df_results)

# Step 8: Predict class for a new test point
new_test_point = np.array([[5.1, 3.5, 1.4, 0.2]])
prediction = classifier.predict(new_test_point)
print(f"\nPredicted class for {new_test_point[0]}: {prediction[0]}")
