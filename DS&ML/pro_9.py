import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 1: Load Iris dataset
iris = sns.load_dataset('iris')
df = pd.DataFrame(iris)

# Step 2: Separate features and target
X = df.iloc[:, :-1].values  # All columns except last
y = df.iloc[:, -1].values   # Last column (species)

# Optional: If using CSV file instead
# dataset = pd.read_csv('iris.csv')
# X = dataset.iloc[:, :4].values
# y = dataset['variety'].values

# Step 3: Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train Gaussian Naive Bayes classifier
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Step 5: Predict on test set
y_pred = classifier.predict(X_test)

# Step 6: Display predicted vs actual labels
print("\nPredicted Labels\n", y_pred, "\n")
print("\nActual Labels\n", y_test, "\n")

# Step 7: Evaluate the model
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("\nConfusion Matrix:\n", cm)
print("\nAccuracy:", accuracy)

# Step 8: Count mislabeled instances
mislabeled_count = (y_test != y_pred).sum()
print(f"\nNumber of mislabeled points: {mislabeled_count} out of {len(y_test)}")

# Step 9: Display actual vs predicted for all records
print("\nActual vs Predicted Labels:")
comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison_df)
