import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

iris = pd.read_csv("iris (2).csv")

print("Shape of the dataset (rows, columns):", iris.shape)
print("Size of the dataset (total elements):", iris.size)
print("\nFirst 5 rows of the dataset:")
print(iris.head())
print("\nLast 5 rows of the dataset:")
print(iris.tail())
print("\nNumber of samples for each variety:")
print(iris["variety"].value_counts())
print("\nDescriptive statistics:")
print(iris.describe())
print("\nMedian values of each column:")
print(iris.median(numeric_only=True))
print("\nStandard deviation of each column:")
print(iris.std(numeric_only=True))
print("\nConcise summary of dataset:")
print(iris.info())

sns.pairplot(iris, hue="variety", kind="scatter", diag_kind="hist")
plt.suptitle("Pairplot of Iris Dataset", y=1.02, color="white")
plt.style.use("dark_background")

sns.displot(iris["sepal_length"], bins=10, color="g")
plt.title("Distribution of Sepal Length", fontsize=10, color="white")

plt.show()