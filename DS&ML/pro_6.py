# Import required packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
iris = pd.read_csv("iris.csv")

# Step 2: Display dataset info
print("Shape of the Dataset:", iris.shape)
print("\nFirst five rows:")
print(iris.head())
print("\nLast five rows:")
print(iris.tail())
print("\nSize of the Dataset:", iris.size)
print("\nNumber of samples available for each Variety:")
print(iris["variety"].value_counts())
print("\nDescription of the dataset:")
print(iris.describe())

# Step 3: Pairplot for visualizing relationships
sns.set(style="darkgrid")
sns.pairplot(iris, hue="variety", kind="scatter", diag_kind="hist")
plt.show()

# Step 4: Distribution plot for Sepal Length 
plt.style.use("seaborn-white")  # change background to white
sns.displot(iris["sepal_length"], bins=10, color="green")
plt.title("Distribution of Sepal Length", fontsize=12, color="black")  # change text color to black
plt.xlabel("Sepal Length", color="black")
plt.ylabel("Frequency", color="black")
plt.show()

