
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Create CSV file
data_dict = {
    "Fruits": ["Apples", "Bananas", "Cherries", "Oranges", "Grapes", "Mangoes"],
    "Sales": [120, 90, 60, 150, 80, 110]
}
df = pd.DataFrame(data_dict)
df.to_csv("Sales.csv", index=False)   # Creates CSV in same folder

# Step 2: Read the CSV file
data = pd.read_csv("Sales.csv")

print(data)

# Step 3: Plot Bar Chart
plt.figure(figsize=(8, 5))
plt.bar(data["Fruits"], data["Sales"], color='skyblue', edgecolor='black')
plt.title("Fruit Sales (in kg) - Bar Chart")
plt.xlabel("Fruits")
plt.ylabel("Sales (kg)")
plt.show()

# Step 4: Plot Scatter Plot
plt.figure(figsize=(8, 5))
plt.scatter(data["Fruits"], data["Sales"], color='red', marker='o', s=100)
plt.title("Fruit Sales (in kg) - Scatter Plot")
plt.xlabel("Fruits")
plt.ylabel("Sales (kg)")
plt.show()
