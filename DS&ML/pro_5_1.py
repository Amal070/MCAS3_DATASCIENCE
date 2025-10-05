# Import required packages
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Create a dictionary with fruits data
fruits_data = {
    "Fruits": ["Apple", "Banana", "Mango", "Orange", "Grapes"],
    "Sales": [50, 30, 70, 45, 60]
}

# Step 2: Convert dictionary to DataFrame
df = pd.DataFrame(fruits_data)

# Step 3: Save DataFrame to CSV
csv_file = "fruits_sales.csv"
df.to_csv(csv_file, index=False)
print(f"Data saved to {csv_file}")

# Step 4: Read CSV file back
data = pd.read_csv(csv_file)
print("Data read from CSV:")
print(data)

# Step 5: Plot Bar Chart
plt.figure(figsize=(6,4))
plt.bar(data['Fruits'], data['Sales'], color='green', edgecolor='black')
plt.title("Fruit Sales (Bar Chart)")
plt.xlabel("Fruits")
plt.ylabel("Sales (kg)")
plt.show()

# Step 6: Plot Scatter Plot
plt.figure(figsize=(6,4))
plt.scatter(data['Fruits'], data['Sales'], color='red', s=100)  # s=100 for bigger markers
plt.title("Fruit Sales (Scatter Plot)")
plt.xlabel("Fruits")
plt.ylabel("Sales (kg)")
plt.show()
