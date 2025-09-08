import matplotlib.pyplot as plt

# Fruits and Sales (in kg)
fruits = ['Apples', 'Bananas', 'Cherries', 'Dates', 'Grapes']
sales = [10, 20, 30, 40, 50]

# Bar Chart
plt.bar(fruits, sales,color='yellow',edgecolor='black')
plt.title('Fruit Sales (kg) - Bar Chart')
plt.xlabel('Fruits')
plt.ylabel('Sales')
plt.show()

# Scatter Plot
plt.scatter(fruits, sales,color='red',edgecolor='black')
plt.title('Fruit Sales (kg) - Scatter Plot')
plt.xlabel('Fruits')
plt.ylabel('Sales')
plt.show()
