import random
import matplotlib.pyplot as plt

# Generate 50 random ages between 10 and 80
ages = [random.randint(10, 80) for _ in range(50)]

# Histogram with bin size of 5
plt.hist(ages, bins=5,color='orange',edgecolor='black')
plt.title('Random Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()
