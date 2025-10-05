import numpy as np
import matplotlib.pyplot as plt

#Generate 50 random ages between 10 and 80
ages = np.random.randint(10, 81, 50)

print("Generated ages:\n", ages)

# Plot histogram with bin size of 5
plt.figure(figsize=(6,4))
plt.hist(ages, bins=5, color='blue', edgecolor='black')
plt.title("Age Distribution Histogram")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()
