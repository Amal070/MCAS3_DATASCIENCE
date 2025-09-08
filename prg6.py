import matplotlib.pyplot as plt
import numpy as np

# Generate 50 random ages between 10 and 80
ages = np.random.randint(10, 81, size=50)

# Plot histogram with bin size = 5
plt.hist(ages, bins=range(10, 85, 5), color="skyblue", edgecolor="black")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Histogram of Random Ages (10–80)")
plt.show()
