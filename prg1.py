import numpy as np

# Create a 2D NumPy array
matrix_2D = np.array([
    [2, 4, 6, 8],
    [1, 3, 5, 7],
    [3, 6, 9, 12],
    [1, 5, 8, 10]
])

# Display the matrix
print("Matrix:")
print(matrix_2D)

# Display all elements excluding the first row
print("\nDisplay all elements excluding the first row:")
print(matrix_2D[1:4, :])

# Display all elements excluding the last row
print("\nDisplay all elements excluding the last row:")
print(matrix_2D[0:3, :])

# Display all elements of 1st and 2nd columns in 2nd and 3rd rows
print("\nDisplay all the elements of 1st and 2nd columns in 2nd and 3rd rows:")
print(matrix_2D[1:3, 0:2])

# Display the elements of 2nd and 3rd columns
print("\nDisplay the elements of 2nd and 3rd columns:")
print(matrix_2D[:, 1:3])

# Display 2nd and 3rd element of 1st row
print("\nDisplay 2nd and 3rd element of 1st row:")
print(matrix_2D[0, 1:3])
