import numpy as np

# Generate a random 2x2 matrix
matrix = np.random.randint(10, size=(2, 2))
print("Randomly Generated Matrix:\n", matrix)

# Determinant
det = np.linalg.det(matrix)
print("\nDeterminant of Matrix:", det)

# Inverse (only if determinant ≠ 0)
if det != 0:
    inverse = np.linalg.inv(matrix)
    print("\nInverse of the Matrix:\n", inverse)
else:
    print("\nMatrix is singular; inverse does not exist.")

# Rank
rank = np.linalg.matrix_rank(matrix)
print("\nRank of the Matrix:", rank)

# Eigenvalues and Eigenvectors
EVal, EVect = np.linalg.eig(matrix)
print("\nEigen Values:\n", EVal)
print("\nEigen Vectors:\n", EVect)

# Flatten into 1D array
array_1D = matrix.flatten()
print("\nTransform Matrix into 1D array:", array_1D)
