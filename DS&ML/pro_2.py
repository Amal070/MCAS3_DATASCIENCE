# Required package: numpy
# Install using: pip install numpy

import numpy as np

# Define two 2x2 matrices
ar1 = np.array([[1, 2],
                [3, 4]])

ar2 = np.array([[5, 6],
                [7, 8]])

print("Add two matrices:")
print(np.add(ar1, ar2))

print("\nSubtract two matrices:")
print(np.subtract(ar1, ar2))

print("\nMultiply individual elements of the matrices:")
print(np.multiply(ar1, ar2))

print("\nDivide the elements of the matrices:")
print(np.divide(ar1, ar2))

print("\nPerform matrix multiplication:")
print(np.dot(ar1, ar2))

print("\nDisplay the transpose of the matrices:")
print("Transpose of ar1:\n", ar1.transpose())
print("Transpose of ar2:\n", ar2.transpose())

print("\nSum of diagonal elements of the matrices:")
print("Trace of ar1:", np.trace(ar1))
print("Trace of ar2:", np.trace(ar2))
