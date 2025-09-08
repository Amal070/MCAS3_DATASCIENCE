import numpy as np

arr1 = np.array([[1,2],
                 [5,6]])
arr2 = np.array([[9,10],
                  [13,14]])
print(np.add(arr1,arr2))
print()
print(np.subtract(arr1,arr2))
print()
print(np.multiply(arr1,arr2))
print()
print(np.divide(arr1,arr2))
print()
print(np.dot(arr1,arr2))
print()
print(arr1.transpose())
print()
print(arr1.trace())

OUTPUT:
[[10 12]
 [18 20]]

[[-8 -8]
 [-8 -8]]

[[ 9 20]
 [65 84]]

[[0.11111111 0.2       ]
 [0.38461538 0.42857143]]

[[ 35  38]
 [123 134]]

[[1 5]
 [2 6]]

7

