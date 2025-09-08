import numpy as np
ar_name=np.array([[1,2,3,4],[5,6,7,8],
                  [9,10,11,12],
                  [13,14,15,16]])
print(ar_name)
print()
print(ar_name[1:, :])
print()
print(ar_name[:, 0:3])
print()
print(ar_name[1:3,0:2])
print()
print(ar_name[ : ,1:3])
print()
print(ar_name[1:3, : ])

OUTPUT:
[[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]
 [13 14 15 16]]

[[ 5  6  7  8]
 [ 9 10 11 12]
 [13 14 15 16]]

[[ 1  2  3]
 [ 5  6  7]
 [ 9 10 11]
 [13 14 15]]

[[ 5  6]
 [ 9 10]]

[[ 2  3]
 [ 6  7]
 [10 11]
 [14 15]]

[[ 5  6  7  8]
 [ 9 10 11 12]]

