import numpy as np

# --- Create Matrix U from user input ---
m = int(input("Enter row size: "))
n = int(input("Enter column size: "))

print("Enter values in single line (space-separated):")
entries = list(map(int, input().split()))
U = np.array(entries).reshape(m, n)

print("\nMatrix U:")
print(U)

# --- Generate random matrix Y ---
m = 5
Y = np.random.rand(1, m)
print("\nMatrix Y:")
print(Y)

# --- Input for p and create random vector B1 ---
p = int(input("\nEnter value for p: "))
B1 = np.random.rand(p, 1)
print("\nVector B1:")
print(B1)

# --- Create zero matrix W1 ---
W1 = np.zeros((1, p))
print("\nMatrix W1:")
print(W1)

# --- Generate random scalar B2 ---
B2 = np.random.rand(1)
print("\nScalar B2:")
print(B2)

# --- Perform dot product and addition ---
Z1 = np.dot(W1, U) + B1
print("\nMatrix Z1:")
print(Z1)