import numpy as np
import sys

def read_matrix_with_size(filename):
    with open(filename) as f:
        data = f.read().split()
        n = int(data[0])
        matrix = np.array(data[1:], dtype=float).reshape(n, n)
        return matrix

if len(sys.argv) < 4:
    print("Incorrect number of arguments")
    sys.exit(1)


A = read_matrix_with_size(sys.argv[1])
B = read_matrix_with_size(sys.argv[2])
C_cpp = np.loadtxt(sys.argv[3])

print(f"Matrix size: {A.shape[0]}x{A.shape[1]}")


C_numpy = np.dot(A, B)


diff = np.max(np.abs(C_cpp - C_numpy))
print(f"Max diff: {diff:.2e}")

if diff < 1e-10:
    print("Correct")
else:
    print("Incorrect")