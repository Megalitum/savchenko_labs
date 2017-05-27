import numpy as np
from scipy.optimize import linprog

with np.load('impc.npz') as f:
    l, u = f['a_low'], f['a_up']

n = len(l)
c = np.ones(2*n)
c[n:] = -1
A11 = np.repeat(-np.eye(n), n-1, axis=0)
mask1 = np.tile(np.eye(n), (n,1))[np.arange(n*n) % (n+1) != 0 ]
A12 = np.repeat(l, n-1, axis=0) * mask1
A1 = np.hstack((A11,A12))
A21 = np.repeat(np.eye(n), n-1, axis=0)
A22 = -np.repeat(u, n-1, axis=0) * mask1
A2 = np.hstack((A21,A22))
A3 = -np.hstack((np.eye(n), np.ones((n,n)) - np.eye(n)))
A4 = np.hstack((np.ones((n,n)) - np.eye(n), np.eye(n)))
A5 = np.hstack((np.eye(n), -np.eye(n)))
A6 = np.hstack((-np.eye(n), np.zeros((n,n))))
A = np.vstack((A1,A2,A3,A4,A5,A6))
b = np.zeros((2*n*(n+1), 1))
b[2*n*(n-1):2*n*(n-1)+n] = -1
b[2*n*n - n:2*n*n] = 1
np.savetxt('a.txt', A, '%2.4f')
np.savetxt('b.txt', b, '%2.4f')
x = linprog(c, A, b)
print(x)
