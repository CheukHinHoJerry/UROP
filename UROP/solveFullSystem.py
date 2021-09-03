import numpy as np
from scipy.optimize import fsolve
import tensorflow as tf

# number of intervals, the number of intervals of fine grid is N*N^2 (number of points are N+1 and N^2+1 respectively)
N = 10
# step size on fine grid
h = (2 / N) / (N**3)
# assigning derivative matrix
Dx1 = np.eye(N**4 + 1, k=1) - np.eye(N**4 + 1, k=-1)
Dx1 = Dx1 / (2 * h)
# First and last row of Dx1 is zero for applying boundary condition
Dx1[0, :] = np.zeros(N**4 + 1)
Dx1[N**4, :] = np.zeros(N**4 + 1)

Dx2 = np.eye(N**4 + 1, k=1) + np.eye(N**4 + 1, k=-1) - 2 * np.eye(N**4 + 1, k=0)
Dx2 = Dx2 / (h ** 2)

# For applying boundary condition
Dx2[0, :] = np.hstack((-1, np.zeros(N**4)))
Dx2[N**3, :] = np.hstack((np.zeros(N**4), -1))

# define FineFunc that value of r0 changes each loop in order to apply the boundary condition
def solveFullSystem(a,b):
    r0 = np.hstack([a, np.zeros(N ** 4 - 1), b])
    def fullSystemFunc(u):
        return u * (Dx1 @ u) - Dx2 @ u - r0
    return fsolve(fullSystemFunc,np.linspace(a,b,N**4+1))