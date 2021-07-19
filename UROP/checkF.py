'''This file is for checking F by discretization of coarse grid
and discretization of fine grid only'''

import numpy as np
from scipy.optimize import fsolve

u_range = np.linspace(-1.5, 1.5, 61)

# number of intervals, the number of intervals of fine grid is N*N^2 (number of points are N+1 and N^2+1 respectively)
N = 10

# step size on fine grid
h = (2 / N) / (N**3)
# assigning derivative matrix
Dx1 = np.eye(N**3 + 1, k=1) - np.eye(N**3 + 1, k=-1)
Dx1 = Dx1 / (2 * h)
# First and last row of Dx1 is zero for applying boundary condition
Dx1[0, :] = np.zeros(N**3 + 1)
Dx1[N**3, :] = np.zeros(N**3 + 1)

Dx2 = np.eye(N**3 + 1, k=1) + np.eye(N**3 + 1, k=-1) - 2 * np.eye(N**3 + 1, k=0)
Dx2 = Dx2 / (h ** 2)

# For applying boundary condition
Dx2[0, :] = np.hstack((-1, np.zeros(N**3)))
Dx2[N**3, :] = np.hstack((np.zeros(N**3), -1))


# define FineFunc that value of r0 changes each loop in order to apply the boundary condition
def fineFunc(u):
    return u * (Dx1 @ u) - Dx2 @ u - r0


# fine grid discretization

fine_left_deri = np.zeros([len(u_range) - 1], dtype=float)
fine_right_deri = np.zeros([len(u_range) - 1], dtype=float)

for i in range(len(u_range)-1):
        a = u_range[i]
        b = u_range[i+1]
        print(a, b)
        # initializing a vector for implementing boundary condition (where the boundary condition is different
        # for each loop
        r0 = np.hstack([a, np.zeros(N**3 - 1), b])

        # solving the system with some initial guess
        fineSol = fsolve(fineFunc, np.linspace(a, b, N**3 + 1))
        fineSol = np.array(fineSol)

        # derivative within the fine solution
        # compare the derivative of different fine grids, seems weird
        print((fineSol[1:] - fineSol[:100]) / h)

        # initialize vector for implementing boundary condition
        r1 = np.hstack([1, np.zeros(N**3 - 1), 0])
        r2 = np.hstack([0, np.zeros(N**3 - 1), 1])

        # first two entries of a particular row stores the derivative of end points, the last four entries are
        # dy_{i,i}/dx (xi) , dy_{i,i}/dx (xi+1), dy_{i,i+1}/dx (xi) , dy_{i,i+1}/dx (xi+1),
        fine_right_deri[i] = (fineSol[1] - fineSol[0]) / h
        fine_left_deri[i] = (fineSol[N**3] - fineSol[N**3 - 1]) / h
        print(i, fineSol)

fine_F = np.sum((fine_right_deri[1:] - fine_left_deri[:99])**2)
print(fine_F)   # 1.0176217115192623

# coarse grid discretization

N = 10
# boundary condition for the target problem, u(-1)=a, u(1)=b
a = 0.6
b = -0.8
r0 = np.hstack([a, np.zeros(N * N**2 - 1), b])
# step size on fine grid
h = (2 / N) / (N**2)
# assigning derivative matrix
Dx1 = np.eye(N * N**2 + 1, k=1) - np.eye(N * N**2 + 1, k=-1)
Dx1 = Dx1 / (2 * h)
# First and last row of Dx1 is zero for applying boundary condition
Dx1[0, :] = np.zeros(N * N**2 + 1)
Dx1[N * N ** 2, :] = np.zeros(N * N**2 + 1)

Dx2 = np.eye(N * N**2 + 1, k=1) + np.eye(N * N**2 + 1, k=-1) - 2 * np.eye(N * N**2 + 1, k=0)
Dx2 = Dx2 / (h ** 2)

# For applying boundary condition
Dx2[0, :] = np.hstack((-1, np.zeros(N * N**2)))
Dx2[N * N**2, :] = np.hstack((np.zeros(N * N**2), -1))


def coarseFunc(u):
    return u * (Dx1 @ u) - Dx2 @ u - r0


# solving the system with some initial guess
sol = fsolve(coarseFunc, np.ones(N * N ** 2 + 1))
# print("The solution for is", sol)

coarse_left_deri = np.zeros(N-1)
coarse_right_deri = np.zeros(N-1)
for i in range(1, N):
    coarse_left_deri[i-1] = (sol[i*N**2]-sol[i*N**2-1])/h
    coarse_right_deri[i-1] = (sol[i*N**2+1]-sol[i*N**2])/h


coarse_F = np.sum((coarse_left_deri-coarse_right_deri)**2)

