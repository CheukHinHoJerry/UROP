"""
# range of x: [-1,1], periodic boundary condition is assumed.
# problem : u*u_x-u_xx=0, u(-1)=c, u(1)=d

# first we obtaining training data by solving the problem on a fine grid within some interval [xj,xj+1]
# here we use N=10 for partition on coarse grid and fine grid respectively
# Let solution be in range [-1,1], consider on the interval [xi,xi+1]. First we obtains training data, randomly chosen (a,b) , range of a and b between
# [-1,1], about 1681 training data, equally spaced, with step=0.05. i.e. (-1,-1), (-1,-0.95)....(-1,0.95),(-1,1)
# (-0.95,-1), (-0.95,-0.95), ....(-0,95, 0.95), (-0.95,1).

# next we use the solution obtained above to find another 4 expected output of the NN
# Let phi_i be the solution on the interval [xi,xi+1], then notice that, phi_i satisfy:
# phi_i*phix_i-phixx_i=0, phi_i(xi)=u(xi), phi_i(xi+1)=u(xi+1).
# It is clear that phi_i depends on ui and ui+1 only.
# differentiate both side w.r.t. ui=u(xi) gives:
# (1):-yxx_i,i+phi_i*yx_i,i+y_i,i*phix_i=0, y_i,i(xi)=1, y_i,i(xi+1)=0
# similarly, consider differentiate w.r.t ui+1=u(xi+1) gives:
# (2):-yxx_{i,i+1}+phi_i*yx_{i,i+1}+y_{i,i+1}*phix_i=0, y_{i,i+1}(xi)=0, y_{i,i+1}(xi+1)=1
# for solving the above two system, notice that we can still us Dx1 and Dx2.
"""

import numpy as np
from scipy.optimize import fsolve

u_range = np.linspace(-1.5, 1.5, 61)

# number of intervals, the number of intervals of fine grid is N*N (number of points are N+1 and N^2+1 respectively)
N = 10

# step size on fine grid
h = (2 / N) / N
# assigning derivative matrix
Dx1 = np.eye(N + 1, k=1) - np.eye(N + 1, k=-1)
Dx1 = Dx1 / (2 * h)
# First and last row of Dx1 is zero for applying boundary condition
Dx1[0, :] = np.zeros(N + 1)
Dx1[N, :] = np.zeros(N + 1)

Dx2 = np.eye(N + 1, k=1) + np.eye(N + 1, k=-1) - 2 * np.eye(N + 1, k=0)
Dx2 = Dx2 / (2 * h * 2 * h)

# For applying boundary condition
Dx2[0, :] = np.hstack((-1, np.zeros(N)))
Dx2[N, :] = np.hstack((np.zeros(N), -1))

# define empty matrix and data_x for network training
target = np.zeros([len(u_range) * len(u_range), 6], dtype=float)
data_x = np.zeros([len(u_range) * len(u_range), 2], dtype=float)
sol = np.zeros([len(u_range) * len(u_range), N + 1], dtype=float)


# define FineFunc that value of r0 changes each loop in order to apply the boundary condition
def fineFunc(u):
    return u * (Dx1 @ u) - Dx2 @ u - r0


def partialuFunc1(y):
    return -Dx2 @ y + (Dx1 @ y) * fineSol + (Dx1 @ fineSol) * y - r1


def partialuFunc2(y):
    return -Dx2 @ y + (Dx1 @ y) * fineSol + (Dx1 @ fineSol) * y - r2


count = 0
for i in range(len(u_range)):
    for j in range(len(u_range)):
        a = u_range[i]
        b = u_range[j]
        data_x[i * len(u_range) + j, 0] = a
        data_x[i * len(u_range) + j, 1] = b
        count = count + 1
        # initializing a vector for implementing boundary condition (where the boundary condition is different
        # for each loop
        r0 = np.hstack([a, np.zeros(N - 1), b])

        # solving the system with some initial guess
        fineSol = fsolve(fineFunc, np.linspace(a, b, N + 1))
        print("The solution for the ", i * len(u_range) + j, " th data is: ", fineSol)
        print("The error vector for the ", i * len(u_range) + j, " th data is: ", fineFunc(fineSol))

        # initialize vector for implementing boundary condition
        r1 = np.hstack([1, np.zeros(N - 1), 0])
        r2 = np.hstack([0, np.zeros(N - 1), 1])

        partialuSol1 = fsolve(partialuFunc1, np.linspace(1, 0, N + 1))
        partialuSol2 = fsolve(partialuFunc2, np.linspace(0, 1, N + 1))

        # first two entries of a particular row stores the derivative of end points, the last four entries are
        # dy_{i,i}/dx (xi) , dy_{i,i}/dx (xi+1), dy_{i,i+1}/dx (xi) , dy_{i,i+1}/dx (xi+1),
        target[i * len(u_range) + j, 0] = (fineSol[1] - fineSol[0]) / h
        target[i * len(u_range) + j, 1] = (fineSol[N] - fineSol[N - 1]) / h
        target[i * len(u_range) + j, 2] = (partialuSol1[1] - partialuSol1[0]) / h
        target[i * len(u_range) + j, 3] = (partialuSol1[N] - partialuSol1[N - 1]) / h
        target[i * len(u_range) + j, 4] = (partialuSol2[1] - partialuSol2[0]) / h
        target[i * len(u_range) + j, 5] = (partialuSol2[N] - partialuSol2[N - 1]) / h

print(target)
print(count)
np.savetxt('target2_10interval.txt', target, delimiter=',')
np.savetxt('data_x2_10interval.txt', data_x, delimiter=',')