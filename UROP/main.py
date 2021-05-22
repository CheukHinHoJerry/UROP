import numpy as np
import sympy as sp
from scipy.optimize import fsolve

a = -2
b = -3

# range of x: [-1,1], periodic boundary condition is assumed.
# problem : u*u_x-u_xx=0, u(-1)=a, u(1)=b


# first we solve solution on coarse grid, using finite difference scheme, 10 nodal points
N = 10
h = 2 / N
# assigning derivative matrix
Dx1 = np.eye(N, k=1) - np.eye(N, k=-1)
Dx1 = Dx1 / (2 * h)
Dx1[0, :] = np.zeros(N)
Dx1[N - 1, :] = np.zeros(N)
Dx2 = np.eye(N, k=1) + np.eye(N, k=-1) - 2 * np.eye(N, k=0)
Dx2 = Dx2 / ((2 * h) * (2 * h))
Dx2[0, :] = np.zeros(N)
Dx2[N - 1, :] = np.zeros(N)
Dx2[0, 0] = -1
Dx2[N - 1, N - 1] = -1

#initializing a vector for implementing boundary condition
r = np.hstack([a, np.zeros(N - 2), b])


def CoarseFunc(u):
    return u * (Dx1 @ u) - Dx2 @ u - r


coarseSol = fsolve(CoarseFunc, np.array([10, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
print(coarseSol)
print(CoarseFunc(coarseSol))

# Next for each of the [u(x_j),u(x_j+1)], we need to apply finite difference method again to solve the solution locally
#  and aim at obtaining training data for the NN, such that for each input: (u(i),u(i+1)) on the coarse grid, we
# hope to give 6 different output for the gradient descent algorithm

# Consider the problem on the subinterval [x_j,x_j+1] on [-1,1] such that :
# phi_j*phi_j_x-phi_j_xx=0, phi_j(x_j)=u(xj), u(x_j+1)=u(x_j+1). We hope to obtain such phi_j and approximate the
# derivative of phi_j at x_j and x_j+1 respectively, and this is 2 of the target output for the NN.

# consider the local problem on [xj, xj+1]: y*y_x-y_xx=0; y(xj)=u(xj); y(xj+1)=u(j+1).
# Similar to the above, use finite difference scheme with N2=10
# after this step, we expect to have y0,y2, ..., yN-1, where yi is the solution on [xi,xi+1]
N2 = 10
h2 = 2 / N2

#define FineFunc that value of r2 changes each loop in order to apply the boundary condition

def FineFunc(u):
    return u * (Dx1 @ u) - Dx2 @ u - r2


for i in range(1):
    r2 = np.hstack([coarseSol[i], np.zeros(N-2), coarseSol[i + 1]])
    fineSol = fsolve(FineFunc, np.array([10, 2, 3, 4, 5, 6, 7, 8, 9, 10]))

print(fineSol)
print(FineFunc(fineSol))


###################################Steepest gradient descent algorithm ###############################################################

#Consider the objective function F(u1,u2,....,uN), where F is defined as:
# sum_i=1^N(partial y/partial x (ui-1)-partial y/partial x (ui))^2
# we want to find the minimizer of F, namely u*.
#what we need to do is to compute the value of 5 derivative by NN at each iteration, where the Gradient
#descent scheme is given by u(k+1)=u(k)-grad(F), i.e. for each i, ui(k+1)=ui(k)-partial(F)/partial(ui)



