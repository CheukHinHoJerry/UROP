import numpy as np
import sympy as sp
from scipy.optimize import fsolve

a = 1
b = 1

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
Dx2[0, 1] = -1 * a
Dx2[N - 1, N - 1] = -1 * b


def func(u):
    return u * (Dx1 @ u) - Dx2 @ u

print(func(10*np.ones(10)))
sol = fsolve(func, np.array([10,2,3,4,5,6,7,8,9,10]))
print(sol)


# Next for each of the [u(x_j),u(x_j+1)], we need to apply finite difference method again to solve the solution locally
#  and aim at obtaining training data for the NN, such that for each input: (u(i),u(i+1)) on the coarse grid, we
# hope to give 6 different output for the gradient descent algorithm

# Consider the problem on the subinterval [x_j,x_j+1] on [-1,1] such that :
# phi_j*phi_j_x-phi_j_xx=0, phi_j(x_j)=u(xj), u(x_j+1)=u(x_j+1). We hope to obtain such phi_j and approximate the
# derivative of phi_j at x_j and x_j+1 respectively, and this is 2 of the target output for the NN.



