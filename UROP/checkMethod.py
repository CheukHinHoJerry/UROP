"""
In this file, we check the validity of the proposed method. Suppose the solution if the minimizer of F, then
suppose u is the solution obtained from directly solving the equation on fine grid using finite difference method,
which has size N*N, then we should have that F(u*) being small, where u* is the vector obtained from extracting
0, N-1, ..., N*(N-1) th element of u.

"""
import numpy as np
from keras.models import load_model
from scipy.optimize import fsolve

data_x = np.loadtxt('data_x2.txt', delimiter=',')
target = np.loadtxt('target2.txt', delimiter=',')
model = load_model('model2.h5')

# for a=b=0, solve the equation on coarse grid

N = 10
# boundary condition for the target problem, u(-1)=a, u(1)=b
a = 0.3
b = -0.3
r0 = np.hstack([a, np.zeros(N * N - 2), b])
# step size on fine grid
h = (2 / N) / N
# assigning derivative matrix
Dx1 = np.eye(N * N, k=1) - np.eye(N * N, k=-1)
Dx1 = Dx1 / (2 * h)
# First and last row of Dx1 is zero for applying boundary condition
Dx1[0, :] = np.zeros(N * N)
Dx1[N * N - 1, :] = np.zeros(N * N)

Dx2 = np.eye(N * N, k=1) + np.eye(N * N, k=-1) - 2 * np.eye(N * N, k=0)
Dx2 = Dx2 / (2 * h * 2 * h)

# For applying boundary condition
Dx2[0, :] = np.zeros(N * N)
Dx2[N * N - 1, :] = np.zeros(N * N)
Dx2[0, 0] = -1
Dx2[N * N - 1, N * N - 1] = -1


def coarseFunc(u):
    return u * (Dx1 @ u) - Dx2 @ u - r0


# solving the system with some initial guess
sol = fsolve(coarseFunc, np.ones(N * N))
print("The solution for is", sol)
print("The error vector is: ", coarseFunc(sol))

# checking the input from NN
diff_sol = np.zeros([2, N - 1])
for i in range(N - 1):
    diff_sol[:, i] = np.array([(sol[(1 + i * N)] - sol[(0 + i * N)]) / h, (sol[i * (N - 1)] - sol[i * (N - 2)]) / h])
diff_sol = np.hstack((diff_sol, np.array([np.array([(sol[N * N - 1] - sol[N * N - 2]) / h,
                                         (sol[N * N - 1] - sol[0]) / h])]).T))

nn_sol = np.zeros([N - 1, 6])
for k in range(N - 1):
    nn_sol[k, :] = model.predict(np.array([sol[k:k + 2]]).T)
nn_sol = np.vstack((nn_sol, model.predict(np.array([sol[N - 1], sol[0]]).T)))
nn_diff_sol = nn_sol[N - 1, 0:2].T

print("The error from the original fine solution to that of the output from NN is:", diff_sol - nn_diff_sol)
