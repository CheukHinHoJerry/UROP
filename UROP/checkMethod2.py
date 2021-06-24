"""
In this file, we check the validity of the proposed method. Suppose the solution if the minimizer of F, then
suppose u is the solution obtained from directly solving the equation on fine grid using finite difference method,
which has size N*N, then we should have that F(u*) being small, where u* is the vector obtained from extracting
0, N-1, ..., N*(N-1) th element of u.

"""
import numpy as np
from tensorflow.keras.models import load_model
from scipy.optimize import fsolve

data_x = np.loadtxt('data_x2_10interval.txt', delimiter=',')
target = np.loadtxt('target2_10interval.txt', delimiter=',')
model = load_model('model/model_10intervals.h5')

# for a and b, solve the equation on fine grid

N = 10
# boundary condition for the target problem, u(-1)=a, u(1)=b
a = 0.6
b = -0.8
r0 = np.hstack([a, np.zeros(N * N ** 2 - 1), b])
# step size on fine grid
h = (2 / N) / (N ** 2)
# assigning derivative matrix
Dx1 = np.eye(N * N ** 2 + 1, k=1) - np.eye(N * N ** 2 + 1, k=-1)
Dx1 = Dx1 / (2 * h)
# First and last row of Dx1 is zero for applying boundary condition
Dx1[0, :] = np.zeros(N * N ** 2 + 1)
Dx1[N * N ** 2, :] = np.zeros(N * N ** 2 + 1)

Dx2 = np.eye(N * N ** 2 + 1, k=1) + np.eye(N * N ** 2 + 1, k=-1) - 2 * np.eye(N * N ** 2 + 1, k=0)
Dx2 = Dx2 / (2 * h * 2 * h)

# For applying boundary condition
Dx2[0, :] = np.hstack((-1, np.zeros(N * N ** 2)))
Dx2[N * N ** 2, :] = np.hstack((np.zeros(N * N ** 2), -1))


def BurgFunc(u):
    return u * (Dx1 @ u) - Dx2 @ u - r0

def smallIntervalBurgFunc(u):
    return u * (dDx1 @ u) - dDx2 @ u - r0


# solving the system with some initial guess
sol = fsolve(BurgFunc, np.ones(N * N ** 2 + 1))
print("The solution for is", sol)
real_coarse_sol = np.array([sol[0]])

for i in range(1, N + 1):
    real_coarse_sol = np.hstack((real_coarse_sol, sol[i * N ** 2]))
print("The coarse sol is", real_coarse_sol)
print("The error vector is: ", BurgFunc(sol))

print("The left derivative at first node is: ", (real_coarse_sol[1] - sol[N ** 2 - 1]) / h)
print("The right derivative at first node is: ", (sol[N ** 2 + 1] - real_coarse_sol[1]) / h)

print(model.predict(np.array([[real_coarse_sol[0:0 + 2]]])))
print(model.predict(np.array([[real_coarse_sol[1:1 + 2]]])))

left_deri = np.zeros(N - 1)
right_deri = np.zeros(N - 1)
for i in range(1, N):
    left_deri[i - 1] = (sol[i * N ** 2] - sol[i * N ** 2 - 1]) / h
    right_deri[i - 1] = (sol[i * N ** 2 + 1] - sol[i * N ** 2]) / h

F = np.sum((left_deri - right_deri) ** 2)
print(F)

# assigning derivative matrix
dDx1 = np.eye(N ** 2 + 1, k=1) - np.eye(N ** 2 + 1, k=-1)
dDx1 = dDx1 / (2 * h)
# First and last row of Dx1 is zero for applying boundary condition
dDx1[0, :] = np.zeros(N ** 2 + 1)
dDx1[N ** 2, :] = np.zeros(N ** 2 + 1)

dDx2 = np.eye(N ** 2 + 1, k=1) + np.eye(N ** 2 + 1, k=-1) - 2 * np.eye(N ** 2 + 1, k=0)
dDx2 = dDx2 / (2 * h * 2 * h)

# For applying boundary condition
dDx2[0, :] = np.hstack((-1, np.zeros(N ** 2)))
dDx2[N ** 2, :] = np.hstack((np.zeros(N ** 2), -1))

sep_left_deri=np.zeros(N)
sep_right_deri=np.zeros(N)

for i in range(N):
    r0 = np.hstack([real_coarse_sol[i], np.zeros(N ** 2 - 1), real_coarse_sol[i+1]])
    sol = fsolve(smallIntervalBurgFunc, np.ones(N ** 2 + 1))
    sep_left_deri[i] = (sol[1]-sol[0])/h
    sep_right_deri[i] = (sol[N**2]-sol[N**2-1])/h

sep_F=np.sum((sep_left_deri[1:N]-sep_right_deri[0:N-1])**2)
print(sep_F)
# # checking the input from NN
# diff_sol = np.zeros([2, N])
# for i in range(N - 1):
#     diff_sol[:, i] = np.array(
#         [(sol[(i * N + 1)] - sol[(i * N)]) / h, (sol[((i + 1) * N)] - sol[(i + 1) * N - 1]) / h])
#     print("The", i, "th diff_sol is: ", diff_sol)
# # diff_sol[:, N-2] = np.array([(sol[N * N - 1] - sol[N * N - 2]) / h,
# #                                                    (sol[N * N - 1] - sol[0]) / h])
# print(diff_sol.shape)
# nn_sol = np.zeros([N, 6])
# for k in range(N):
#     nn_sol[k, :] = model.predict(np.array([[real_coarse_sol[k:k + 2]]]))
# # nn_sol = np.vstack((nn_sol, model.predict(np.array([[real_coarse_sol[N - 1], real_coarse_sol[0]]]))))
# print(nn_sol)
# nn_diff_sol = nn_sol[:, 0:2].T
# print("The solution from nn is", nn_diff_sol)
# print("The original solution is", diff_sol)
# print("The error from the original fine solution to that of the output from NN is:", diff_sol - nn_diff_sol)
