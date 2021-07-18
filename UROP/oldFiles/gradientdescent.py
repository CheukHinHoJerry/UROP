"""
# Consider the objective function F(u1,....,uN-1), notice that u0 and uN are given by the boundary
# condition, and therefore no need to be considered. Here F is defined as:
# sum_i=1^N(partial y/partial x (ui-1)-partial y/partial x (ui))^2
# we want to find the minimizer of F, namely u*.
# what we need to do is to compute the value of 5 derivative by NN at each iteration, where the Gradient
# descent scheme is given by u(k+1)=u(k)-grad(F), i.e. for each i, ui(k+1)=ui(k)-partial(F)/partial(ui)
"""
import numpy as np
from tensorflow.keras.models import load_model


# from keras.models import load_model


def udiff(saveArray):
    return np.linalg.norm(saveArray[-1] - saveArray[-2])


def coarseFunc(u):
    return u * (Dx1 @ u) - Dx2 @ u - r0


# defining target function, take DU as an input of size (which contains different partial derivative from the NN

# number of intervals
N = 10

# boundary condition for the target problem, u(-1)=a, u(1)=b
a = 0.6
b = -0.8

# a = -0.3
# b = 1
#
# a = 0.8
# b = 0.2

r0 = np.hstack([a, np.zeros(N - 1), b])
# step size on coarse grid
h = 2 / N
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

# import model for looping
data_x = np.loadtxt('data_x2_10interval.txt', delimiter=',')
target = np.loadtxt('target2_10interval.txt', delimiter=',')
model = load_model('../model/model_10intervals.h5')
N = 10

sol = np.array([0.6, 0.44647016, 0.22890003, -0.03090544, -0.28457717, -0.48797398,
                -0.62682007, -0.71150875, -0.7596639, -0.78595875, -0.8])
# sol = np.array([-0.3, -0.19805781, -0.11122511, -0.03186265, 0.04548198, 0.12572309,
#                 0.21455435, 0.32025255, 0.45737722, 0.65668735, 1.])
# sol = np.array([0.8, 0.79858638, 0.79590838, 0.79084667, 0.78132081, 0.76353872,
#                 0.7308424, 0.67236035, 0.57274539, 0.41640301, 0.2])

# initial guess
u_array = np.empty([N + 1, 1])
u_iter = np.linspace(a, b, N + 1)[1:-1]
u_iter = np.copy(sol[1:-1])
u = np.hstack([a, u_iter, b])
print(coarseFunc(sol))

# instead of defining function F, we set the stopping criteria as |e_k|=|uk+1-u_k| since then we don't need to compute
# all partial derivative for every loop

count = 0
alpha = 0.001
tol = 0.0000001

# while np.linalg.norm(u - sol) / np.linalg.norm(sol) > tol:
while count < 100:
    # defining array for storing partial derivative for each loop (since u are different for each loop)
    store = np.zeros([N, 6])
    count = count + 1

    for i in range(N):
        store[i, :] = model.predict(np.array([u[i:i + 2]]))
    # store = np.vstack((store, model.predict(np.array([[u[N - 1], u[0]]]))))
    # using the prediction to do the iteration
    # print(count, "th store", store)

    F = np.linalg.norm(store[0:N - 1, 1] - store[1:N, 0]) ** 2
    print(count)
    print("The error of F of last iteration is: ", F)
    # update u[2,N-2],which is the sol except the first two and last two entry
    # the first two corresponding to u[2]
    grad = 2 * (store[0:N - 3, 1] - store[1:N - 2, 0]) * (store[1:N - 2, 4]) + 2 * (
            store[1:N - 2, 5] - store[2:N - 1, 2]) * (store[1:N - 2, 1] - store[2:N - 1, 0]) + 2 * (
                   store[2:N - 1, 1] - store[3:N, 0]) * (store[2:N - 1, 3])
    u_iter[1:-1] = u_iter[1:-1] - alpha * grad
    print(store)
    # update the second last and second first entry, i.e. u[1] and u[N-1], where u is the solution
    u_iter[0] = u_iter[0] - alpha * (2 * (store[0, 5] - store[1, 2]) * (store[0, 1] - store[1, 0]) + 2 * (
            store[1, 1] - store[2, 0]) * (store[1, 3]))
    u_iter[-1] = u_iter[-1] - alpha * (2 * (store[N - 3, 1] - store[N - 2, 0]) * (store[N - 2, 4]) + 2 * (
            store[N - 2, 5] - store[N - 1, 2]) * (store[N - 2, 1] - store[N - 1, 0]))

    u = np.hstack([a, u_iter, b])
    u_array = np.append(u_array, u)
    # print(u_iter)
    print("u: ", u)
    print("Error array: ", u - sol)
    print("Error with fine grid:", np.linalg.norm(u - sol) / np.linalg.norm(sol))

# print(count)
print("end")
print(coarseFunc(u))
