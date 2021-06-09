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


N = 10
# boundary condition for the target problem, u(-1)=a, u(1)=b
a = -0.3
b = 0.3
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
model = load_model('model/model_10intervals.h5')
N = 10
sol = np.array([-3.00000000e-01, -2.23016556e-01, -1.58719619e-01, -1.02147175e-01,
                -5.00444455e-02, 2.26571717e-10, 5.00444454e-02, 1.02147175e-01,
                1.58719619e-01, 2.23016556e-01, 3.00000000e-01])
# initial guess
u_array = np.empty([N + 1, 1])
u_iter = np.linspace(a, b, N + 1)[1:-1]
u = np.hstack([a, u_iter, b])
# u = sol
print(coarseFunc(sol))

# instead of defining function F, we set the stopping criteria as |e_k|=|uk+1-u_k| since then we don't need to compute
# all partial derivative for every loop

count = 0
alpha = 0.0001
tol = 0.0000001

while np.linalg.norm(coarseFunc(u)) > tol:
    # defining array for storing partial derivative for each loop (since u are different for each loop)
    store = np.zeros([N, 6])
    count = count + 1

    for i in range(N):
        store[i, :] = model.predict(np.array([u[i:i + 2]]))
    store = np.vstack((store, np.zeros(6)))
    # store = np.vstack((store, model.predict(np.array([[u[N - 1], u[0]]]))))
    # using the prediction to do the iteration
    print(count, "th store", store)
    grad = 2 * (store[0:N - 1, 5] - store[1:N, 2]) * (store[0:N - 1, 1] - store[1:N, 0]) + 2 * (
            store[1:N, 1] - store[2:N + 1, 0]) * (store[1:N, 3])
    u_iter = u_iter - alpha * grad
    u = np.hstack([a, u_iter, b])
    u_array = np.append(u_array, u)
    # print(u_iter)
    print("Error for each entry when plugging in into original discretized system :",
          np.linalg.norm(coarseFunc(u)) / N)
    print("Error with fine grid:", np.linalg.norm(u - sol) / np.linalg.norm(sol))

# print(count)
print("end")
print(coarseFunc(u))
