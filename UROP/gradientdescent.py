"""
# Consider the objective function F(u1,....,uN-1), notice that u0 and uN are given by the boundary
# condition, and therefore no need to be considered. Here F is defined as:
# sum_i=1^N(partial y/partial x (ui-1)-partial y/partial x (ui))^2
# we want to find the minimizer of F, namely u*.
# what we need to do is to compute the value of 5 derivative by NN at each iteration, where the Gradient
# descent scheme is given by u(k+1)=u(k)-grad(F), i.e. for each i, ui(k+1)=ui(k)-partial(F)/partial(ui)
"""
import numpy as np
from keras.models import load_model


def udiff(saveArray):
    return np.linalg.norm(saveArray[-1] - saveArray[-2])


def coarseFunc(u):
    return u * (Dx1 @ u) - Dx2 @ u - r0


N = 10
# boundary condition for the target problem, u(-1)=a, u(1)=b
a = 0.1
b = 0.1
r0 = np.array([np.hstack([a, np.zeros(N - 2), b])]).T
# step size on coarse grid
h = 2 / N
# assigning derivative matrix
Dx1 = np.eye(N, k=1) - np.eye(N, k=-1)
Dx1 = Dx1 / (2 * h)
# First and last row of Dx1 is zero for applying boundary condition
Dx1[0, :] = np.zeros(N)
Dx1[N - 1, :] = np.zeros(N)

Dx2 = np.eye(N, k=1) + np.eye(N, k=-1) - 2 * np.eye(N, k=0)
Dx2 = Dx2 / (2 * h * 2 * h)

# For applying boundary condition
Dx2[0, :] = np.zeros(N)
Dx2[N - 1, :] = np.zeros(N)
Dx2[0, 0] = -1
Dx2[N - 1, N - 1] = -1

# import model for looping
data_x = np.loadtxt('data_x.txt', delimiter=',')
model = load_model('model1.h5')
N = 10

# initial guess
u_array = np.empty([N, 1])
u_iter = np.array([np.ones(N-2)]).T
u = np.vstack([a, u_iter, b])

# instead of defining function F, we set the stopping criteria as |e_k|=|uk+1-u_k| since then we don't need to compute
# all partial derivative for every loop


count = 0
for i in range(10):
    # defining array for storing partial derivative for each loop (since u are different for each loop)
    store = np.empty([N - 1, 6])
    count = count+1
    for k in range(N - 1):
        print(u[k:k+2])
        store[k, :] = model.predict(u[k:k+2].T).T[:,0]
    store = np.vstack((store, model.predict(np.array([u[N - 1], u[0]]).T).T[:, 0]))
    # using the prediction to do the iteration
    print(store)
    u_iter = u_iter - 2 * (np.array([store[0:N - 2, 5]]).T - np.array([store[1:N-1, 2]]).T) * \
             (np.array([store[0:N - 2, 1]]).T - np.array([store[1:N - 1, 0]]).T) - 2 * (
            np.array([store[1:N - 1, 4]]).T - np.array([store[2:N, 0]]).T) * np.array([store[1:N - 1, 3]]).T
    u = np.vstack([a, u_iter, b])
    print(u)
    u_array = np.hstack([u_array, u])
    print(u_array)


print(u)
print(coarseFunc(u), "123")
