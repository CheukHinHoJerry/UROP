"""
# Consider the objective function F(u1,....,uN-1), notice that u0 and uN are given by the boundary
# condition, and therefore no need to be considered. Here F is defined as:
# sum_i=1^N(partial y/partial x (ui-1)-partial y/partial x (ui))^2
# we want to find the minimizer of F, namely u*.
# what we need to do is to compute the value of 5 derivative by NN at each iteration, where the Gradient
# descent scheme is given by u(k+1)=u(k)-grad(F), i.e. for each i, ui(k+1)=ui(k)-partial(F)/partial(ui)
"""
import numpy as np
import sympy as sp
from scipy.optimize import fsolve
import tensorflow as tf
import scipy.io
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from numpy import linalg
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from keras.models import load_model


def udiff(saveArray):
    return np.norm(saveArray[-1] - saveArray[-2])


def coarseFunc(u):
    return u * (Dx1 @ u) - Dx2 @ u - r0


N = 10
# boundary condition for the target problem, u(-1)=a, u(1)=b
a = 0.3
b = 0.5
r0 = np.hstack([a, np.zeros(N - 2), b])

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
u_array = np.empty([N - 2, 1])
u = np.ones(N - 2)
print(model.predict(data_x))

# instead of defining function F, we set the stopping criteria as |e_k|=|uk+1-u_k| since then we don't need to compute
# all partial derivative for every loop


while udiff(u_array) < 0.001:
    # defining array for storing partial derivative for each loop (since u are different for each loop)
    store = np.empty([N - 1, 6])
    for i in range(N - 1):
        prediction = model.predict(u_array[i + 1] - u_array[i])
        store[:, i] = prediction

    # using the prediction to do the iteration
    u = u - 2 * (store[0:N - 3, 5] - store[1:N - 2, 1]) * (store[0:N - 3, 1] - store[1:N - 2, 0]) - 2 * (
            store[1:N - 2, 4] - store[2:N - 1, 0]) * (store[1:N - 2, 4])
    u_array = np.append(u_array, u)

print(coarseFunc(u))
