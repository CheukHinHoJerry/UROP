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


u_range = np.linspace(-1, 1, 41)

N = 10

# step size on fine grid
h = (2 / N) / N

# assigning derivative matrix
Dx1 = np.eye(N, k=1) - np.eye(N, k=-1)
Dx1 = Dx1 / (2 * h)
# First and last row of Dx1 is zero for applying boundary condition
Dx1[0, :] = np.zeros(N)
Dx1[N - 1, :] = np.zeros(N)

Dx2 = np.eye(N, k=1) + np.eye(N, k=-1) - 2 * np.eye(N, k=0)
Dx2 = Dx2 / (2*h*2*h)

# For applying boundary condition
Dx2[0, :] = np.zeros(N)
Dx2[N - 1, :] = np.zeros(N)
Dx2[0, 0] = -1
Dx2[N - 1, N - 1] = -1

# define empty matrix and data_x for network training
target = np.empty([6, len(u_range) * len(u_range)], dtype=float)
data_x = np.empty([2, len(u_range) * len(u_range)], dtype=float)


# define FineFunc that value of r0 changes each loop in order to apply the boundary condition
def fineFunc(u):
    return u * (Dx1 @ u) - Dx2 @ u - r0


def partialuFunc1(y):
    return -Dx2 @ y + (Dx1 @ y) * fineSol + Dx1 @ fineSol * y - r1


def partialuFunc2(y):
    return -Dx2 @ y + (Dx1 @ y) * fineSol + Dx1 @ fineSol * y - r2


count = 0
for i in range(len(u_range)):
    for j in range(len(u_range)):
        a = u_range[i]
        b = u_range[j]
        data_x[0, i * len(u_range) + j] = a
        data_x[1, i * len(u_range) + j] = b
        count = count + 1
        # initializing a vector for implementing boundary condition (where the boundary condition is different
        # for each loop
        r0 = np.hstack([a, np.zeros(N - 2), b])

        # solving the system with some initial guess
        fineSol = fsolve(fineFunc, np.array([10, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
        print("The solution for the ", i * len(u_range) + j, " th data is: ", fineSol)
        print("The error vector for the ", i * len(u_range) + j, " th data is: ", fineFunc(fineSol))

        # initialize vector for implementing boundary condition
        r1 = np.hstack([1, np.zeros(N - 2), 0])
        r2 = np.hstack([0, np.zeros(N - 2), 1])

        partialuSol1 = fsolve(partialuFunc1, np.array([10, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
        partialuSol2 = fsolve(partialuFunc2, np.array([10, 2, 3, 4, 5, 6, 7, 8, 9, 10]))

        # first two entries of a particular row stores the derivative of end points, the last four entries are
        # dy_{i,i}/dx (xi) , dy_{i,i}/dx (xi+1), dy_{i,i+1}/dx (xi) , dy_{i,i+1}/dx (x[[[[[[[[[[i+1),
        target[0, i * len(u_range) + j] = (fineSol[1] - fineSol[0]) / h
        target[1, i * len(u_range) + j] = (fineSol[N - 1] - fineSol[N - 2]) / h
        target[2, i * len(u_range) + j] = (partialuSol1[1] - partialuSol1[0]) / h
        target[3, i * len(u_range) + j] = (partialuSol1[N - 1] - partialuSol1[N - 2]) / h
        target[4, i * len(u_range) + j] = (partialuSol2[1] - partialuSol2[0]) / h
        target[5, i * len(u_range) + j] = (partialuSol2[N - 1] - partialuSol2[N - 2]) / h

print(target)
print(count)


""" Training of NN  """

# def step_decay(epoch):
#     initial_learning_rate = 0.01
#     lrate = initial_learning_rate * 0.96 ** (int((epoch - 1) / 200))
#     return lrate

"""
def normalize(X, Y):
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    Y = (Y - np.mean(Y, axis=0)) / np.std(Y, axis=0)
    return X, Y, np.std(X, axis=0), np.std(Y, axis=0), np.mean(X, axis=0), np.mean(Y, axis=0)


def denomalize(prediction, meany, stdy):
    prediction = prediction * stdy + meany
    return prediction

print(data_x)
print(np.std(data_x, axis=0))
normalized_data_x, normalized_target, std_x, std_y, mean_x, mean_y = normalize(data_x, target)
train_x, valid_x, train_y, valid_y = train_test_split(normalized_data_x, normalized_target, test_size=0.3, random_state= 42 )

model = tf.keras.models.Sequential()
#lrate = tf.keras.callbacks.LearningRateScheduler(step_decay)
checkpoint_filepath = '/tmp/checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)
earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=6)
model.add(tf.keras.Input(shape=2))
model.add(tf.keras.layers.Dense(5, activation='relu', activity_regularizer=regularizers.l2(1e-4)))
model.add(tf.keras.layers.Dense(5, activation='relu', activity_regularizer=regularizers.l2(1e-4)))
model.add(tf.keras.layers.Dense(5, activation='relu', activity_regularizer=regularizers.l2(1e-4)))
model.add(tf.keras.layers.Dense(6, activation='linear', activity_regularizer=regularizers.l2(1e-4)))

model.compile(optimizer='adam', loss='mse', metrics=['MeanSquaredError'])

model.fit(train_x, train_y, validation_data=(valid_x, valid_y), epochs=20000, batch_size=1,
          callbacks=[earlystop_callback, model_checkpoint_callback])

###################################Steepest gradient descent algorithm ################################################

# Consider the objective function F(u0,u1,....,uN-1), where F is defined as:
# sum_i=1^N(partial y/partial x (ui-1)-partial y/partial x (ui))^2
# we want to find the minimizer of F, namely u*.
# what we need to do is to compute the value of 5 derivative by NN at each iteration, where the Gradient
# descent scheme is given by u(k+1)=u(k)-grad(F), i.e. for each i, ui(k+1)=ui(k)-partial(F)/partial(ui)
"""