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
    return np.norm(saveArray[-1] - saveArray[- 2])


# import model for looping
data_x = np.loadtxt('data_x.txt', delimiter=',')
model = load_model('model1.h5')
N = 10

# initial guess
u_array = np.empty([0])
u = np.ones(N - 2)
print(model.predict(data_x))

# instead of defining function F, we set the stopping criteria as |e_k|=|uk+1-u_k| since then we don't need to compute
# all partial derivative for every loop

while udiff(u_array) < 0.001:
    #calculate different derivative by NN
    partialDeri=
    for i in range(N-1):
        model.predict(u_array[i+1]-u_array[i])

    u = u_array[-1] -
