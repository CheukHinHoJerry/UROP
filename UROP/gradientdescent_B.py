"""
# Consider the objective function F(u1,....,uN-1), notice that u0 and uN are given by the boundary
# condition, and therefore no need to be considered. Here F is defined as:
# sum_i=1^N(partial y/partial x (ui-1)-partial y/partial x (ui))^2
# we want to find the minimizer of F, namely u*.
# what we need to do is to compute the value of 5 derivative by NN at each iteration, where the Gradient
# descent scheme is given by u(k+1)=u(k)-grad(F), i.e. for each i, ui(k+1)=ui(k)-partial(F)/partial(ui)
@@@@@
"""
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import time


def calError(prediction, target):
    error = 0
    for i in range(len(target)):
        error = error + np.linalg.norm(prediction[i] - target[i]) / np.linalg.norm(target[i])
    error = error / len(target)
    print(error)


def udiff(saveArray):
    return np.linalg.norm(saveArray[-1] - saveArray[-2])


# def coarseFunc(u):
#     return u * (Dx1 @ u) - Dx2 @ u - r0


# defining target function, take DU as an input of size (which contains different partial derivative from the NN

# number of intervals
N = 10
h = (2 / N) / (N ** 3)

# boundary condition for the target problem, u(-1)=a, u(1)=b
a = 0.6
b = -0.8

# a = -0.3
# b = 1
#
# a = 0.8
# b = 0.2

# a = 0.1
# b = -0.3
# import model for looping
model = load_model('model/10outputs_model_1000*10intervals.h5')
# model = load_model('model/10outputs_model_100*10intervals_remove4.h5')
# data_x = np.loadtxt('10_outputs_data_x_100*10intervals_moreData.txt', delimiter=',')
# target = np.loadtxt('10_outputs_target_100*10intervals_moreData.txt', delimiter=',')
# prediction = model.predict(data_x)


sol = np.array([0.6, 0.4692596, 0.3257162, 0.17255989, 0.01405396, -0.14489576,
                -0.2993287, -0.44483116, -0.57800285, -0.69669375, -0.8])

# sol = np.array([0.1, 0.05812646, 0.01576398, -0.02673208, -0.06900187, -0.11069313,
#                 -0.15147279, -0.19103717, -0.22912029, -0.26549974, -0.3])

# initial guess
u_array = np.zeros([N + 1, 1])
u_iter = np.linspace(a, b, N + 1)[1:-1]
print(u_iter)
# u_iter = np.copy(sol[1:-1])
u = np.hstack([a, u_iter, b])
print(u[N])
# instead of defining function F, we set the stopping criteria as |e_k|=|uk+1-u_k| since then we don't need to compute
# all partial derivative for every loop

count = 0
alpha1 = 0.001
alpha2 = 1e-17
tol = 0.0000001
save_deri_f_error = 999
save_fine_error = 999
# while np.linalg.norm(u - sol) / np.linalg.norm(sol) > tol:
while count < 1000:
    # defining array for storing partial derivative for each loop (since u are different for each loop)
    store = np.zeros([N, 10])
    count = count + 1

    for i in range(N):
        store[i, :] = model.predict(np.array([u[i:i + 2]]))

    # store = np.vstack((store, model.predict(np.array([[u[N - 1], u[0]]]))))
    # using the prediction to do the iteration
    # print(count, "th store", store)

    A = np.linalg.norm(store[0:N - 1, 1] - store[1:N, 0]) ** 2
    B = np.linalg.norm((store[2:N - 1, 6] - 2 * u_iter[1:N - 2] + store[1:N - 2, 7]) / (h ** 2) + u_iter[1:N - 2] * ((
            store[2:N - 1, 6] - store[1:N - 2, 7]) / (2 * h)))
    print("Error of A", A)
    print("Error of B", B)
    if A < save_deri_f_error:
        save_fine_error = A
    print(count)
    # update u[2,N-2],which is the sol except the first two and last two entry
    # print(store)

    grad1 = 2 * (store[0:N - 3, 1] - store[1:N - 2, 0]) * (- store[1:N - 2, 4]) + 2 * (
                store[1:N - 2, 1] - store[2:N - 1, 0]) * (
                    store[1:N - 2, 5] - store[2:N - 1, 2]) + 2 * (
                    store[2:N - 1, 1] - store[3:N, 0]) * (store[2:N - 1, 3])
    print(grad1.shape)
    # 3 parts of derivatives, grad2 = grad21 + grad22 + grad23
    print((store[1:N - 2, 6] - 2 * u_iter[1:N - 2] + store[0:N - 3, 7]) / (h ** 2))
    grad21 = 2 * ((store[1:N - 2, 6] - 2 * u_iter[0:N - 3] + store[0:N - 3, 7]) / (h ** 2) + u_iter[0:N - 3] * (
            store[1:N - 2, 6] - store[0:N - 3, 7]) / (2 * h)) * (
                     store[1:N - 2, 8] / (h ** 2) + u_iter[0:N - 3] * store[1:N - 2, 8] / (2 * h))

    grad22 = 2 * ((store[2:N - 1, 6] - 2 * u_iter[1:N - 2] + store[1:N - 2, 7]) / (h ** 2) + u_iter[1:N - 2] * (
            store[2:N - 1, 6] - store[1:N - 2, 7]) / (2 * h)) * (
                     (store[2:N - 1, 8] - 2 + store[1:N - 2, 9]) / (h ** 2) + (store[2:N - 1, 6] - store[1:N - 2, 7]) / (2 * h) +
                     u_iter[1:N - 2] / (2 * h) * (store[2:N - 1, 8] - store[1: N - 2, 9]))

    grad23 = 2 * ((store[3:N, 6] - 2 * u_iter[2:N-1] + store[2:N - 1, 7]) / (h ** 2) + u_iter[2:N-1] * (store[3:N, 6] - store[2:N - 1, 7]) / (2 * h)) * (
                     store[2:N - 1, 9] / (h ** 2) - u_iter[2:N-1] * store[2:N - 1, 9] / (2 * h))

    # print(grad1)
    # print("grad2 first:",2 * (u_iter[1: -1]*(
    #     store[1:N - 2, 6] - store[0:N - 3, 7])) * (1 / (4 * h ** 2) + u_iter[1: -1] * 1 / (2 * h)))
    # print("grad2 second:",(
    #                 (store[2:N - 1, 6] - 2 * u_iter[1: -1] + store[1:N - 2, 7]) / (4 * h ** 2) + u_iter[1: -1] * (
    #                 store[2:N - 1, 6] - store[1:N - 2, 7]) / (2 * h)) * ((store[2:N - 1, 8] - 2 + store[1:N - 2, 9]) / (4 * h ** 2) + (
    #         store[2:N - 1, 6] - store[1:N - 2, 7]) / (
    #                      2 * h) +
    #              u_iter[1: -1] / (2 * h) * (store[2:N - 1, 8] - store[1: N - 2, 9])))
    '''
        u_iter[1:-1] = u_iter[1:-1] - alpha1 * grad1 \
            - alpha2 * (grad21 + grad22 + grad23)
    '''
    u_iter[1:-1] = u_iter[1:-1] - alpha2 * (grad21 + grad22 + grad23)
    # update the second last and second first entry, i.e. u[1] and u[N-1], where u is the solution
    '''
    u_iter[0] = u_iter[0] - alpha1 * (2 * (store[0, 5] - store[1, 2]) * (store[0, 1] - store[1, 0]) + 2 * (
            store[1, 1] - store[2, 0]) * (store[1, 3])) \
    '''
    u_iter[0] = u_iter[0] - alpha2 * (
                2 * ((store[1, 6] - 2 * u_iter[0] + store[0, 7]) / (h ** 2) + u_iter[0] * (
                store[1, 6] - store[0, 7]) / (2 * h)) * (
                        (store[1, 8] - 2 + store[0, 9]) / (h ** 2) + (
                        store[1, 6] - store[0, 7]) / (2 * h) +
                        u_iter[0] / (2 * h) * (store[1, 8] - store[0, 9])) + 2 * (
                            (store[2, 6] - 2 * u_iter[1] + store[1, 7]) / (h ** 2) + u_iter[1] * (
                            store[2, 6] - store[1, 7]) / (2 * h)) * (
                        store[1, 9] / (h ** 2) - u_iter[1] * store[1, 9] / (2 * h)))
    '''
        u_iter[-1] = u_iter[-1] - alpha1 * (2 * (store[N - 3, 1] - store[N - 2, 0]) * (- store[N - 2, 4]) + 2 * (
                store[N - 2, 5] - store[N - 1, 2]) * (store[N - 2, 1] - store[N - 1, 0])) \
     \
            - alpha2 * (
                    2 * ((store[N - 2, 6] - 2 * u_iter[-2] + store[N - 3, 7]) / (h ** 2) + u_iter[-2] * (
                    store[N - 2, 6] - store[N - 3, 7])/(2 * h)) * (
                            store[N - 2, 8] / (h ** 2) + u_iter[-2] * store[N - 2, 8] / (2 * h))
                    + 2 * (
                            (store[N - 1, 6] - 2 * u_iter[-1] + store[N - 2, 7]) / (h ** 2) + u_iter[
                        -1] * (
                                    store[N - 1, 6] - store[N - 2, 7]) / (2 * h)) * (
                            (store[N - 1, 8] - 2 + store[N - 2, 9]) / (h ** 2) + (
                            store[N - 1, 6] - store[N - 2, 7]) / (
                                    2 * h) +
                            u_iter[-1] / (2 * h) * (store[N - 1, 8] - store[N - 2, 9])))
    '''

    u_iter[-1] = u_iter[-1] - alpha2 * (
            2 * ((store[N - 2, 6] - 2 * u_iter[-2] + store[N - 3, 7]) / (h ** 2) + u_iter[-2] * (
            store[N - 2, 6] - store[N - 3, 7]) / (2 * h)) * (
                    store[N - 2, 8] / (h ** 2) + u_iter[-2] * store[N - 2, 8] / (2 * h))
            + 2 * ((store[N - 1, 6] - 2 * u_iter[-1] + store[N - 2, 7]) / (h ** 2) + u_iter[-1]
                   * (store[N - 1, 6] - store[N - 2, 7]) / (2 * h)) * (
                    (store[N - 1, 8] - 2 + store[N - 2, 9]) / (h ** 2) + (
                    store[N - 1, 6] - store[N - 2, 7]) / (2 * h) +
                    u_iter[-1] / (2 * h) * (store[N - 1, 8] - store[N - 2, 9])))

    u = np.hstack([a, u_iter, b])
    u_array = np.append(u_array, u)
    # print(u_iter)
    print("u: ", u)
    print("Error array: ", u - sol)
    print("Error with fine grid:", np.linalg.norm(u - sol) / np.linalg.norm(sol))
    if np.linalg.norm(u - sol) / np.linalg.norm(sol) < save_fine_error:
        save_fine_error = np.linalg.norm(u - sol) / np.linalg.norm(sol)

# print(count)
print("end")
