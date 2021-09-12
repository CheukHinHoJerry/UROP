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
from sklearn.model_selection import train_test_split
import time
import tensorflow as tf
from solveFullSystem import solveFullSystem
import json


def calError(prediction, target):
    error = 0
    for i in range(len(target)):
        error = error + np.linalg.norm(prediction[i] - target[i]) / np.linalg.norm(target[i])
    error = error / len(target)
    print(error)


def udiff(saveArray):
    return np.linalg.norm(saveArray[-1] - saveArray[-2])


model = load_model('model/10outputs_model_1000*10intervals.h5')

# def coarseFunc(u):
#     return u * (Dx1 @ u) - Dx2 @ u - r0


# number of intervals
N = 10
h = (2 / N) / (N ** 3)
f = open('result.txt', 'w')
save_ab = []
save_min_error_A = []
save_min_error_B = []
save_min_error = []
save_solve_fullSystem_time = []
save_solve_nn_time = []
# testing interpolation, using (tanh(x1),tan(x2)) with x1,x2 runs through [-5,5]
for i in range(10):
    for j in range(10):
        # boundary condition for the target problem, u(-1)=a, u(1)=b
        a = 1.5 * np.tanh(0.2 * (-5 + i) + 0.1)
        b = 1.5 * np.tanh(0.2 * (-5 + j) + 0.2)
        r0 = np.hstack([a, np.zeros(N ** 4 - 1), b])
        count = 0
        alpha1 = 0.001  # 0.001
        alpha2 = 0.000000000000000005
        # calculaing full system solution and record the time needed
        print("solving Full system")
        start_solving_full_system = time.time()
        # sol = solveFullSystem(a, b)
        sol = np.linspace(a, b, N ** 4 + 1)
        end_solving_full_system = time.time()
        print("end of solving Full system")

        tmp = []
        for coarse_point in range(N + 1):
            tmp.append(sol[coarse_point * N ** 3])
        coarse_sol = np.array(tmp)
        # initial guess, using linear function as guessing
        u_iter = np.linspace(a, b, N + 1)[1:-1]
        u = np.hstack([a, u_iter, b])
        tol = 0.0000001
        min_error = np.inf
        min_error_A = np.inf
        min_error_B = np.inf
        patience = 0
        start_solving_with_NN = time.time()

        # stopping criteria 1 :
        # while(current_error_with_fine_grid > tol):
        # stopping criteria 2: patience
        while (patience < 6):
            # stopping criteria 3: count
            # while count < 10000:

            # defining array for storing partial derivative for each loop (since u are different for each loop)
            store = np.zeros([N, 10])
            count = count + 1

            # using nn
            for i in range(N):
                store[i, :] = model.predict(np.array([u[i:i + 2]]))

            A = np.linalg.norm(store[0:N - 1, 1] - store[1:N, 0]) ** 2
            B = np.linalg.norm((store[2:N - 1, 6] - 2 * u_iter[1:N - 2] + store[1:N - 2, 7]) / (h ** 2)
                               + u_iter[1:N - 2] * ((store[2:N - 1, 6] - store[1:N - 2, 7]) / (2 * h)))
            # print("A", A)
            # print("B", B)
            if A < min_error_A:
                min_error_A = A
            if B < min_error_B:
                min_error_B = B
            # update u[2,N-2],which is the sol except the first two and last two entry
            # print(store)

            grad1 = 2 * (store[0:N - 3, 1] - store[1:N - 2, 0]) * (- store[1:N - 2, 4]) + 2 * (
                    store[1:N - 2, 1] - store[2:N - 1, 0]) * (
                            store[1:N - 2, 5] - store[2:N - 1, 2]) + 2 * (
                            store[2:N - 1, 1] - store[3:N, 0]) * (store[2:N - 1, 3])
            # 3 parts of derivatives, grad2 = grad21 + grad22 + grad23
            grad21 = 2 * ((store[1:N - 2, 6] - 2 * u_iter[0:N - 3] + store[0:N - 3, 7]) / (h ** 2) + u_iter[0:N - 3] * (
                    store[1:N - 2, 6] - store[0:N - 3, 7]) / (2 * h)) * (
                             store[1:N - 2, 8] / (h ** 2) + u_iter[0:N - 3] * store[1:N - 2, 8] / (2 * h))

            grad22 = 2 * ((store[2:N - 1, 6] - 2 * u_iter[1:N - 2] + store[1:N - 2, 7]) / (h ** 2) + u_iter[1:N - 2] * (
                    store[2:N - 1, 6] - store[1:N - 2, 7]) / (2 * h)) * (
                             (store[2:N - 1, 8] - 2 + store[1:N - 2, 9]) / (h ** 2) + (
                             store[2:N - 1, 6] - store[1:N - 2, 7]) / (2 * h) +
                             u_iter[1:N - 2] / (2 * h) * (store[2:N - 1, 8] - store[1: N - 2, 9]))

            grad23 = 2 * ((store[3:N, 6] - 2 * u_iter[2:N - 1] + store[2:N - 1, 7]) / (h ** 2) + u_iter[2:N - 1] * (
                    store[3:N, 6] - store[2:N - 1, 7]) / (2 * h)) * (
                             store[2:N - 1, 9] / (h ** 2) - u_iter[2:N - 1] * store[2:N - 1, 9] / (2 * h))

            # update interior of u_iter
            u_iter[1:-1] = u_iter[1:-1] - alpha1 * grad1 \
                           - alpha2 * (grad21 + grad22 + grad23)

            # update the second last and second first entry, i.e. u[1] and u[N-1], where u is the solution
            u_iter[0] = u_iter[0] - alpha1 * (2 * (store[0, 5] - store[1, 2]) * (store[0, 1] - store[1, 0])
                                              + 2 * (store[1, 1] - store[2, 0]) * (store[1, 3])) \
                        - alpha2 * (2 * ((store[1, 6] - 2 * u_iter[0] + store[0, 7]) / (h ** 2)
                                         + u_iter[0] * (store[1, 6] - store[0, 7]) / (2 * h)) *
                                    ((store[1, 8] - 2 + store[0, 9]) / (h ** 2) + (store[1, 6] - store[0, 7]) / (
                                            2 * h) +
                                     u_iter[0] / (2 * h) * (store[1, 8] - store[0, 9]))
                                    + 2 * ((store[2, 6] - 2 * u_iter[1] + store[1, 7]) / (h ** 2)
                                           + u_iter[1] * (store[2, 6] - store[1, 7]) / (2 * h)) * (
                                            store[1, 9] / (h ** 2)
                                            - u_iter[1] * store[1, 9] / (
                                                    2 * h)))

            u_iter[-1] = u_iter[-1] - alpha1 * (2 * (store[N - 3, 1] - store[N - 2, 0]) * (- store[N - 2, 4]) + 2 * (
                    store[N - 2, 5] - store[N - 1, 2]) * (store[N - 2, 1] - store[N - 1, 0])) \
                         - alpha2 * (2 * (
                    (store[N - 2, 6] - 2 * u_iter[-2] + store[N - 3, 7]) / (h ** 2) + u_iter[-2] * (
                    store[N - 2, 6] - store[N - 3, 7]) / (2 * h))
                                     * (store[N - 2, 8] / (h ** 2) + u_iter[-2] * store[N - 2, 8] / (2 * h))
                                     + 2 * ((store[N - 1, 6] - 2 * u_iter[-1] + store[N - 2, 7]) / (h ** 2)
                                            + u_iter[-1] * (store[N - 1, 6] - store[N - 2, 7]) / (2 * h))
                                     * ((store[N - 1, 8] - 2 + store[N - 2, 9]) / (h ** 2)
                                        + (store[N - 1, 6] - store[N - 2, 7]) / (2 * h) +
                                        u_iter[-1] / (2 * h) * (store[N - 1, 8] - store[N - 2, 9])))

            # update u
            u = np.hstack([a, u_iter, b])
            current_error_with_fine_grid = np.linalg.norm(u - coarse_sol) / np.linalg.norm(coarse_sol)
            # print("current error with fine grid:", current_error_with_fine_grid)
            # print("patience", patience)
            if current_error_with_fine_grid < min_error:
                min_error = current_error_with_fine_grid
                patience = 0
            else:
                patience += 1
        end_solving_with_NN = time.time()
        line = "(a,b)=(" + str(a) + "," + str(b) + ")" + "  min_A:" + str(min_error_A) + ", min_B:" + str(
            min_error_B) + ", min_error:" + \
               str(min_error) + ", Time for solving full system:" + str(
            end_solving_full_system - start_solving_full_system) + \
               ", Time for iteration using DNN：" + str(end_solving_with_NN - start_solving_with_NN)
        save_ab.append([a, b])
        save_min_error_A.append(min_error_A)
        save_min_error_B.append(min_error_B)
        save_min_error.append(min_error)
        save_solve_fullSystem_time.append(end_solving_full_system - start_solving_full_system)
        save_solve_nn_time.append(end_solving_with_NN - start_solving_with_NN)
        f.write(line)
        f.write('\n')
        print("finished one iteration, proceed to next pair of (a,b)")
save_dict = {"(a,b)": save_ab, "save_min_error_A": save_min_error_A, "save_min_error_B": save_min_error_B,
             "min_error": save_min_error, "save_solve_fullSystem_time": save_solve_fullSystem_time,
             "save_solve_nn_time": save_solve_nn_time}

np.save(save_dict)
a_file = open("result.json", "w")
json.dump(save_dict, a_file)
a_file.close()

# a_file = open("result.json", "r")
# output = a_file.read()
# print(output)
# a_file.close()
