"""
use function in solveFullSystem.py
solve the equation for (a, b) = (tan(x1), tan(x2)) where x1,x2 runs through [-5,5]
"""

import numpy as np
import time
from scipy.optimize import fsolve

# number of intervals, the number of intervals of fine grid is N*N^2 (number of points are N+1 and N^2+1 respectively)
N = 10
# step size on fine grid
h = 2 / (N ** 4)
# assigning derivative matrix
Dx1 = np.eye(N ** 4 + 1, k=1) - np.eye(N ** 4 + 1, k=-1)
Dx1 = Dx1 / (2 * h)
# First and last row of Dx1 is zero for applying boundary condition
Dx1[0, :] = np.zeros(N ** 4 + 1)
Dx1[N ** 4, :] = np.zeros(N ** 4 + 1)

Dx2 = np.eye(N ** 4 + 1, k=1) + np.eye(N ** 4 + 1, k=-1) - 2 * np.eye(N ** 4 + 1, k=0)
Dx2 = Dx2 / (h ** 3)

# For applying boundary condition
Dx2[0, :] = np.hstack((-1, np.zeros(N ** 4)))
Dx2[N ** 4, :] = np.hstack((np.zeros(N ** 4), -1))


# define FineFunc that value of r0 changes each loop in order to apply the boundary condition
def solveFullSystem(start, end):
    r0 = np.hstack([start, np.zeros(N ** 4 - 1), end])

    def fullSystemFunc(u):
        return u * (Dx1 @ u) - Dx2 @ u - r0

    return fsolve(fullSystemFunc, np.linspace(start, end, N ** 4 + 1))


# testing interpolation, using (tanh(x1),tanh(x2)) with x1,x2 runs through [-5,5]
testing_x = []
testing_y = []
full_sol = []
time_used = []

for i in range(20):
    for j in range(20):
        # boundary condition for the target problem, u(-1)=a, u(1)=b
        a = 1.5 * np.tanh(0.2 * (-5 + i) + 0.1)
        b = 1.5 * np.tanh(0.2 * (-5 + j) + 0.2)
        r0 = np.hstack([a, np.zeros(N ** 4 - 1), b])

        # calculating full system solution and record the time needed
        print("solving Full system", i, j)
        start_solving_full_system = time.time()
        sol = solveFullSystem(a, b)
        end_solving_full_system = time.time()
        print("end of solving Full system", i, j)

        tmp = []
        for coarse_point in range(N + 1):
            tmp.append(sol[coarse_point * N ** 2])
        coarse_sol = np.array(tmp)

        testing_x.append(np.array([a, b]))
        testing_y.append(np.array(coarse_sol))
        full_sol.append(sol)
        time_used.append(end_solving_full_system - start_solving_full_system)
        print("x: ", np.array([a, b]))
        print("y: ", np.array(coarse_sol))
        print("full sol: ", np.array(sol))
        print("time: ", end_solving_full_system - start_solving_full_system)

np.savetxt('data/testing_data_x_1000*10intervals.txt', testing_x, delimiter=',')
np.savetxt('data/testing_data_target_1000*10intervals.txt', testing_y, delimiter=',')
np.savetxt('data/testing_data_full_sol_1000*10intervals.txt', full_sol, delimiter=',')
np.savetxt('data/testing_data_time_used_1000*10intervals.txt', time_used, delimiter=',')

#####################################################################
# Test the code in local computer by using a small dimension (100*10 instead of 1000*10)

'''
import numpy as np
import time
from scipy.optimize import fsolve

# number of intervals, the number of intervals of fine grid is N*N^2 (number of points are N+1 and N^2+1 respectively)
N = 10
# step size on fine grid
h = 2 / (N ** 3)
# assigning derivative matrix
Dx1 = np.eye(N ** 3 + 1, k=1) - np.eye(N ** 3 + 1, k=-1)
Dx1 = Dx1 / (2 * h)
# First and last row of Dx1 is zero for applying boundary condition
Dx1[0, :] = np.zeros(N ** 3 + 1)
Dx1[N ** 3, :] = np.zeros(N ** 3 + 1)

Dx2 = np.eye(N ** 3 + 1, k=1) + np.eye(N ** 3 + 1, k=-1) - 2 * np.eye(N ** 3 + 1, k=0)
Dx2 = Dx2 / (h ** 3)

# For applying boundary condition
Dx2[0, :] = np.hstack((-1, np.zeros(N ** 3)))
Dx2[N ** 3, :] = np.hstack((np.zeros(N ** 3), -1))


# define FineFunc that value of r0 changes each loop in order to apply the boundary condition
def solveFullSystem(start, end):
    r0 = np.hstack([start, np.zeros(N ** 3 - 1), end])

    def fullSystemFunc(u):
        return u * (Dx1 @ u) - Dx2 @ u - r0

    return fsolve(fullSystemFunc, np.linspace(start, end, N ** 3 + 1))


# testing interpolation, using (tanh(x1),tanh(x2)) with x1,x2 runs through [-5,5]
testing_x = []
testing_y = []
full_sol = []
time_used = []

for i in range(20):
    for j in range(20):
        # boundary condition for the target problem, u(-1)=a, u(1)=b
        a = 1.5 * np.tanh(0.2 * (-5 + i) + 0.1)
        b = 1.5 * np.tanh(0.2 * (-5 + j) + 0.2)
        r0 = np.hstack([a, np.zeros(N ** 3 - 1), b])

        # calculating full system solution and record the time needed
        print("solving Full system", i, j)
        start_solving_full_system = time.time()
        sol = solveFullSystem(a, b)
        end_solving_full_system = time.time()
        print("end of solving Full system", i, j)

        tmp = []
        for coarse_point in range(N + 1):
            tmp.append(sol[coarse_point * N ** 2])
        coarse_sol = np.array(tmp)

        testing_x.append(np.array([a, b]))
        testing_y.append(np.array(coarse_sol))
        full_sol.append(sol)
        time_used.append(end_solving_full_system - start_solving_full_system)
        print("x: ", np.array([a, b]))
        print("y: ", np.array(coarse_sol))
        print("full sol: ", np.array(sol))
        print("time: ", end_solving_full_system - start_solving_full_system)

np.savetxt('data/testing_data_x_100*10intervals.txt', testing_x, delimiter=',')
np.savetxt('data/testing_data_target_100*10intervals.txt', testing_y, delimiter=',')
np.savetxt('data/testing_data_full_sol_100*10intervals.txt', full_sol, delimiter=',')
np.savetxt('data/testing_data_time_used_100*10intervals.txt', time_used, delimiter=',')
'''