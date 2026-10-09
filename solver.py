import numpy as np
import matplotlib.pyplot as plt

#f: lambda func
def RK4(f, f_init, t0, t_end, h):
    n_steps = int((t_end - t0) / h)

    values = np.zeros((n_steps + 1, *np.shape(f_init)))
    time = np.zeros(n_steps + 1)

    values[0] = f_init
    time[0] = t0

    t = t0
    ft = f_init

    for i in range(n_steps):
        k1 = f(t, ft)
        k2 = f(t + 0.5 * h, ft + k1 * 0.5 * h)
        k3 = f(t + 0.5 * h, ft + k2 * 0.5 * h)
        k4 = f(t + h, ft + k3 * h)

        ft = ft + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t += h

        values[i + 1] = ft
        time[i + 1] = t

    return values, time

# #f = lambda t, y: t - y
# f = lambda t, y: np.array([y[0] + y[1], y[0] - y[1]])
# t0 = 0
# #y0 = 1
# y0 = np.array([1, 0])
# h = 0.05
# t_end = 0.6

# y, t = RK4(f, y0, t0, t_end, h)

# # plt.plot(t, y)
# # plt.show()

# x_exact = np.cosh(np.sqrt(2) * t) + np.sinh(np.sqrt(2) * t) / np.sqrt(2)
# y_exact = np.sinh(np.sqrt(2) * t) / np.sqrt(2)

# error_x = np.max(np.abs(y[:, 0] - x_exact))
# error_y = np.max(np.abs(y[:, 1] - y_exact))

# print(error_x)
# print(error_y)

# print(y[:3])