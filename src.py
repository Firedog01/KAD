from pandas import read_csv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
# import inspect as ins


# data is array got from csv file
# main function
def do_calculations(data, funcs):
    regression = find_reg(data, funcs)
    # TODO calculations on regression

    dim = len(data[0])
    if dim == 2:
        plot_2d(data, regression)
    elif dim == 3:
        plot_3d(data, regression)


def var(data):
    avg = np.average(data)
    variance = 0
    for num in data:
        variance += (num - avg) ** 2
    return variance / len(data)


def cov(data_x, data_y):
    avg_x = np.average(data_x)
    avg_y = np.average(data_y)
    len_data = len(data_x)
    covariance = 0
    for n in range(0, len_data):
        covariance += data_x[n] * data_y[n] - avg_x * avg_y
    return covariance / len_data


# COST (średni błąd kwadratowy)
def cost(array_x, array_y, array_a):
    Z = [array_x.dot(array_a)]
    Z_Y = Z - array_y
    Z_Y = np.linalg.matrix_power(Z_Y, 2)
    return np.avg(Z_Y)


# Max z ERR (krotka błędów)
def err_dev(f_points, y_points):
    err = f_points - y_points
    return max(err)


def fuv():
    pass

# ----------- regresja -----------
def find_reg(data, funcs):
    col_x = get_col(data, "x")
    dim = len(data[0])  # zakładamy, że są to macierze
    # print(dim, col_x)
    Y = get_col(data, "y")
    X = np.array([])
    for func in funcs:
        temp_arr = []
        for row in col_x:
            # print(row)
            if dim == 2:
                temp_arr.append(func(row[0]))
            elif dim == 3:
                temp_arr.append(func(row[0], row[1]))
        temp_col = np.array(temp_arr)[np.newaxis].T
        if len(X) == 0:
            X = temp_col
        else:
            X = np.concatenate((X, temp_col), axis=1)

    A = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(Y))  # (Xᵀ·X)⁻¹·(Xᵀ·Y)
    A = A.T[0]

    regression = []
    for idx, alpha in enumerate(A):
        regression.append([alpha, funcs[idx]])
    return regression


def get_col(data, mode):
    dim = len(data[0])
    if mode == "x":
        if dim == 2:
            return np.array(data[:, 0])[np.newaxis].T
        elif dim == 3:
            return data[:, :(dim - 1)]
    elif mode == "y":
        return np.array(data[:, (dim - 1)])[np.newaxis].T
    return None


# ----------- rysowanie -----------
def plot_2d(data, reg):
    for row in data:
        plt.scatter(row[0], row[1], color="black", marker=".", s=10)  # s - size
    x_points = []
    y_points = []
    max_val = int(max(data[:, 0]) * 100)
    min_val = int(min(data[:, 0]) * 100)
    for point in range(min_val, max_val):
        x = point / 100
        x_points.append(x)
        y_points.append(get_val(reg, x))
    plt.plot(x_points, y_points)
    plt.show()


def plot_3d(data, reg):
    fig = plt.figure(clear=True)
    ax = fig.add_subplot(projection='3d')
    for row in data:
        ax.scatter(row[0], row[1], row[2], color="blue", marker=".", s=10)
    x_points = []
    y_points = []  # z and y swapped for convenience
    z_points = []
    min_x = min(data[:, 0])
    max_x = max(data[:, 0])
    min_y = min(data[:, 1])
    max_y = max(data[:, 1])

    (x, y) = np.meshgrid(np.linspace(min_x, max_x + 0.1),
                         np.linspace(min_y, max_y + 0.1))
    z = get_val(reg, x, y)
    ax.plot_surface(x, y, z, cmap=cm.hot)
    plt.tight_layout()
    plt.show()


def get_val(regression, x1, x2=None):
    ret = 0
    for cell in regression:
        if x2 is None:
            ret += cell[0] * cell[1](x1)
        else:
            ret += cell[0] * cell[1](x1, x2)
    return ret


# ----------- wczytywanie -----------
def get_data(filename):
    reader = read_csv(filename, header=None)
    return np.array(reader.values.tolist())
