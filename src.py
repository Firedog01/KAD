from pandas import read_csv
import matplotlib.pyplot as plt
import numpy as np


# Variance
def var(data):
    avg = np.average(data)
    variance = 0
    for num in data:
        variance += (num - avg) ** 2
    return variance / len(data)


# Covariance
def cov(data_x, data_y):
    avg_x = np.average(data_x)
    avg_y = np.average(data_y)
    len_data = len(data_x)
    covariance = 0
    for n in range(0, len_data):
        covariance += data_x[n] * data_y[n] - avg_x * avg_y
    return covariance / len_data


# X and Y need to be 2D arrays!
def find_A(X, Y, has_const=True):
    if has_const:
        lenx = len(X[:, 0])
        ones = [1.0] * lenx
        col_ones = np.array(ones)[np.newaxis].T
        X = np.concatenate((col_ones, X), axis=1)
        # print(X)
    A = np.linalg.inv(X.transpose().dot(X))\
        .dot(X.transpose().dot(Y))

    tuple_A = tuple()
    for row in A:
        tuple_A = tuple_A + (row[0],)
    return tuple_A


    # (X^T * X)^-1 * X^T * Y

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

# https://www.urbandictionary.com/define.php?term=F.U.V.
def fuv():
    pass


def calc_A_and_plot(data, funcs):
    A = alt_find_A(data, funcs)
    dim = len(data[0])
    if len(A) == 1 and dim == 2:
        a = A[0]
        print("f(x) =", a, "* x")
        plot_2d(data, lambda x: a * x)
    if len(A) == 2 and dim == 2:
        a = A[1]
        b = A[0]
        print("f(x) =", a, "* x +", b)
        plot_2d(data, lambda x: a * x + b)


def alt_find_A(data, funcs):
    col_x = get_col(data, "x")
    len_x = len(col_x[0])  # zakładamy, że są to macierze
    Y = get_col(data, "y")
    X = np.array([])
    for func in funcs:
        if len_x == 1:
            temp_arr = []
            for row in col_x:
                temp_arr.append(func(row[0]))
            temp_col = np.array(temp_arr)[np.newaxis].T

            if len(X) == 0:
                X = temp_col
            else:
                X = np.concatenate((X, temp_col), axis=1)
        elif len_x == 2:
            pass  # todo

    A = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose().dot(Y))

    tuple_A = tuple()
    for row in A:
        tuple_A = tuple_A + (row[0],)
    return tuple_A




def get_col(data, mode):
    len_d = len(data[0]) - 1
    if mode == "x":
        if len_d == 1:
            return np.array(data[:, 0])[np.newaxis].T
        else:
            return data[:, :len_d].T
    elif mode == "y":
        return np.array(data[:, len_d])[np.newaxis].T
    return None

# ----------- wczytywanie -----------
def get_data(filename):
    reader = read_csv(filename, header=None)
    return np.array(reader.values.tolist())

# ----------- rysowanie -----------
def plot_2d(data, f):
    for row in data:
        plt.scatter(row[0], row[1], color="black", marker=".", s=10)  # s - size
    x_points = []
    y_points = []
    max_val = int(max(data[:, 0]) * 100)
    for point in range(0, max_val):
        x = point / 100
        x_points.append(x)
        y_points.append(f(x))
    plt.plot(x_points, y_points)
    plt.show()


def plot_3d(data, label):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for row in data:
        ax.scatter(row[0], row[1], row[2], label=label, color="blue", marker=".", s=10)
    plt.show()
