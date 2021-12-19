from pandas import read_csv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


# data is array got from csv file
# main function
def do_calculations(data, funcs):
    regression = find_reg(data, funcs)
    print_A(regression)
    # print(data)
    y_pred = get_y_predicted(regression, data)
    y_real = get_col(data, "y")
    # print(y_pred)
    print("średni błąd kwadratowy:", avg_quad_dif(y_pred, data))
    print("największe odchylenie:", max(get_dif(y_real, y_pred, True)))
    print("współczynnik R**2:", get_R_squared(y_real, y_pred))

    dim = len(data[0])
    # print(dim)
    if dim == 2:
        plot_2d(data, regression)
    elif dim == 3:
        plot_3d(data, regression)
    histogram(get_dif(y_real, y_pred), 10)

    print("")


# ----------- różne -----------
def print_A(regression):
    letter = "a"
    for row in regression:
        print(letter, "=", row[0])
        letter = chr(ord(letter) + 1)


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


def get_val(regression, x1, x2=None):
    ret = 0
    for cell in regression:
        if x2 is None:
            ret += cell[0] * cell[1](x1)
        else:
            ret += cell[0] * cell[1](x1, x2)
    return ret


def get_y_predicted(regression, data):
    y_predicted = []
    for row in data:
        if len(row) == 2:
            y_predicted.append(get_val(regression, row[0]))
        elif len(row) == 3:
            y_predicted.append(get_val(regression, row[0], row[1]))
    return y_predicted


def get_dif(y_real, y_pred, absolute=False):
    err = []
    for i in range(len(y_real)):
        if absolute:
            err.append(abs(y_real[i] - y_pred[i])[0])
        else:
            err.append((y_real[i] - y_pred[i])[0])
    return err


# ----------- matematyczne -----------
def avg_quad_dif(y_predicted, data):
    val = 0
    for idx, row_real in enumerate(data):
        y_pred = y_predicted[idx]
        if len(row_real) == 2:
            val += (y_pred - row_real[1]) ** 2
        elif len(row_real) == 3:
            val += (y_pred - row_real[2]) ** 2
    # return np.sqrt(val / len(data))
    return val / len(data)


def var(data, avg):
    variance = 0
    for num in data:
        variance += (num - avg) ** 2
    return float(variance / len(data))


# ----------- współczynnik R**2 -----------
def get_R_squared(y_real, y_pred):
    avg = sum(y_real) / len(y_real)
    var_err = var(y_pred, avg)
    var_e = var(y_real, avg)
    # print(var_err, var_e)
    return var_err / var_e


# ----------- regresja -----------
def find_reg(data, funcs):
    col_x = get_col(data, "x")
    dim = len(data[0])  # zakładamy, że są to macierze
    Y = get_col(data, "y")
    X = np.array([])
    for func in funcs:
        temp_arr = []
        for row in col_x:
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


def histogram(dif, bins):
    plt.hist(dif, bins=bins)
    plt.show()


# ----------- wczytywanie -----------
def get_data(filename):
    reader = read_csv(filename, header=None)
    return np.array(reader.values.tolist())
