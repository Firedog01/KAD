import numpy as np
import src


# Wczytywanie
data1 = src.get_data("data1.csv")
data2 = src.get_data("data2.csv")
data3 = src.get_data("data3.csv")
data4 = src.get_data("data4.csv")

model1 = [
    lambda x: x
]
model2 = [
    lambda x: x,
    lambda x: 1
]
model3 = [
    lambda x: x ** 2,
    lambda x: np.sin(x),
    lambda x: 1,
]
model4 = [
    lambda x1, x2: x1,
    lambda x1, x2: x2,
    lambda x1, x2: 1,
]
model5 = [
    lambda x1, x2: x1 ** 2,
    lambda x1, x2: x1 * x2,
    lambda x1, x2: x2 ** 2,
    lambda x1, x2: x1,
    lambda x1, x2: x2,
    lambda x1, x2: 1
]

data_options = {
    1: 'data 1',
    2: 'data 2',
    3: 'data 3',
    4: 'data 4',
}

model123_options = {
    1: 'f(X) = a * X',
    2: 'f(X) = a * X + b',
    3: 'f(X) = a * X**2 + b * sin(X) + c',
}

model45_options = {
    4: 'f(X1, X2) = a * X1 + b * X2 + c',
    5: 'f(X1, X2) = a * X1**2 + b * X1*X2 + c * X2**2 + d * X1 + e * X2 + f',
}

def print_options(options):
    for key in options.keys():
        print("[", key, "]", "-", options[key])
    return ": "


def calc(d, m):
    _data = None
    _model = None
    if d == 1:
        _data = data1
    elif d == 2:
        _data = data2
    elif d == 3:
        _data = data3
    else:
        _data = data4

    if m == 1:
        _model = model1
    elif m == 2:
        _model = model2
    elif m == 3:
        _model = model3
    elif m == 4:
        _model = model4
    else:
        _model = model5
    src.do_calculations(_data, _model)


if __name__ == "__main__":
    while True:
        print("Wybierz zbiór danych do analizowania")
        data = int(input(print_options(data_options)))
        print("Wybierz model")
        print(data)
        if data <= 2:
            model = int(input(print_options(model123_options)))
        else:
            model = int(input(print_options(model45_options)))
        calc(data, model)






# print("data1 f(X) = a * X")
# src.do_calculations(data1, model1)
# print("data1 f(X) = a * X + b")
# src.do_calculations(data1, model2)
# print("data1 f(X) = a * X**2 + b * sin(X) + c")
# src.do_calculations(data1, model3)
#
# print("data2 f(X) = a * X")
# src.do_calculations(data2, model1)
# print("data2 f(X) = a * X + b")
# src.do_calculations(data2, model2)
# print("data2 f(X) = a * X**2 + b * sin(X) + c")
# src.do_calculations(data2, model3)

# print("data3 f(X1, X2) = a * X1 + b * X2 + c")
# src.do_calculations(data3, model4)
# print("data3 f(X1, X2) = a * X1**2 + b * X1*X2 + c * X2**2 + d * X1 + e * X2 + f")
# src.do_calculations(data3, model5)
#
# print("data4 f(X1, X2) = a * X1 + b * X2 + c")
# src.do_calculations(data4, model4)
# print("data4 f(X1, X2) = a * X1**2 + b * X1*X2 + c * X2**2 + d * X1 + e * X2 + f")
# src.do_calculations(data4, model5)


