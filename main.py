import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
import src

# Dla każdego modelu przedstaw:
# ? sposób przygotowania danych do zastosowania prostej regresji liniowej
# + wyznaczone wartości parametrów
# + wykres przedstawiający modelowaną funkcję(-) na tle danych punktów(+)
# + średni błąd kwadratowy dotyczący wartości funkcji w danych punktach
# + największą wartość odchylenia wartości funkcji od danych punktów
# + wartość współczynnika R**2
# - histogram odchyleń wartości funkcji od danych
# - (*) test hipotezy statystycznej, że błędy mają rozkład normalny (test chi-kwadrat Pearsona lub test Shapiro-Wilka)
# - komentarz na temat przydatności zastosowania rozważanego modelu

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
print("data1 f(X) = a * X")
src.do_calculations(data1, model1)
# print("data1 f(X) = a * X + b")
# src.do_calculations(data1, model2)
# print("data1 f(X) = a * X**2 + b * sin(X) + c")
# src.do_calculations(data1, model3)
#
print("data2 f(X) = a * X")
src.do_calculations(data2, model1)
# print("data2 f(X) = a * X + b")
# src.do_calculations(data2, model2)
# print("data2 f(X) = a * X**2 + b * sin(X) + c")
# src.do_calculations(data2, model3)

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
# print("data3 f(X1, X2) = a * X1 + b * X2 + c")
# src.do_calculations(data3, model4)
# print("data3 f(X1, X2) = a * X1**2 + b * X1*X2 + c * X2**2 + d * X1 + e * X2 + f")
# src.do_calculations(data3, model5)
#
# print("data4 f(X1, X2) = a * X1 + b * X2 + c")
# src.do_calculations(data4, model4)
# print("data4 f(X1, X2) = a * X1**2 + b * X1*X2 + c * X2**2 + d * X1 + e * X2 + f")
# src.do_calculations(data4, model5)


