import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
import src

# Dla każdego modelu przedstaw:
# ? sposób przygotowania danych do zastosowania prostej regresji liniowej
# - wyznaczone wartości parametrów
# - wykres przedstawiający modelowaną funkcję(-) na tle danych punktów(+)
# - średni błąd kwadratowy dotyczący wartości funkcji w danych punktach
# - największą wartość odchylenia wartości funkcji od danych punktów
# - wartość współczynnika R**2
# - histogram odchyleń wartości funkcji od danych
# - (*) test hipotezy statystycznej, że błędy mają rozkład normalny (test chi-kwadrat Pearsona lub test Shapiro-Wilka)
# - komentarz na temat przydatności zastosowania rozważanego modelu

# Wczytywanie
data1 = src.get_data("data1.csv")
data2 = src.get_data("data2.csv")
data3 = src.get_data("data3.csv")
data4 = src.get_data("data4.csv")

src.calc_A_and_plot(data1, [
    (lambda x: x)
])
src.calc_A_and_plot(data1, [
    (lambda x: x),
    (lambda x: 1)
])
src.calc_A_and_plot(data1, [
    (lambda x: x * x),
    (lambda x: np.sin(x)),
    (lambda x: 1),
])

src.calc_A_and_plot(data2, [
    (lambda x: x)
])
src.calc_A_and_plot(data2, [
    (lambda x: x),
    (lambda x: 1)
])
src.calc_A_and_plot(data2, [
    (lambda x: x * x),
    (lambda x: np.sin(x)),
    (lambda x: 1),
])

# data_x = data1[:, 0]
# data_y = data1[:, 1]
# # print(data_x, data_y)
# print(src.var(data_x), src.cov(data_x, data_x))
# a = src.cov(data_x, data_y) / src.var(data_x)
# print(a)

