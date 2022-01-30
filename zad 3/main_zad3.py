import math

from data_generation import generate_data
import random
from kohonen import commit_kohonen
import numpy as np
import matplotlib.pyplot as plt


def get_data():
    if ver == 0:
        return generate_data((0, 0), 2, 200)
    if ver == 1:
        data_0 = generate_data((-3, 0), 1, 100)
        data_1 = generate_data((3, 0), 1, 100)
        for row in data_1:
            data_0.append(row)
        random.shuffle(data_0)
        return data_0


def count_deads(neurons):
    frags = 0
    if ver == 0:
        for neuron in neurons:
            if math.dist(neuron.w, (0, 0)) > 2:
                frags -= - 1
    if ver == 1:
        for neuron in neurons:
            if math.dist(neuron.w, (-3, 0)) > 1 and math.dist(neuron.w, (3, 0)) > 1:
                frags -= - 1
    return frags


def get_avg_and_std_dev(values):
    avg = sum(values) / len(values)
    variance = np.sum((values - avg) ** 2) / len(values)
    std_dev = np.sqrt(variance)
    return avg, std_dev


def test_neuron_count():
    q_errs = []
    values = []
    for n_neurons in range(2, 21, 2):
        q_err, neurons = commit_kohonen(data, n_neurons, f_lambda, f_eta, prevent_dead)
        q_errs.append(q_err)
        values.append(str(n_neurons))
    plt.bar(values, q_errs)
    plt.show()


def test_learning_params(n_iterations):
    q_errs = []
    dead_neuron_arr = []
    print("=" * n_iterations)
    for i in range(n_iterations):
        print("-", end='')
        q_err, neurons = commit_kohonen(data, 20, f_lambda, f_eta, rand_radius, prevent_dead)
        q_errs.append(q_err)
        dead_neuron_arr.append(count_deads(neurons))
    print("\n")
    q_errs = np.array(q_errs)
    dead_neuron_arr = np.array(dead_neuron_arr)

    avg_q, std_dev_q = get_avg_and_std_dev(q_errs)
    avg_d, std_dev_d = get_avg_and_std_dev(dead_neuron_arr)
    print("średnia błędu kwantyzacji:", avg_q)
    print("odchylenie standardowe błędu kwantyzacji:", std_dev_q)
    print("wartość minimalna błędu", min(q_errs))
    print("średnia liczba martwych neuronów", avg_d)
    print("odchylenie standardowe martwych neuronów", std_dev_d)


if __name__ == "__main__":
    """
     lambda:
        współczynnik do funkcji sąsiedztwa,
        by promień sąsiedztwa zmniejszał się z biegiem iteracji
    eta:
        jak bardzo neuron będzie się zbliżał do punktu
        
    dobre kombinacje ver 0:
        l = 15 / i, e = 0.5
        l = 10 / i, e = 0.5 - 0.0020 * i
        l = 10 / i ** 1.2, e = np.exp(-i ** 2 * 0.00003) * 0.5
        l = np.exp(-i ** 1.5 * 0.0005) * 2, e = np.exp(-i ** 1.5 * 0.0005) * 0.5
        
    dobre kombinacje ver 1:
        l = 12 / i, e = 0.5
        l = np.exp(-i ** 1.7 * 0.0009) * 2, e = 0.5 - 0.0020 * i
        l = np.exp(-i ** 1.8 * 0.0009) * 2.1, e = 0.5 - 0.0020 * i
        l = np.exp(-i ** 1.8 * 0.0009) * 2.05, e = 0.5 - 0.0020 * i
        
    """
    ver = 0
    data = get_data()
    rand_radius = 3
    prevent_dead = False

    def f_lambda(i):
        return 12 / i

    def f_eta(i):
        return 0.5


    q_err1, neurons1 = commit_kohonen(data, 20, f_lambda, f_eta, rand_radius, prevent_dead=prevent_dead, make_gif=True)
    print(q_err1, count_deads(neurons1))

    # test_neuron_count()
    test_learning_params(100)


"""
sources:
https://www.analyticsvidhya.com/blog/2021/04/k-means-clustering-simplified-in-python/
https://realpython.com/k-means-clustering-python/
"""
