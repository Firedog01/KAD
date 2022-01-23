from data_generation import generate_data
import random
from kohonen import commit_kohonen
import numpy as np

def get_data(ver: int):
    if ver == 0:
        return generate_data((0, 0), 2, 200)
    if ver == 1:
        data_0 = generate_data((-3, 0), 1, 100)
        data_1 = generate_data((3, 0), 1, 100)
        for row in data_1:
            data_0.append(row)
        random.shuffle(data_0)
        return data_0


def test_neuron_count():
    for i in range(2, 20, 2):
        q_err = commit_kohonen(data, i, f_lambda, f_eta)
        print(i, " neuronów: ", q_err)




def test_learning_params(n_iterations):
    q_errs = []
    for i in range(n_iterations):
        q_err, n_dead_nodes = commit_kohonen(data, 20, f_lambda, f_eta)
        q_errs.append(q_err)
    q_errs = np.array(q_errs)

    avg = sum(q_errs) / len(q_errs)
    print("średnia błędu kwantyzacji:", avg)
    variance = np.sum((q_errs - avg) ** 2) / len(q_errs)
    std_dev = np.sqrt(variance)
    print("odchylenie standardowe błędu kwantyzacji:", std_dev)
    print("wartość minimalna błędu", min(q_errs))




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

    """
    data = get_data(1)

    def f_lambda(i):
        return np.exp(-i ** 1.5 * 0.0005) * 2

    def f_eta(i):
        return np.exp(-i ** 1.5 * 0.0005) * 0.5

    ret = commit_kohonen(data, 20, f_lambda, f_eta)
    print(ret)
    # test_learning_params(100)


"""
sources:
https://www.analyticsvidhya.com/blog/2021/04/k-means-clustering-simplified-in-python/
https://realpython.com/k-means-clustering-python/
"""
