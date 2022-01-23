from data_generation import generate_data
import random
import kohonen as k
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


if __name__ == "__main__":
    data = get_data(1)

    def f_lambda(n_iter):
        return np.exp(-n_iter ** 2 * 1) * 2.1

    def f_eta(n_iter):
        return np.exp(-n_iter ** 2 * 0.005)

    k.commit_kohonen(data, 20, f_lambda, f_eta, make_gif=True)

"""
sources:
https://www.analyticsvidhya.com/blog/2021/04/k-means-clustering-simplified-in-python/
https://realpython.com/k-means-clustering-python/
"""
