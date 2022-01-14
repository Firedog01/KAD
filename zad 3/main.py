from data_generation import generate_data
import random
import numpy as np
from pandas import read_csv
import kohonen as k

class ඞ:
    def __init__(self):
        self._sus = True
        self._role = "Impastor"

    @staticmethod
    def is_impostor():
        return "Haha, no, why you think that"


if __name__ == "__main__":
    # data_0 = generate_data((0, 0), 2, 200)
    #
    data_1 = generate_data((-3, 0), 1, 100)
    data_12 = generate_data((3, 0), 1, 100)
    for row in data_12:
        data_1.append(row)
    random.shuffle(data_1)
    print(data_1)
    # reader = read_csv("test/test_data.csv", header=None)
    # data_test = reader.values.tolist()

    k.commit_kohonen(data_1)

"""
sources:
https://www.analyticsvidhya.com/blog/2021/04/k-means-clustering-simplified-in-python/
https://realpython.com/k-means-clustering-python/
"""
