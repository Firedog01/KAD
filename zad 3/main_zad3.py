from data_generation import generate_data
import random
import kohonen as k

if __name__ == "__main__":
    # data = generate_data((0, 0), 2, 200)
    #
    data = generate_data((-3, 0), 1, 100)
    data_1 = generate_data((3, 0), 1, 100)
    for row in data_1:
        data.append(row)
    random.shuffle(data)

    k.commit_kohonen(data, make_gif=True)

"""
sources:
https://www.analyticsvidhya.com/blog/2021/04/k-means-clustering-simplified-in-python/
https://realpython.com/k-means-clustering-python/
"""
