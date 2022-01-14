from itertools import cycle
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import imageio
import os

"""
http://galaxy.agh.edu.pl/~vlsi/AI/koho_t/
https://gdudek.el.pcz.pl/files/SI/SI_wyklad7.pdf
http://michalbereta.pl/dydaktyka/ZSI/lab_neuronowe_II/Sieci_Neuronowe_2.pdf
http://zsi.tech.us.edu.pl/~nowak/wi/som.pdf
"""

N_NEURONS = 5
N_ITERATIONS = 50
LAMBDA = 4  # ???
RANDOM_DIST_RADIUS = 5


def commit_kohonen(data: list):
    """
    winner takes most
    """
    remove_old_images()
    nodes = generate_nodes(5)
    data_loop = cycle(data)
    n_iter = 1
    eta = 1  # współczynnik nauki
    lbda = LAMBDA
    eta_diff = eta / N_ITERATIONS
    save_frame(0, data, nodes)
    for point in data_loop:
        if n_iter > N_ITERATIONS:
            break
        print(str(n_iter) + "/" + str(N_ITERATIONS))
        w = find_winner(point, nodes)
        nodes = move_nodes(nodes, point, w, eta)
        save_frame(n_iter, data, nodes)
        n_iter -=- 1
        eta /= 1.4
        # eta -= eta_diff
        lbda /= 1.7

    make_animation()


def generate_nodes(n_nodes: int):
    """
    generates given number of nodes
    """
    nodes = []
    for i in range(n_nodes):
        x = (random.random() - 0.5) * RANDOM_DIST_RADIUS * 2
        y = (random.random() - 0.5) * RANDOM_DIST_RADIUS * 2
        nodes.append((x, y))
    return nodes


def move_nodes(nodes: list, x: tuple, w: tuple, eta: int):
    """
    for one point moves all according to some wzory
    """
    ret_nodes = []
    for i in nodes:
        ret_nodes.append(i + eta * gauss_proximity(i, w) * np.subtract(x, i))
    return ret_nodes


def gauss_proximity(i: tuple, w: tuple):
    """
    G(i, x) = exp( -d^2(i, w) / (2 lambda^2) )
    """
    return np.exp(
        -(math.dist(i, w)**2)
        /
        (2*LAMBDA**2)
    )


def find_winner(x: tuple, neurons: list) -> tuple:
    """
    find the winner
    """
    d = math.dist(x, neurons[0])
    ret_tuple = neurons[0]

    for i in neurons:
        di = math.dist(x, i)
        if di < d:
            d = di
            ret_tuple = i
    return ret_tuple


""" ------------------------ """


def save_frame(i: int, data: list, neurons: list):
    """
    saves frame for current data and neurons with name i.png
    """
    # filesize 500x500
    fig = plt.figure(figsize=(10, 10), dpi=50)

    # move axis
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    plt.tight_layout(pad=0.2)
    for row in data:
        plt.scatter(row[0], row[1], color="blue", marker=".", s=10)
    for row in neurons:
        plt.scatter(row[0], row[1], color="red", marker="o", s=25)
    plt.xlim([-RANDOM_DIST_RADIUS, RANDOM_DIST_RADIUS])
    plt.ylim([-RANDOM_DIST_RADIUS, RANDOM_DIST_RADIUS])
    path = "images/{:04d}.png".format(i)
    plt.savefig(path)
    plt.close()


def make_animation():
    """
    combines all images from images/, saves gif in output/
    """
    filenames = sorted(os.listdir("images"))
    with imageio.get_writer('output/anim.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread('images/' + filename)
            writer.append_data(image)


def remove_old_images():
    """
    clear images/
    """
    filenames = os.listdir("images")
    for file in filenames:
        os.remove('images/' + file)
