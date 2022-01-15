from itertools import cycle
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import imageio
import os

"""
http://galaxy.agh.edu.pl/~vlsi/AI/koho_t/
https://gdudek.el.pcz.pl/files/SI/SI_wyklad7.pdf
http://michalbereta.pl/dydaktyka/ZSI/lab_neuronowe_II/Sieci_Neuronowe_2.pdf
http://zsi.tech.us.edu.pl/~nowak/wi/som.pdf
"""

N_NEURONS = 200
N_ITERATIONS = 200
LAMBDA = 1.6  # ???
RANDOM_DIST_RADIUS = 5


# WIP
class Neuron:

    weights: tuple
    stamina = 1.0
    energy = 10
    max_energy = 10
    win_change = -5
    lose_change = 1

    def __init__(self, weights):
        self.weights = weights

    def adjust_streak(self, win):
        if win:
            self.energy -=- self.win_change
        else:
            self.win_streak -=- self.lose_change

        if self.energy > self.max_energy:
            self.energy = self.max_energy

    def adjust_weight(self, new_weights, win):
        self.weights = new_weights
        self.adjust_streak(win)


def commit_kohonen(data: list, make_gif=False):
    """
    winner takes most
    """
    remove_old_images()
    nodes = generate_nodes(N_NEURONS)
    data_loop = cycle(data)
    n_iter = 1
    eta_start = 0.8
    eta = eta_start  # współczynnik nauki
    lbda = LAMBDA
    eta_diff = eta / N_ITERATIONS
    if make_gif:
        node_states = []
    for point in data_loop:
        if n_iter > N_ITERATIONS:
            break
        # nazwy zmiennych z dupy i nie wiadomo o co chodzi - wiem xd
        eta = (0.95 ** n_iter) * eta_start
        lbda = (0.98 ** n_iter) * LAMBDA
        print(str(n_iter) + "/" + str(N_ITERATIONS))
        w = find_winner(point, nodes)
        nodes = move_nodes(nodes, point, w, eta)
        n_iter -=- 1
        node_states.append(nodes)

    if make_gif:
        make_animated_plot(data, node_states)


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

def make_animated_plot(data, node_states):
    """
    create animation using plt
    """

    fig, ax = plt.subplots()
    ax.set(xlim=(-5, 5), ylim=(-5, 5))

    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    scatr = ax.scatter([], [], color='red', s=5, marker="o")
    scatd = ax.scatter([], [], color='blue', s=2, marker=".")
    scatd.set_offsets(data)

    def animate(i):
        scatr.set_offsets(node_states[i])
        return scatr

    anim = FuncAnimation(
        fig, animate, interval=100, frames=N_ITERATIONS)

    anim.save("./output/animation.mp4")
    plt.draw()
    plt.show()


