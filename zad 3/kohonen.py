from itertools import cycle
import numpy as np
import math

import random
import Neuron
from make_gif import *

"""
http://galaxy.agh.edu.pl/~vlsi/AI/koho_t/
https://gdudek.el.pcz.pl/files/SI/SI_wyklad7.pdf
http://michalbereta.pl/dydaktyka/ZSI/lab_neuronowe_II/Sieci_Neuronowe_2.pdf
http://zsi.tech.us.edu.pl/~nowak/wi/som.pdf
"""


def commit_kohonen(data: list, n_neurons: int, f_lambda, f_eta, rand_radius=5, prevent_dead=True, make_gif=False, graph_size=5):
    """
    winner takes most
    """
    N_ITERATIONS = 500

    neurons = generate_nodes(n_neurons, rand_radius)
    data_loop = cycle(data)
    # print("Przypadkowe rozmieszczanie neuronów")
    # print("błąd kwantyzacji na początku", quantisation_err(data, neurons))

    node_states = []  # to make an animation
    n_iter = 1
    last_max_distance = math.inf
    for point in data_loop:
        if n_iter > N_ITERATIONS:  # warunek ilości cykli
            break
        if last_max_distance <= 0:  # warunek przesunięcia
            break

        eta = f_eta(n_iter)
        lbda = f_lambda(n_iter)

        w = find_winner(point, neurons, prevent_dead)
        max_dis = move_nodes(neurons, point, w, eta, lbda)
        if max_dis < last_max_distance:
            last_max_distance = max_dis

        node_states.append([i.w for i in neurons])
        n_iter -= - 1

    q_err = quantisation_err(data, neurons)
    # print("proces nauki zakończony w", n_iter - 1, "iteracjach")

    save_frame(0, data, neurons, graph_size)

    if make_gif:
        make_animated_plot(data, node_states, graph_size)

    return q_err, neurons


def generate_nodes(n_nodes: int, r: float):
    """
    generates given number of nodes in a square
    """
    nodes = []
    for i in range(n_nodes):
        x = (random.random() - 0.5) * 2 * r
        y = (random.random() - 0.5) * 2 * r
        nodes.append(Neuron.Neuron((x, y)))
    return nodes


def find_winner(x: tuple, neurons: list, prevent_dead) -> Neuron:
    """
    find the winner
    """
    d = math.dist(x, neurons[0].w)
    ret_neuron = neurons[0]

    for i in neurons:
        if prevent_dead:
            i.gain_energy()
        di = math.dist(x, i.w)
        if di < d:
            d = di
            ret_neuron = i

    if prevent_dead:
        ret_neuron.lose_energy()
    return ret_neuron


def move_nodes(nodes: list, x: tuple, w: Neuron, eta: float, lbda: float):
    """
    moves winner node colser to point x.
    Additionally, according to Gauss proximity function moves neighbours to same point
    """
    max_dis = 0
    for node in nodes:
        dis = node.adjust_weight(node.w + eta * gauss_proximity(node.w, w.w, lbda) * np.subtract(x, node.w))
        if dis > max_dis:
            max_dis = dis
    return max_dis


def gauss_proximity(i: tuple, w: tuple, lbda: float) -> float:
    """
    G(i, x) = exp( -d^2(i, w) / (2 lambda^2) )
    """
    if lbda == 0:
        print("lambda zero")
    return np.exp(
        -(math.dist(i, w) ** 2)
        /
        (2 * lbda ** 2)
    )


def quantisation_err(data: list, neurons: list) -> float:
    err = 0
    for point in data:
        min_dist = math.inf
        for neuron in neurons:
            d = math.dist(point, neuron.w)
            if d < min_dist:
                min_dist = d
        err += min_dist
    return err / len(data)


def get_min_dist(point: tuple, neurons: list) -> float:
    min_dist = math.inf
    for neuron in neurons:
        d = math.dist(point, neuron.w)
        if d < min_dist:
            min_dist = d
    return min_dist
