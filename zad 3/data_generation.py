from math import pi, cos, sin, sqrt
from random import random
from typing import Tuple
import numpy as np


def get_random_point(center: Tuple[float, float], radius: float) -> Tuple[float, float]:
    shift_x, shift_y = center
    a = random() * 2 * pi
    r = radius * sqrt(random())
    return r * cos(a) + shift_x, r * sin(a) + shift_y

def generate_data(center: tuple, radius: float, count: int):
    points = list()
    for i in range(count):
        point = get_random_point(center, radius)
        points.append(point)
    return points
