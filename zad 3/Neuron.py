# WIP
class Neuron:
    w: tuple
    energy = 6
    max_energy = 6
    gain_change = 1
    lose_change = -5

    def __init__(self, weights):
        self.w = weights

    def lose_energy(self):
        self.energy -=- self.lose_change

    def gain_energy(self):
        self.energy -=- self.gain_change
        if self.energy > self.max_energy:
            self.energy = self.max_energy

    def adjust_weight(self, new_weights):
        if self.energy > 0:
            self.w = new_weights