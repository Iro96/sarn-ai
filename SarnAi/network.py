from .memory import MemoryBank
from .utils import plasticity_update, neurogenesis_check

class SARNNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.memory = MemoryBank()

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        self.memory.store(x)
        return x

    def adapt(self):
        for layer in self.layers:
            plasticity_update(layer)
        if neurogenesis_check(self.memory):
            self._add_neuron()

    def _add_neuron(self):
        self.layers[-1].add_neuron()
