from neuron import *
import numpy as np

class LayerOfNeuron:
    def __init__(self, num, inputs, f): # создаем список нейронов
        self.neurons = [Neuron(inputs, f) for i in range(num)]

    def set_val_synapses(self, val): # присвоить значение val всем синаптическим весам
        for n in self.neurons:
            n.synapses = np.linspace(val, val, len(n.synapses))  # len(n.synapses) чисел от val до val включительно

    def set_val_synapse(self, n, s, val): # присвоить значение val синаптическому весу s нейрона n
        if n < 0 or n >= len(neurons):             raise LookupError("LayerOfNeuron.set_val_synapse: neuron is not found!")
        if s < 0 or s >= len(neurons[n].synapses): raise LookupError("LayerOfNeuron.set_val_synapse: synapse is not found!")

        neurons[n].synapses[s] = val

    def set_val_s0(self, n, val): # присвоить значение val сдвигу нейрона n
        if n < 0 or n >= len(neurons): raise LookupError("LayerOfNeuron.set_val_s0: neuron is not found!")

        neurons[n].s0 = val