import numpy as np

class Neuron:
    def __init__(self, inputs, f):
        self.__inputs = inputs # количество входов
        self.__f      = f      # передаточная функция

        self.v = 0 # значение последнего рассчитанного локального поля

        self.s0       = (-0.5 + np.random.sample())    # значение сдвига
        self.synapses = np.random.sample(inputs) - 0.5 # веса входов нейрона

        self.r = 1.0 # скорость обучения для нейрона

        self.memory_synapses = np.zeros(inputs)

    def __call__(self, x):
        # if len(x) != len(self.synapses): raise ValueError("Neuron.__call__: incorrect input vector size!")

        self.v = (np.sum(x * self.synapses) + self.s0)
        return self.__f(self.v)