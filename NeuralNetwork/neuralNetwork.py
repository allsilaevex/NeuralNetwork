from layerOfNeuron import *
import pickle
import random
import numpy as np

class NeuralNetwork:
    # с lambda не сохраняется
    #f  = lambda self, x: (1 / (1 + np.exp(-x)))
    #df = lambda self, x: (self.f(x) * (1 - self.f(x)))

    def  f(self, x): # передаточная функция
        try:
            return 1.0 / (1.0 + np.exp(-x))
        except OverflowError:
            print("NeuralNetwork.f: WARNING!!! OverflowError, returned 0.0!")
            return 0.0

    def df(self, x): # производная функции f
        try:
            return self.f(x) * (1.0 - self.f(x))
        except OverflowError:
            print("NeuralNetwork.df: WARNING!!! OverflowError, returned 0.0!")
            return 0.0

    __alpha = 0.0001 # постоянная момента

    def __init__(self, inp, h_layers, out):
        if type(inp)      != int:  raise TypeError("NeuralNetwork.__init__: number of inputs is not 'int'!")
        if type(h_layers) != list: raise TypeError("NeuralNetwork.__init__: list of hidden layers is not 'list'!")
        if type(out)      != int:  raise TypeError("NeuralNetwork.__init__: number of outputs is not 'int'!")

        if inp <= 0:           raise ValueError("NeuralNetwork.__init__: number of inputs <= 0!")
        if len(h_layers) == 0: raise ValueError("NeuralNetwork.__init__: length of list == 0!")
        if 0 in h_layers:      raise ValueError("NeuralNetwork.__init__: in list of hidden layers found 0!")
        if out <= 0:           raise ValueError("NeuralNetwork.__init__: number of outputs <= 0!")

        self.__inp                     = inp           # количество входов
        self.__num_neurons_in_h_layers = sum(h_layers) # количество нейронов в скрытых слоях
        self.__h_layers                = len(h_layers) # количество скрытых слоев
        self.__out                     = out           # количество выходов

        self.hidden_layers = [LayerOfNeuron(h_layers[i], inp if i == 0 else h_layers[i - 1], self.f) for i in range(self.__h_layers)] # создаем скрытые слои
        self.hidden_layers.append(LayerOfNeuron(out, h_layers[len(h_layers) - 1], self.f)) # и выходной считаем за последний скрытый

        self.__init_weights() # переинициализируем веса по некоторой формуле

    def __init_weights(self): # некоторые математические действия над весами
        p    = self.__num_neurons_in_h_layers
        beta = 0.7 * np.power(p, (1.0 / self.__inp))

        sm = 0
        for h in self.hidden_layers:
            for n in h.neurons:
                for s in n.synapses:
                    sm += s * s
        norm_syns = np.sqrt(sm)

        for h in self.hidden_layers:
            for n in h.neurons:
                for s in n.synapses:
                    s = (beta * s) / norm_syns
            n.s0 = random.uniform(-beta, beta)

    def __call__(self, x): # расчет реакции сети на входной вектор x
        if type(x) != list and type(x) != np.ndarray: raise TypeError("NeuralNetwork.__call__: input vector is not 'list' or 'numpy.ndarray'!")
        if len(x)  != self.__inp:                    raise ValueError("NeuralNetwork.__call__: incorrect input vector size!")

        for l in range(self.__h_layers + 1): # по всем скрытым + по выходному
            temp_x = np.fromiter((self.hidden_layers[l].neurons[n](x) for n in range(len(self.hidden_layers[l].neurons))), np.float)
            x = temp_x
        return x

    @classmethod
    def save(cls, network, name_file): # метод сохранения экземпляра сети
        if type(network)   != NeuralNetwork: raise TypeError("NeuralNetwork.save: SAVE ONLY NeuralNetwork OBJECTS!")
        if type(name_file) != str:           raise TypeError("NeuralNetwork.save: file name is invalid!")

        with open(name_file, 'wb') as file:
            pickle.dump(network, file)

    @classmethod
    def load(cls, name_file): # метод загрузки экземпляра сети
        with open(name_file, 'rb') as file:
            network = pickle.load(file)
            if type(network) != NeuralNetwork:
                raise TypeError("LOAD ONLY NeuralNetwork OBJECTS!")
            return network

    def training(self, inp, norm, a): # тренировка методом обратного распространения ошибки(a - скорость обучения)
        # полагаемся на адекватного учителя и не производим 2х лишних сравнений
        #if len(inp)  != self.__inp: raise ValueError("NeuralNetwork.training: incorrect input vector size!")
        #if len(norm) == self.__out: raise ValueError("NeuralNetwork.training: incorrect output vector size!")

        RES_input_layer = inp # выход входного слоя = входному вектору

        RES_hidden_layers = [] # запоминаем выходы всех скрытых слоев(далее выходной слой = последний скрытый)
        for l in range(self.__h_layers + 1):
            temp_inp = np.fromiter((self.hidden_layers[l].neurons[n](inp) for n in range(len(self.hidden_layers[l].neurons))), np.float)
            inp = temp_inp
            RES_hidden_layers.append(inp)

        delta = [[] for i in range(self.__h_layers + 1)] # массив локальных градиентов узлов сети

        E     = norm - RES_hidden_layers[len(RES_hidden_layers) - 1] # сигнал ошибки
        PHI   = np.fromiter((self.df(self.hidden_layers[self.__h_layers].neurons[n].v) for n in range(self.__out)), np.float)

        delta[self.__h_layers] = E * PHI # для выходного слоя

        for l in range(self.__h_layers - 1, -1, -1): # проходим по всем скрытым слоям КРОМЕ последнего
            num_neurons = len(self.hidden_layers[l].neurons)

            PHI = np.fromiter((self.df(self.hidden_layers[l].neurons[n].v) for n in range(num_neurons)), np.float)
            
            temp = np.empty(num_neurons)
            for n in range(num_neurons): # проходим по всем нейронам слоя
                W       = np.fromiter((self.hidden_layers[l + 1].neurons[ne].synapses[n] for ne in range(len(self.hidden_layers[l + 1].neurons))), np.float)
                SUM     = np.sum(delta[l + 1] * W)
                temp[n] = PHI[n] * SUM # расчитываем локальный градиент для нейрона n

            delta[l] = temp

        for l in range(self.__h_layers, 0, -1): # изменяем веса синаптических связей скрытых слоев КРОМЕ первого
            for n in range(len(self.hidden_layers[l].neurons)):
                self.hidden_layers[l].neurons[n].synapses       += a * self.hidden_layers[l].neurons[n].r * ((1 - self.__alpha) * delta[l][n] * RES_hidden_layers[l - 1] + self.__alpha * self.hidden_layers[l].neurons[n].memory_synapses)
                self.hidden_layers[l].neurons[n].memory_synapses = self.hidden_layers[l].neurons[n].synapses

        for n in range(len(self.hidden_layers[0].neurons)): # изменяем веса синаптических связей первого слоя
                self.hidden_layers[0].neurons[n].synapses       += a * self.hidden_layers[0].neurons[n].r * ((1 - self.__alpha) * delta[0][n] * RES_input_layer + self.__alpha * self.hidden_layers[0].neurons[n].memory_synapses)
                self.hidden_layers[0].neurons[n].memory_synapses = self.hidden_layers[0].neurons[n].synapses

        return (np.sum(E * E) / 2.0) # общая ошибка