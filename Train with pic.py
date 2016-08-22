import sys
import random
import time
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, "./NeuralNetwork")
import neuralNetwork as NN

def get_vect_from_pic(pix, num, line): # получаем из набора pix цифру num со строки line
    ret = np.empty(900)
    k   = 0
    for i in range(num * 30, (num + 1) * 30):
        for j in range(line * 30, (line + 1) * 30):
            ret[k] = (pix[i, j][0] / 255.0) - 0.5
            k += 1
    return ret

def train_with_pic(network, name_of_sample):
    # иницализация изображения
    image = Image.open(name_of_sample) # открываем изображение
    pix   = image.load()               # выгружаем значения пикселей

    MAX_SET  = 52   # сколько шрифтов в обучающей выборке(ВСЕГО 49 учебных и 3 рукописных)
    MAX_LINE = 49   # сколько шрифтов обучают выборку
    MAX_ERA  = 1000 # ограничение по количеству эпох обучения
    MAX_ERR  = 0.15 # максимальная ошибка после обучения
    EPS      = 0.1  # малое значение

    in_vec   = []
    norm_vec = []
    for line in range(MAX_LINE): # создаем обучающую выборку
        for num in range(10):
            in_vec.append(get_vect_from_pic(pix, num, line))
            temp = np.zeros(10) + EPS
            temp[num] = 1 - EPS
            norm_vec.append(temp)

    SET = [[in_vec[i], norm_vec[i]] for i in range(len(in_vec))]

    print("Обучающая выборка готова. Начинаем обучение...")
    start_time = time.time()

    err = 0   # ошибка
    era = 0   # эпоха
    eta = 0.1 # скорость обучения

    for k in range(MAX_ERA): # начинаем обучение
        for i in range(len(in_vec)):
            e = network.training(SET[i][0], SET[i][1], eta)
            if e > err: err = e

        print("Эпоха {}: ошибка {}; скорость обучения {}.".format(k, err, eta))

        if err <= MAX_ERR: # критерий останова
            era = k
            break

        err_old = err
        err     = 0

        random.shuffle(SET) # случайно перемешиваем обучающую выборку

    if era == 0: era = MAX_ERA # если не закончили по критерию останова

    print("Обучение закончено за {:.3f} сек. на {} эпохе.\n\n\n".format((time.time() - start_time), era))

    test_with_pic(network, name_of_sample) # произвольное тестирование на выборке

def test_with_pic(network, name_of_sample):
    image = Image.open(name_of_sample) # открываем изображение
    pix   = image.load()               # выгружаем значения пикселей

    num  = 0
    line = 0
    while num >= 0:
        num = int(input("Введите цифру(для выхода введите -1): "))
        if num < 0: break
        line = int(input("Введите строку: "))
        res_vec = network(get_vect_from_pic(pix, num, line))
        print("\nРезультат: {} ({})".format((list(res_vec)).index(np.max(res_vec)), max(res_vec)))
        print("Выход сети: {}\n".format(list(res_vec)))

if __name__ == '__main__':
    # для изображения 30x30 распознавания цифр 0..9
    my_network = NN.NeuralNetwork(900, [30], 10) # кол. входов, кол. скрытых слоев == длине массива И кол. нейронов в слое под индексом i == массив[i], кол. выходов
    
    train_with_pic(my_network, "sample_mini.png")

    NN.NeuralNetwork.save(my_network, "myNNmini.nn")

    #my_network = NN.NeuralNetwork.load("myNN.nn")
    #test_with_pic(my_network, "sample_mini.png")