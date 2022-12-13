import numpy as np


def loadGlove(path):
    file = open(path, 'r',  encoding="utf8")
    model = {}
    for l in file:
        line = l.split()
        word = line[0]
        value = np.array([float(val) for val in line[1:]])
        model[word] = value
    return model


glove = loadGlove('glove.6B.50d.txt')

x = glove['queen'] - glove['woman'] + glove['man']

print(x)
