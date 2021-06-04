import numpy as np


def loadGloveModel(File):
    print("Loading Glove Model")
    f = open(File, 'r')
    gloveModel = {}
    for line in f:
        splitLines = line.split()
        word = splitLines[0]
        wordEmbedding = np.array([float(value) for value in splitLines[1:]])
        gloveModel[word] = wordEmbedding
    print(len(gloveModel), " words loaded!")
    return gloveModel


model = loadGloveModel("./../resources/glove.6B/glove.6B.50d.txt")
