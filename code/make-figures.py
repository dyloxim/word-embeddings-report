import svgwrite
import csv
import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

# ====================== #
# | internal functions | #
# ====================== #


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


def new_color_grad(lower, mid, upper, bounds):
    def grad_get_color(val):
        output = [0, 0, 0]
        for i in range(0, 3):
            if val > 0:
                output[i] = mid[i] + (val/bounds) * (upper[i] - mid[i])
            else:
                output[i] = mid[i] + (-val/bounds) * (lower[i] - mid[i])
        return [min(255, max(0, round(255*val))) for val in output]
    return grad_get_color


def rgb_to_hex(arr):
    return '#{:02x}{:02x}{:02x}'.format(*arr)


def make_fig__wiki_vecs():
    wiki_model = loadGloveModel("./../resources/glove.6B/glove.6B.50d.txt")
    # twitter_model = loadGloveModel(
    #     "./../resources/glove.twitter.27B/glove.twitter.27B.25d.txt")
    blue_red_grad = new_color_grad(
        [0, 0, 1], [1, 1, 1], [1, 0, 0], 3)
    green_red_grad = new_color_grad(
        [0, 1, 0], [0.95, 0.95, 0.95], [1, 0, 0], 3)

    height = 24
    width = 12

    fig1 = svgwrite.Drawing('figures/python/wiki_vecs.svg', profile='tiny')
    words = ["red", "orange", "monday", "tuesday", "bicycle",
             "scooter", "boy", "girl", "if", "then", "while"]
    for j, word in enumerate(words):
        for i, elem in enumerate(wiki_model[word]):
            fig1.add(fig1.text(word, insert=(0, (height - 2) + j * height)))
            fig1.add(fig1.rect((90 + i * width, (j * height)), (width, height),
                               fill=rgb_to_hex(blue_red_grad(elem))))
    fig1.save()

    # create csv of values
    frame = pd.DataFrame(np.array([wiki_model[word]
                         for word in words]), index=words)
    Z = linkage(frame, method='complete', metric='seuclidean')
    # Plot with Custom leaves
    fig, ax = plt.subplots()
    dendrogram(Z, leaf_rotation=30, leaf_font_size=12, labels=frame.index)
    # Show the graph
    ax.set_ylabel('nearness', fontsize=12)
    fig.savefig("figures/python/wiki_vecs_dendogram.svg", format='svg')


make_fig__wiki_vecs()
