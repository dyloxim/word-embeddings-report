import numpy as np
import svgwrite

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


wiki_model = loadGloveModel("./../resources/glove.6B/glove.6B.50d.txt")
# twitter_model = loadGloveModel(
#     "./../resources/glove.twitter.27B/glove.twitter.27B.25d.txt")


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


blue_red_grad = new_color_grad([0, 0, 1], [1, 1, 1], [1, 0, 0], 3)
green_red_grad = new_color_grad([0, 1, 0], [0.95, 0.95, 0.95], [1, 0, 0], 3)

height = 15
width = 10

fig1 = svgwrite.Drawing('figures/python/wiki_vecs.svg', profile='tiny')

words = ["king", "queen", "boy", "girl", "man", "woman", "purple",
         "orange", "green", "red", "monday", "tuesday", "wednesday", "tomorrow"]

for j, word in enumerate(words):
    for i, elem in enumerate(wiki_model[word]):
        fig1.add(fig1.text(word, insert=(0, height + j * height)))
        fig1.add(fig1.rect((90 + i * width, j * height), (width, height),
                           fill=rgb_to_hex(blue_red_grad(elem))))
fig1.save()
