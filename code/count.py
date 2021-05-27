import os
import glob
import time


def remove_special(string):
    remove = ['.', ',', '?', '!', ':', '(', ')', '[', ']', ';', '*',
              '']
    return ''.join(i for i in string if not i in remove)


def tokenize(d):
    lines = []
    for line in d:
        lines.append(line)
    lines = ''.join(lines)
    tokens = [remove_special(token).lower() for token in lines.split()]
    return [token for token in tokens if token != '']


corpus = []

for fname in glob.glob("./code/dialogues/*.txt"):
    with open(fname) as f:
        corpus.append(tokenize(f))

for d in corpus:
    print(f'{d[0]}: {len(d)}~{len(set(d))}')

vocab = list(set.union(*[set(d) for d in corpus]))

print(f'total: {len(vocab)}')

# corpus_i = []
# for d in corpus:
#     corpus_i.append([vocab.index(w) for w in d])
#     print(f'done {d[0]}')
# time taken: ~830 ish

corpus_i = json.load(open('./code/save/corpus_i.txt'))

# document_term_matrix = [[d.count(w) for d in corpus] for w in vocab]
# time taken ~ 225.65

# dtm_i = [[d.count(n) for d in corpus_i] for n in range(len(vocab))]
dtm_i = json.load(open('./code/save/dtm_i.txt'))


def wv(word):
    wv = ""
    try:
        wv = dtm_i[vocab.index(word)]
    except ValueError:
        wv = None
        print(f"word '{word}' not in vocabulary")
    return wv


def v_len(v):
    norm = 0
    for c in v:
        norm += c*c
    return math.sqrt(norm)
