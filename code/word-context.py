import os
import glob
import json
import math
import numpy as np


def load_corpus(folder):
    corpus = []
    for fname in glob.glob(folder):
        with open(fname) as f:
            corpus.append(" ".join(line.strip() for line in f))
    return corpus


def simple_preprocess(token):
    return token.lower()


def simple_tokenize(document, ignore, delim):
    next_token = ""
    tokens = []
    for c in document:
        if c in delim:
            if len(next_token) > 0:
                tokens.append(
                    simple_preprocess(next_token))
                next_token = ""
        elif c not in ignore:
            next_token += c
    return tokens


def make_numeric_corpus(text_corpus):
    numeric_corpus = text_corpus.copy()
    vocab = []
    vocab_length = 0
    for i, doc in enumerate(numeric_corpus):
        for j, token in enumerate(doc):
            try:
                numeric_corpus[i][j] = vocab.index(token)
            except ValueError:
                numeric_corpus[i][j] = vocab_length
                vocab.append(token)
                vocab_length += 1
    return [numeric_corpus, vocab]


def populate_context_word_matrix(C_numeric, vocab, win_size):
    context_word_mat = [[0 for index in vocab] for index in vocab]
    for tokenized_doc in C_numeric:
        for target_word_pk, target_word_idx in enumerate(tokenized_doc):
            context_pks_before = range(
                max(0, target_word_pk - win_size),
                target_word_pk)
            context_pks_after = range(
                target_word_pk + 1,
                min(len(tokenized_doc), target_word_pk + win_size + 1))
            for context_word_pk in context_pks_before:
                context_word_idx = tokenized_doc[context_word_pk]
                context_word_mat[target_word_idx][context_word_idx] += 1
            for context_word_pk in context_pks_after:
                context_word_idx = tokenized_doc[context_word_pk]
                context_word_mat[target_word_idx][context_word_idx] += 1
    return context_word_mat


def word_vector(word, matrix):
    return matrix[vocab.index(word)]


def co_occurrence_count(word, context):
    return word_context_mat[vocab.index(word)][vocab.index(context)]


def co_occurrence_submatrix(documents, words):
    return [[co_occurrence_count(document, word) for word in words]
            for document in documents]


def norm(x):
    return math.sqrt(sum([a*a for a in x]))


def dot(x, y):
    result = 0
    for idx, a in enumerate(x):
        result += a * y[idx]
    return result


def sim(x, y):
    return dot(x, y)/(norm(x)*norm(y))


def most_similar(word, matrix, n):
    result = []
    smallest_value = 0
    for context in vocab:
        similarity = sim(word_vector(word, matrix),
                         word_vector(context, matrix))
        if similarity > smallest_value:
            if len(result) < n:
                result.append([context, similarity])
            else:
                result = sorted(result, key=lambda l: l[1])
                print(f'{result[0]} smaller than {[context, similarity]}!')
                result[0] = [context, similarity]
                result = sorted(result, key=lambda l: l[1])
                smallest_value = result[0][1]
    return result


def populate_pmi_wc_matrix(wc_matrix):
    pmi_wc_mat = [[0 for index in wc_matrix[0]]
                  for index in wc_matrix[0]]
    word_count = np.sum(wc_matrix)
    word_counts = [np.sum(wv) for wv in wc_matrix]
    for w in range(len(wc_matrix[0])):
        for c in range(len(wc_matrix[0])):
            p_wc = wc_matrix[w][c] / word_count
            p_w = word_counts[w] / word_count
            p_c = word_counts[c] / word_count
            pmi_wc_mat[w][c] = math.log(1 + (p_wc / (p_w * p_c)))
    return pmi_wc_mat


def populate_tf_idf_wc_matrix(wc_matrix):
    pmi_wc_mat = [[0 for index in wc_matrix[0]]
                  for index in wc_matrix[0]]
    word_count = np.sum(wc_matrix)
    word_counts = [np.sum(wv) for wv in wc_matrix]
    for w in range(len(wc_matrix[0])):
        for c in range(len(wc_matrix[0])):
            p_wc = wc_matrix[w][c] / word_count
            p_w = word_counts[w] / word_count
            p_c = word_counts[c] / word_count
            pmi_wc_mat[w][c] = math.log(1 + (p_wc / (p_w * p_c)))
    return pmi_wc_mat


re_eval = True

if re_eval is True:
    corpus = load_corpus("./code/dialogues/*.txt")

    ignore_list = ["'", "\"", ".", ",", ";", "?", ":", "!", "(", ")", "[", "]"]
    delim_list = [" "]

    C_text = [simple_tokenize(d, ignore_list, delim_list) for d in corpus]

    C_numeric, vocab = make_numeric_corpus(C_text)

    word_context_mat = populate_context_word_matrix(C_numeric, vocab, 10)

    pmi_wc_mat = populate_pmi_wc_matrix(word_context_mat)

    model = {'matrix': word_context_mat, 'vocab': vocab,
             'corpus': corpus, 'processed_corpus': C_numeric,
             'pmi_mat': pmi_wc_mat}

    with open('code/save-data/word-context-model.json', 'w') as f:
        json.dump(model, f)
else:
    f = open('code/save-data/word-context-model.json')
    model = json.load(f)
    word_context_mat = model['matrix']
    vocab = model['vocab']
    corpus = model['corpus']
    C_numeric = model['processed_corpus']
    pmi_wc_mat = model['pmi_wc_mat']
    f.close()
