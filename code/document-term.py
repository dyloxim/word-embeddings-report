import os
import glob
import json


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


def populate_doc_term_matrix(C_numeric, vocab):
    document_term_matrix = [[0 for index in vocab] for d in C_numeric]
    for doc_idx, doc in enumerate(C_numeric):
        for vocab_index in doc:
            document_term_matrix[doc_idx][vocab_index] += 1
    return document_term_matrix


def doc_vector(dialogue_name):
    # the first word of each text file is the name of that dialogue
    doc_index = [vocab[W[0]] for W in C_numeric].index(dialogue_name)
    return doc_term_mat[doc_index]


def word_vector(word):
    word_index = vocab.index(word)
    return [doc_term_mat[i][word_index] for i in range(0, len(C_numeric))]


def co_occurrence_count(dialogue_name, word):
    doc_index = [vocab[W[0]] for W in C_numeric].index(dialogue_name)
    word_index = vocab.index(word)
    return doc_term_mat[doc_index][word_index]


def co_occurrence_submatrix(documents, words):
    return [[co_occurrence_count(document, word) for word in words]
            for document in documents]


re_eval = False

if re_eval is True:
    corpus = load_corpus("./code/dialogues/*.txt")

    ignore_list = ["'", "\"", ".", ",", ";", "?", ":", "!", "(", ")", "[", "]"]
    delim_list = [" "]

    C_text = [simple_tokenize(d, ignore_list, delim_list) for d in corpus]

    C_numeric, vocab = make_numeric_corpus(C_text)

    doc_term_mat = populate_doc_term_matrix(C_numeric, vocab)

    model = {'matrix': doc_term_mat, 'vocab': vocab,
             'corpus': corpus, 'processed_corpus': C_numeric}

    with open('code/save-data/document-term-model.json', 'w') as f:
        json.dump(model, f)
else:
    f = open('code/save-data/document-term-model.json')
    model = json.load(f)
    doc_term_mat = model['matrix']
    vocab = model['vocab']
    corpus = model['corpus']
    C_numeric = model['processed_corpus']
    f.close()

co_occurrence_submatrix(["laws", "republic", "cratylus", "meno", "apology", "crito"],
                        ["virtue", "law", "the", "earth", "god", "socrates", "drink"])
