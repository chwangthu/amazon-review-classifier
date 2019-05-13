import numpy as np
import pandas as pd
from gensim.models.word2vec import Word2Vec
from sklearn.preprocessing import scale

def word_vec(text, size, imdb_w2v):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += imdb_w2v.wv[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec

def word2vec(x_train, x_test, dims=100, x_val=None):
    corpus = []
    for item in x_train:
        corpus.append(item)
    if x_val:
        for item in x_val:
            corpus.append(item)
    for item in x_test:
        corpus.append(item)
    model = Word2Vec(min_count=60)
    model.build_vocab(corpus)
    model.train(corpus, total_examples = model.corpus_count, epochs = model.epochs)

    train_vecs = np.concatenate([word_vec(z, dims, model) for z in x_train])
    train_vecs = scale(train_vecs)

    if x_val:
        val_vecs = np.concatenate([word_vec(z, dims, model) for z in x_test])
        val_vecs = scale(val_vecs)

    test_vecs = np.concatenate([word_vec(z, dims, model) for z in x_test])
    test_vecs = scale(test_vecs)
    # print(train_vecs, test_vecs)
    print("Word2vec finished!")
    if x_val:
        return train_vecs, val_vecs, test_vecs
    return train_vecs, test_vecs

