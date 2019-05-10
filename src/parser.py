import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from gensim.models.word2vec import Word2Vec

def read_data(test_size=10000):
    train_df = pd.read_csv("../data/train.csv", sep='\t')
    row, column = train_df.shape
    print("size =", row, column)
    x = []
    y = []
    if test_size > row:
        test_size = row
    for i in range(row):
        x.append(train_df.iloc[i, 2])
        y.append(train_df.iloc[i, 6])
    return x, y

def get_test_data():
    test_df = pd.read_csv("../data/test.csv", sep='\t')
    return test_df['reviewText'], test_df['reviewerID'], test_df['overall'], test_df['Id']

def get_word_count(corpus, test, write_file=False):
    vectorizer = CountVectorizer(stop_words="english", max_features=5000)
    X = vectorizer.fit_transform(corpus).toarray()
    X_test = vectorizer.transform(test).toarray()
    names = vectorizer.get_feature_names()
    row, column = X.shape
    if write_file:
        count_list = np.sum(X, axis=0)
        data_df = pd.DataFrame({'word': names, 'count': count_list})
        data_df.to_csv('../data/word_cnt.csv', index=False, sep=',')
    return X, X_test

if __name__ == "__main__":
    x, y = read_data()
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=33)
    vectorizer = CountVectorizer()
    corpus = [
        'This is the first document.',
        'This document is the second document.',
        'And this is the third one.',
         'Is this the first document?',
    ]
    X = vectorizer.fit_transform(corpus)
    print(X)
    # get_word_count(x, True)

