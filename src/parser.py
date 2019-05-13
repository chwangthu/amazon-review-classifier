import re
import numpy as np
import pandas as pd
from string import punctuation
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from gensim.models.word2vec import Word2Vec

def preprocess(text):
    text = text.lower()
    # remove numbers, punctuations
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans("", "", punctuation))
    tokens = word_tokenize(text)
    tokens = [i for i in tokens if not i in ENGLISH_STOP_WORDS]

    # stem and lemmatization
    lemmatizer=WordNetLemmatizer()
    stemmer= PorterStemmer()
    tokens = [lemmatizer.lemmatize(i) for i in tokens]
    tokens = [stemmer.stem(i) for i in tokens]
    return " ".join(tokens)

def read_data():
    train_df = pd.read_csv("../data/train.csv", sep='\t')
    row, column = train_df.shape
    print("size =", row, column)
    x = []
    y = []
    # x = pd.read_csv("../data/processed.csv", sep='\t')['reviewText']
    x = train_df['reviewText']
    y = train_df['label']
    return x, y, train_df['overall'], train_df['reviewerID'], train_df['asin']

def get_test_data():
    test_df = pd.read_csv("../data/test.csv", sep='\t')
    # x_test = pd.read_csv("../data/test_process.csv", sep="\t")['reviewText']
    x_test = test_df['reviewText']
    return x_test, test_df['reviewerID'], test_df['overall'], test_df['Id'], test_df['asin']

def get_word_count(corpus, test, write_file=False):
    vectorizer = CountVectorizer(stop_words="english")
    X = vectorizer.fit_transform(corpus)
    X_test = vectorizer.transform(test)
    names = vectorizer.get_feature_names()
    row, column = X.shape
    print(X.shape)
    if write_file:
        count_list = np.sum(X, axis=0)
        data_df = pd.DataFrame({'word': names, 'count': count_list})
        data_df.to_csv('../data/word_cnt.csv', index=False, sep=',')
    return X, X_test

def get_tfidf(corpus, test):
    tfidf = TfidfVectorizer(stop_words="english")
    X = tfidf.fit_transform(corpus)
    X_test = tfidf.transform(test)
    names = tfidf.get_feature_names()
    row, column = X.shape
    print(X.shape)
    return X, X_test

if __name__ == "__main__":
    x, y = read_data(10)
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

