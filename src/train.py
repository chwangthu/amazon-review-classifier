import os
import argparse
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from parser import read_data, get_test_data, get_word_count, get_tfidf
from bagging import bagging
from adaboost import adaboost
from vec import word2vec


def write_res(id, res, filename="../data/result.csv"):
    data_df = pd.DataFrame({'Id': id, 'Predicted': res})
    data_df.to_csv(filename, index=False, sep=',')

def test_single(x_train, y_train, x_test, method, id_test):
    print("Use single "+method)
    if method == "NB":
        clf = MultinomialNB()
    elif method == "SVM":
        lclf = LinearSVC()
        clf = CalibratedClassifierCV(lclf, method='sigmoid', cv=3)
    elif method == "DTREE":
        clf = DecisionTreeClassifier(min_samples_leaf = 5)
    clf.fit(x_train, y_train)
    clf_predict = clf.predict_proba(x_test)[:,1]
    write_res(id_test, clf_predict, "../result/single-"+method+".csv")

def test_bagging(x_train, y_train, x_test, method, id_test):
    print("Use bagging + "+method)
    y_test = bagging(x_train, y_train, x_test, method)
    write_res(id_test, y_test, "../result/bag-"+method+".csv")

def test_ada(x_train, y_train, x_test, method, id_test):
    print("Use adaboost------")
    y_test = adaboost(x_train, y_train, x_test, method) 
    write_res(id_test, y_test, "../result/ada-"+method+".csv")

if __name__ == "__main__":
    if not os.path.exists("../result/"):
        os.makedirs("../result/")
    if not os.path.exists("../data/"):
        os.makedirs("../data/")
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--size', default=57039, help='train size')
    parser.add_argument('--method', default='DTREE', help='train method')
    args = parser.parse_args()

    TRAIN_SIZE = int(args.size)
    x_train, y_train, overall_train, revid_train, asin_train = read_data()
    x_test, revid_test, overall_test, id_test, asin_test = get_test_data()
    
    # for count
    # x_cnt_train, x_cnt_test = get_word_count(x_train, x_test)

    # for tfidf
    x_cnt_train, x_cnt_test = get_tfidf(x_train, x_test)
    
    # for word2vec
    # x_cnt_train, x_cnt_test = word2vec(x_train,x_test)
    # x_cnt_train = csr_matrix(x_cnt_train)
    # x_cnt_test = csr_matrix(x_cnt_test)
    # print(x_cnt_train.shape[0], x_cnt_test.shape[0])

    method = args.method.upper()
    if method == "DTREE": # add other features
        l = len(revid_train)
        x_cnt_train = hstack([x_cnt_train, np.mat(revid_train).reshape((l,1))])
        x_cnt_train = hstack([x_cnt_train, np.mat(overall_train).reshape((l,1))])
        x_cnt_train = hstack([x_cnt_train, np.mat(asin_train).reshape((l,1))])
        x_cnt_train = csr_matrix(x_cnt_train)

        l = len(revid_test)
        x_cnt_test = hstack([x_cnt_test, np.mat(revid_test).reshape((l,1))])
        x_cnt_test = hstack([x_cnt_test, np.mat(overall_test).reshape((l,1))])
        x_cnt_test = hstack([x_cnt_test, np.mat(asin_test).reshape((l,1))])
        x_cnt_test = csr_matrix(x_cnt_test)
    elif method == "SVM":
        l = len(revid_train)
        x_cnt_train = hstack([x_cnt_train, np.mat(overall_train).reshape((l,1))], format="csr")
        l = len(revid_test)
        x_cnt_test = hstack([x_cnt_test, np.mat(overall_test).reshape((l,1))], format="csr")

    # test_single(x_cnt_train[list(range(TRAIN_SIZE)),:], list(y_train)[:TRAIN_SIZE], x_cnt_test, method, id_test)
    test_bagging(x_cnt_train[list(range(TRAIN_SIZE)),:], list(y_train)[:TRAIN_SIZE], x_cnt_test, method, id_test)

    # test_ada(x_cnt_train[list(range(TRAIN_SIZE)),:], list(y_train)[:TRAIN_SIZE], x_cnt_test, method, id_test)
