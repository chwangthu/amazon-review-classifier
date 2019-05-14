import os
import argparse
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import scale
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from parser import read_data, get_test_data, get_word_count, get_tfidf
from bagging import bagging
from adaboost import adaboost
from vec import word2vec


def write_res(id, res, filename="../data/result.csv"):
    data_df = pd.DataFrame({'Id': id, 'Predicted': res})
    data_df.to_csv(filename, index=False, sep=',')

def test_single(x_train, y_train, x_test, method, id_test):
    # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
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
    # val_pred = clf.predict_proba(x_val)[:,1]
    # print("Auc score in val set: ", roc_auc_score(y_val, val_pred))
    write_res(id_test, clf_predict, "../result/single-"+method+".csv")

def test_bagging(x_train, y_train, x_test, method, id_test, t=13):
    print("Use bagging + "+method)
    y_test = bagging(x_train, y_train, x_test, method, t)
    write_res(id_test, y_test, "../result/bag-"+str(t)+method+".csv")

def test_ada(x_train, y_train, x_test, method, id_test, t=17):
    print("Use adaboost------")
    y_test = adaboost(x_train, y_train, x_test, method, t)
    write_res(id_test, y_test, "../result/ada-"+str(t)+method+".csv")

if __name__ == "__main__":
    if not os.path.exists("../result/"):
        os.makedirs("../result/")
    if not os.path.exists("../data/"):
        os.makedirs("../data/")
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--size', default=57039, help='train size')
    parser.add_argument('--method', default='SVM', help='train method')
    parser.add_argument('--vec', default='TFIDF', help='vector form')
    args = parser.parse_args()

    TRAIN_SIZE = int(args.size)
    x_train, y_train, overall_train, revid_train, asin_train = read_data()
    x_test, revid_test, overall_test, id_test, asin_test = get_test_data()
    
    vec_form = args.vec.upper()
    if vec_form == "BOW":
        # for BOW
        x_cnt_train, x_cnt_test = get_word_count(x_train, x_test)
    elif vec_form == "TFIDF":
        # for tfidf
        x_cnt_train, x_cnt_test = get_tfidf(x_train, x_test)
    elif vec_form == "WORD2VEC":
        # for word2vec
        x_cnt_train, x_cnt_test = word2vec(x_train,x_test)
        x_cnt_train = csr_matrix(x_cnt_train)
        x_cnt_test = csr_matrix(x_cnt_test)

    method = args.method.upper()

    # add overall
    l = len(revid_train)
    x_cnt_train = hstack([x_cnt_train, np.mat(overall_train).reshape((l,1))], format="csr")
    l = len(revid_test)
    x_cnt_test = hstack([x_cnt_test, np.mat(overall_test).reshape((l,1))], format="csr")

    # add asin
    # l = len(revid_train)
    # x_cnt_train = hstack([x_cnt_train, np.mat(scale(asin_train)).reshape((l,1))], format="csr")
    # l = len(revid_test)
    # x_cnt_test = hstack([x_cnt_test, np.mat(scale(asin_test)).reshape((l,1))], format="csr")

    # add review id
    # l = len(revid_train)
    # x_cnt_train = hstack([x_cnt_train, np.mat(scale(revid_train)).reshape((l,1))], format="csr")
    # l = len(revid_test)
    # x_cnt_test = hstack([x_cnt_test, np.mat(scale(revid_test)).reshape((l,1))], format="csr")

    # test_single(x_cnt_train[list(range(TRAIN_SIZE)),:], list(y_train)[:TRAIN_SIZE], x_cnt_test, "DTREE", id_test)

    # test_single(x_cnt_train[list(range(TRAIN_SIZE)),:], list(y_train)[:TRAIN_SIZE], x_cnt_test, "SVM", id_test)

    # test_single(x_cnt_train[list(range(TRAIN_SIZE)),:], list(y_train)[:TRAIN_SIZE], x_cnt_test, "NB", id_test)

    test_bagging(x_cnt_train[list(range(TRAIN_SIZE)),:], list(y_train)[:TRAIN_SIZE], x_cnt_test, method, id_test, 13)

    test_ada(x_cnt_train[list(range(TRAIN_SIZE)),:], list(y_train)[:TRAIN_SIZE], x_cnt_test, method, id_test, 15)

