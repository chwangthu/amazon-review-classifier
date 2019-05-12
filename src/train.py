import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from parser import read_data, get_test_data, get_word_count
from sklearn.tree import DecisionTreeClassifier
from bagging import bagging
from adaboost import adaboost
from vec import word2vec


def write_res(id, res, filename="../data/result.csv"):
    data_df = pd.DataFrame({'Id': id, 'Predicted': res})
    data_df.to_csv(filename, index=False, sep=',')

def test_bagging(x_train, y_train, x_test, method, id_test):
    y_test = bagging(x_train, y_train, x_test, method)
    write_res(id_test, y_test, "../result/bag-"+method+".csv")

def test_ada(x_train, y_train, x_test, method, id_test):
    y_test = adaboost(x_train, y_train, x_test, method) 
    write_res(id_test, y_test, "../result/ada-"+method+".csv")

if __name__ == "__main__":
    TRAIN_SIZE = 50000
    x_train, y_train, overall_train, revid_train, asin_train = read_data(TRAIN_SIZE)
    x_test, revid_test, overall_test, id_test, asin_test = get_test_data()
    
    #for count
    x_cnt_train, x_cnt_test = get_word_count(x_train, x_test)
    method = "DTREE"
    if method == "DTREE":
        # print(len(x_cnt_train[0]), len(x_cnt_test[0]))
        x_train_tep = []
        x_test_tep = []
        for i in range(TRAIN_SIZE):
            # print(revid_train[i])
            m = np.append(x_cnt_train[i], revid_train[i])
            n = np.append(m, overall_train[i])
            p = np.append(n, asin_train[i])
            x_train_tep.append(p)
        for i in range(len(x_cnt_test)):
            m = np.append(x_cnt_test[i], revid_test[i])
            n = np.append(m, overall_test[i])
            p = np.append(n, asin_test[i])
            x_test_tep.append(p)
        x_cnt_train = x_train_tep
        x_cnt_test = x_test_tep
        # print(len(x_cnt_train[0]), len(x_cnt_test[0]))
    test_bagging(x_cnt_train, y_train, x_cnt_test, method, id_test)
    # test_ada(x_cnt_train, y_train, x_cnt_test, method, id_test)
    
    # x_val = x_cnt_train[8000:]
    # y_val = y_train[8000:]
    # x_cnt_train = x_cnt_train[:8000]
    # y_train = y_train[:8000]

    # for word2vec
    # x_cnt_train, x_cnt_test = word2vec(x_train,x_test)
    # print(len(x_cnt_train), len(x_cnt_test))
    # y_val = y_train[800:]
    # y_train = y_train[:800]
    

    # val = bagging(x_cnt_train, y_train, x_val, method)
    # val = list(map(lambda x: 0 if x < int(11/2) else 1, val))
    # print("-------")
    # print(len(val), len(y_val))
    # print("Result in validation set:\n", classification_report(val, y_val))

    # y_test = bagging(x_cnt_train, y_train, x_cnt_test, method)
    # y_test = adaboost(x_cnt_train, y_train, x_cnt_test, method)
    # write_res(id_test, y_test, "../result/"+method+".csv")

