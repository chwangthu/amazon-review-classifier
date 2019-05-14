import numpy as np
from math import log
from random import randrange
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

def adaboost(x_train, y_train, x_test, classifier="DTREE", T=11):
    # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
    N = x_train.shape[0]
    print("Train size: ", N)
    print("Number of iterations: ", T)
    # y_val_pred = np.zeros(x_val.shape[0])
    y_pred = np.zeros(x_test.shape[0])
    beta_list = []
    clf_list = []
    weight = np.ones(N)*1/N
    for t in range(T):
        print("\r Round: %d/%d" %(t+1, T), end="")
        if classifier == "NB":
            clf = MultinomialNB()
        elif classifier == "SVM":
            lclf = LinearSVC()
            clf = CalibratedClassifierCV(lclf, method='sigmoid', cv=3)
        elif classifier == "DTREE":
            clf = DecisionTreeClassifier(min_samples_leaf = 5)
        clf.fit(x_train, y_train, sample_weight=weight)
        clf_predict = clf.predict(x_train)
        # computer weight of errors
        error_weight = 0
        for i in range(len(clf_predict)):
            if clf_predict[i] != y_train[i]:
                error_weight += weight[i]
        if error_weight > 0.5:
            print("Too many errors, abort loop!")
            error_weight = 0.48
            # continue
        # avoid error_weight to be 0
        error_weight += 0.000001
        beta = error_weight / (1-error_weight)
        # update weight for correct example
        for i in range(len(weight)):
            if(clf_predict[i] == y_train[i]):
                weight[i] = weight[i] * beta
        weight = weight / np.sum(weight)
        # predict
        beta = log(1/beta)
        beta_list.append(beta)
        clf_list.append(clf)
 
    
    # print("Train finished, predict the result...")
    beta_list = np.array(beta_list)/np.sum(beta_list)
    for i in range(len(beta_list)):
        y_pred += np.array(clf_list[i].predict_proba(x_test))[:,1] * beta_list[i]
        # y_val_pred += np.array(clf_list[i].predict_proba(x_val))[:,1] * beta_list[i] 
    # print("Auc score in val set: ", roc_auc_score(y_val, y_val_pred))
    return y_pred