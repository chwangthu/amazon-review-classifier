import numpy as np
from random import randrange
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

def bootstrap_sampling(x_train, y_train):
    set_size = len(x_train)
    x_bs_train, y_bs_train = [], []
    for _ in range(set_size):
        idx = randrange(set_size)
        x_bs_train.append(x_train[idx])
        y_bs_train.append(y_train[idx])
    return x_bs_train, y_bs_train

def bagging(x_train, y_train, x_test, classifier="NB", rounds=11):
    y_test = np.zeros(len(x_test), dtype=int)
    for i in range(rounds):
        print("\r Round: %d/%d" %(i+1, rounds), end="")
        # create bootstrap sample set
        x_bs_train, y_bs_train = bootstrap_sampling(x_train, y_train)
        if classifier == "NB":
            clf = MultinomialNB()
        elif classifier == "SVM":
            clf = LinearSVC()
        elif classifier == "DTREE":
            clf = DecisionTreeClassifier(min_samples_leaf = 5)
        clf.fit(x_bs_train, y_bs_train)
        clf_predict = clf.predict(x_test)
        y_test = y_test + np.array(clf_predict)
    y_test = y_test.tolist()
    y_test = list(map(lambda x: round(x/rounds, 2), y_test))
    return y_test