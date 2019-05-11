import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from parser import read_data, get_test_data, get_word_count
from sklearn.tree import DecisionTreeClassifier
from bagging import bagging
from vec import extract


def write_res(id, res, filename="../data/result.csv"):
    data_df = pd.DataFrame({'Id': id, 'Predicted': res})
    data_df.to_csv(filename, index=False, sep=',')

if __name__ == "__main__":
    x_train, y_train = read_data(10000)
    x_train = list(pd.read_csv("../data/processed.csv", sep="\t")['reviewText'][:10000])
    x_test, id_test, overall_test, id_test = get_test_data()
    x_test = list(pd.read_csv("../data/test_process.csv", sep="\t")['reviewText']) 

    #for count
    x_cnt_train, x_cnt_test = get_word_count(x_train, x_test)
    # x_val = x_cnt_train[8000:]
    # y_val = y_train[8000:]
    # x_cnt_train = x_cnt_train[:8000]
    # y_train = y_train[:8000]

    # # for word2vec
    # x_cnt_train, x_val, x_cnt_test = extract(x_train[:800], x_train[800:], x_train[800:])
    # y_val = y_train[800:]
    # y_train = y_train[:800]
    

    method = "SVM"
    # val = bagging(x_cnt_train, y_train, x_val, method)
    # print("-------")
    # print(len(val), len(y_val))
    # print("Result in validation set:\n", classification_report(val, y_val))

    y_test = bagging(x_cnt_train, y_train, x_cnt_test, method)
    write_res(id_test, y_test, "../result/"+method+".csv")

