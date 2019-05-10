import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from parser import read_data, get_test_data, get_word_count

# using single classifier
def single_clf():
    pass

def write_res(id, res, filename="../data/result.csv"):
    data_df = pd.DataFrame({'Id': id, 'Predicted': res})
    data_df.to_csv(filename, index=False, sep=',')

if __name__ == "__main__":
    x_train, y_train = read_data()
    x_test, id_test, overall_test, id_test = get_test_data()
    x_cnt_train, x_cnt_test = get_word_count(x_train, x_test)

    nb_clf = MultinomialNB()
    nb_clf.fit(x_cnt_train, y_train)   # 学习
    nb_clf_predict = nb_clf.predict(x_cnt_test) 
    write_res(id_test, nb_clf_predict, "result2.csv")
