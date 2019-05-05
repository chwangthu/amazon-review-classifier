import pandas as pd

def read_data():
    train_df = pd.read_csv("../data/train.csv", sep='\t')
    row, column = train_df.shape
    print(row, column)
    for i in range(2):
        for j in range(column):
            print(train_df.iloc[i, j]) #iloc uses number, loc uses label, ix can be both

if __name__ == "__main__":
    read_data()