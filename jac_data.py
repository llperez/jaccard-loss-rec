import pandas as pd
import numpy as np

def get_data():
    ## Retrieve the tv-audience dataset (10% reduced version -- only 3M rows)
    #x = pd.read_csv('https://llperezp-datasets.s3.us-east-2.amazonaws.com/tv-audience-dataset-10pct.csv', header=None)
    x = pd.read_csv('tv-audience-dataset-10pct.csv', header=None)

    ## get only the subgenre and userid columns
    x = x.iloc[:, 4:6]
    x.columns = ['subgenre', 'userid']

    ## build dummy coding and aggregate into binary arrays for each userid
    x_dumm =  (pd.get_dummies(x, columns=['subgenre']).groupby('userid').sum() > 0) * 1
    return x_dumm.to_numpy()

## train/test split
def split_dataset(X, size=2000):
    which = np.random.randint(X.shape[0], size=size)
    X_train = np.delete(X, which, axis=0) * 1.0
    X_test = X[which] * 1.0

    #X_train_scaled = X_train * (np.random.rand(X_train.shape[0], X_train.shape[1]) > 0.95)
    #X_test_scaled = X_test * (np.random.rand(X_test.shape[0], X_test.shape[1]) > 0.95)

    X_train_scaled = (2*X_train - 1)
    X_test_scaled = (2*X_test - 1)
    
    return (X_train, X_test, X_train_scaled, X_test_scaled)


