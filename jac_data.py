import pandas as pd
import numpy as np
from sklearn import model_selection

def get_data():
    ## Retrieve the tv-audience dataset (10% reduced version -- only 3M rows)
    x = pd.read_csv('https://llperezp-datasets.s3.us-east-2.amazonaws.com/tv-audience-dataset-10pct.csv', header=None)
    #x = pd.read_csv('tv-audience-dataset-10pct.csv', header=None)

    ## get only the subgenre and userid columns
    x = x.iloc[:, 4:6]
    x.columns = ['subgenre', 'userid']

    ## build dummy coding and aggregate into binary arrays for each userid
    x_dumm =  (pd.get_dummies(x, columns=['subgenre']).groupby('userid').sum() > 0) * 1
    X = x_dumm.to_numpy()

    ## remove columns without enough samples
    X = X[:,X.sum(axis=0) > 10]
    return X

## train/test split
def split_dataset(X, size=1500, noise_rate=0.35):

    while True:
        X_train, X_test = model_selection.train_test_split(X, test_size=size)
        
        if (X_train.sum(axis=0) == 0).sum() == 0 and (X_test.sum(axis=0) == 0).sum() == 0:
            break

    X_train_scaled = (2*X_train - 1) 
    X_test_scaled =  (2*X_test - 1) * (2*(np.random.rand(X_test.shape[0], X_test.shape[1]) > noise_rate) - 1)

    return (X_train, X_test, X_train_scaled, X_test_scaled)


