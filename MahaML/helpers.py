
import numpy as np


def split_dataset(X,y, test_size = 0.2):
    '''
    Split the dataset into training and testing set.
    '''
    n = X.shape[0]
    n_test = int(n*test_size)
    n_train = n - n_test
    mask = np.random.permutation(n)
    X = X[mask]
    y = y[mask]
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:]
    y_test = y[n_train:]
    return X_train, X_test, y_train, y_test

def acc_score(y_true, y_pred):
    '''
    Calculate the accuracy score.
    '''
    return np.mean(y_true == y_pred)


def mse(y_true, y_pred):
    '''
    Calculate the mean squared error.
    '''
    return np.sum((y_true - y_pred)**2)/len(y_true)

def mae(y_true, y_pred):
    '''
    Calculate the mean absolute error.
    '''
    return np.sum(np.abs(y_true - y_pred))/len(y_true)

def r2_score(y_true, y_pred):
    '''
    Calculate the R2 score.
    '''
    return 1 - np.sum((y_true - y_pred)**2)/np.sum((y_true - np.mean(y_true))**2)

def rmse(y_true, y_pred):
    '''
    Calculate the root mean squared error.
    '''
    return np.sqrt(np.sum((y_true - y_pred)**2)/len(y_true))

def lnrmse(y_true, y_pred):
    '''
    Calculate the log root mean squared error.
    '''
    return np.sqrt(np.sum((np.log(y_true+1) - np.log(y_pred+1))**2)/len(y_true))

def mape(y_true, y_pred):
    '''
    Calculate the mean absolute percentage error.
    '''
    return np.sum(np.abs((y_true - y_pred)/y_true))/len(y_true)
