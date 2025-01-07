import numpy as np
import matplotlib.pyplot as plt
from helpers import split_dataset

class NaiveBayes:
    def __init__(self, alpha=1):
        self.alpha = alpha

    def fit(self, X, y, alpha = 0):
        self.n,self.m = X.shape
        self.prior = np.zeros(len(np.unique(y)))
        self.likelihood = np.zeros((len(np.unique(y)), self.m))
        self.classes = np.unique(y)
        self.alpha = alpha

        for i in range(len(self.classes)):
            self.prior[i] = (np.sum(y==self.classes[i])+self.alpha)/(self.n+self.alpha*len(self.classes))
            for j in range(self.m):
                numm = np.sum(X[y==self.classes[i],j] > 0)
                denn = np.sum(y==self.classes[i])
                self.likelihood[i,j] = (numm+self.alpha)/(denn+self.alpha*len(self.classes))

    def predict(self, X):
        self.pred = np.zeros((X.shape[0], len(self.classes)))
        for i in range(len(self.classes)):
            self.pred[:,i] = np.log(self.prior[i]) + np.sum(np.log(self.likelihood[i,:])*X, axis=1)
        return self.classes[np.argmax(self.pred, axis=1)]

    def evaluate(self, X, y):
        return np.mean(self.predict(X)==y)
    
    def acc_score(self,y_pred, y_test):
        return np.mean(y_pred == y_test)

    def plot(self):
        plt.figure(figsize=(10,5))
        for classes in self.classes:
            plt.bar(range(self.m), self.likelihood[classes], alpha=0.5, label='Class '+str(classes))

        plt.xticks(range(self.m))
        plt.xlabel('Feature')
        plt.ylabel('Probability')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    # from sklearn.datasets import load_iris
    # from sklearn.model_selection import split_dataset
    # iris = load_iris()
    # X = iris.data
    # y = iris.target
    # X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.2)

    # print("My Naive Bayes")
    # nb = NaiveBayes()
    # nb.fit(X_train, y_train , alpha = 1e-9)
    # y_pred = nb.predict(X_test)
    # print('Acc : ', np.mean(y_pred == y_test))
    # # print('Accuracy:', nb.evaluate(X_test, y_test))
    # # nb.plot()
    
    # print('-'*50)
    # print("Inbuilt Naive Bayes ")
    # from sklearn.naive_bayes import GaussianNB
    # clf = GaussianNB()
    # clf.fit(X_train, y_train)
    # print('Acc:', clf.score(X_test, y_test))

    print('\n')
    print('='*50)
    print("Titanic Data")
    print('='*50,'\n')

    folder = './data/titanic/'

    X_train = np.load(folder + 'X_train.npy')
    y_train = np.load(folder + 'y_train.npy')
    X_test = np.load(folder + 'X_test.npy')
    y_test = np.load(folder + 'y_test.npy')
    
    print('\n')
    print('+'*50)
    print("TESTING 1")
    print('+'*50)

    nb2 = NaiveBayes()
    nb2.fit(X_train, y_train, alpha = 1e-9)
    y_pred =  nb2.predict(X_test)
    print('\nMy \tNaive Bayes')
    print('-'*50)
    print(f'Acc :{nb2.acc_score(y_pred, y_test)}')
    
    print("\nInbuilt Naive Bayes ")
    print('-'*50)
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    print('Acc:', clf.score(X_test, y_test))

    print('\n')
    print('+'*50)
    print("TESTING 2")
    print('+'*50)

    combine_X = np.vstack((X_train,X_test))
    combine_y = np.hstack((y_train,y_test))


    X_train, X_test, y_train, y_test = split_dataset(combine_X, combine_y, test_size=0.2)
    nb2 = NaiveBayes()
    nb2.fit(X_train, y_train, alpha = 1)
    y_pred =  nb2.predict(X_test)
    print('\nMy \tNaive Bayes')
    print('-'*50)
    print(f'Acc :{nb2.acc_score(y_pred, y_test)}')
    
    print("\nInbuilt Naive Bayes ")
    print('-'*50)
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    print('Acc:', clf.score(X_test, y_test))

## runner 
'''
conda activate iisc
cd C:\Maha\dev\GitHub\BasicML\MahaML\
python naiveBayes.py

'''

