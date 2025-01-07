from math import e
import numpy as np
import matplotlib.pyplot as plt
from helpers import acc_score, split_dataset

class LRClassifier:
    def __init__(self , alpha = 0.01 , max_iter = 1000 , stochastic = False , stochastic_choice = -1, epsilon = 1e-5):
        self.lr = alpha
        self.max_iter = max_iter
        self.stochastic = stochastic
        self.stochastic_choice = stochastic_choice
        self.epsilon = epsilon

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.n, self.m = X.shape
        self.classes = np.unique(y)
        if self.stochastic :
            if 0 < self.stochastic_choice < self.n:
                self.K = self.stochastic_choice
            else :
                self.K = np.random.randint(1,self.n)
        else :
            self.K = self.n
        self.w = np.random.randn(self.m) + 0.5
        self.b = np.random.randn()
        self.losses = []
        for i in range(self.max_iter):
            w_prev = self.w.copy()
            self.update_weights()
            self.losses.append(self.loss())
            if np.linalg.norm(w_prev - self.w) < self.epsilon:
                print(f'Converged at {i} iteration')
                break
        return self.losses
    
    def update_weights(self):
        dw = np.zeros(self.m)
        db = 0
        for i in range(self.K):
            y_pred = self.sigmoid(np.dot(self.w, self.X_train[i]) + self.b)
            dw += (y_pred - self.y_train[i]) * self.X_train[i]
            db += (y_pred - self.y_train[i])
        if self.lr > 0.001:
            self.lr = self.lr * 0.99
        self.w -= self.lr * dw
        self.b -= self.lr * db

    def predict(self,X):
        y = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            if self.classes.shape[0] > 2:
                y[i] = np.argmax(self.softmax(np.dot(self.w,X[i]) + self.b))
            else:
                y[i] = 1 if self.sigmoid(np.dot(self.w,X[i]) + self.b) > 0.5 else 0
        return y

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0) 

    def loss(self):
        loss = 0
        for i in range(self.n):
            if self.classes.shape[0] > 2:
                y_pred = self.softmax(np.dot(self.w, self.X_train[i]) + self.b)
                loss += -np.log(y_pred[self.y_train[i]])
            else:
                y_pred = self.sigmoid(np.dot(self.w, self.X_train[i]) + self.b)
                loss += self.y_train[i] * np.log(y_pred) + (1 - self.y_train[i]) * np.log(1 - y_pred)
        return -loss/self.n

    def acc_score(self, y_true, y_pred):
        return np.mean(y_true == y_pred)
    
    def plot(self):
        plt.plot(self.losses)
        plt.xlabel('max_iter')
        plt.ylabel('Loss')
        plt.show()


if __name__ == '__main__':
    titanic_testing = True
    iris_testing = False
    if titanic_testing:
        print('\n')
        print('='*50)
        print("Titanic Data")
        print('='*50)

        folder = './data/titanic/'
        X_train = np.load(folder + 'X_train.npy')
        y_train = np.load(folder + 'y_train.npy')
        X_test = np.load(folder + 'X_test.npy')
        y_test = np.load(folder + 'y_test.npy')
        
        print('\n')
        print('+'*50)
        print("TESTING 1")
        print('+'*50)

        lr = LRClassifier(
            alpha=0.05,
            stochastic=True, 
            stochastic_choice = 100
        )
        lr.fit(X_train, y_train)
        y_pred =  lr.predict(X_test)
        lr.plot()
        print('\nMy \tLogistic Regression')
        print('-'*50)
        print(f'Acc: {lr.acc_score(y_test, y_pred)}')

        from sklearn.linear_model import LogisticRegression
        print("\nInbuilt Logistic Regression")
        print('-'*50)
        blr = LogisticRegression()
        blr.fit(X_train, y_train)
        y_pred = blr.predict(X_test)
        print('Acc:', acc_score(y_pred, y_test))


        print('\n')
        print('+'*50)
        print("TESTING 2")
        print('+'*50)

        combine_X = np.vstack((X_train,X_test))
        combine_y = np.hstack((y_train,y_test))
        X_train, X_test, y_train, y_test = split_dataset(combine_X, combine_y, test_size=0.2)

        lr = LRClassifier()
        lr.fit(X_train, y_train)
        y_pred =  lr.predict(X_test)
        print('\nMy \tLogistic Regression')
        print('-'*50)
        print(f'Acc: {acc_score(y_test, y_pred)}')

        print("\nInbuilt Logistic Regression")
        print('-'*50)
        blr = LogisticRegression()
        blr.fit(X_train, y_train)
        y_pred2 = blr.predict(X_test)
        print('Acc:', acc_score(y_pred2, y_test))

    if iris_testing:
        print('\n')
        print('='*50)
        print("Iris Data")
        print('='*50)

        from sklearn.datasets import load_iris
        from sklearn.linear_model import LogisticRegression 
        
        iris = load_iris()
        X = iris.data
        y = iris.target
        X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.2)

        # print('+'*50)
        # print("TESTING 1")
        # print('+'*50)
        lr = LRClassifier()
        lr.fit(X_train, y_train)
        y_pred =  lr.predict(X_test)

        print('\nMy \tLogistic Regression')
        print('-'*50)
        print(f'Acc: {lr.acc_score(y_test, y_pred)}')

        # print("\nInbuilt Logistic Regression")
        # print('-'*50)
        # blr = LogisticRegression()
        # blr.fit(X_train, y_train)
        # print('Acc:', blr.score(X_test, y_test))




## runner 
'''
conda activate iisc
cd C:\Maha\dev\GitHub\BasicML\MahaML\
python logisticRegression.py

'''
