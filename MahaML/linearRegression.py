import numpy as np
import matplotlib.pyplot as plt
from helpers import lnrmse ,rmse, mse, r2_score, split_dataset

class LRegressor:
    def __init__(self):
        pass

    def fit(self, X, y):
        self.X_train = np.hstack((np.ones((X.shape[0],1)), X))
        self.y_train = y

        Xty = np.dot(self.X_train.T, self.y_train)
        XtX = np.dot(self.X_train.T, self.X_train) 

        self.w = np.linalg.solve(XtX, Xty)
        self.w_star = self.w[1:]
        self.b_star = self.w[0]

    def predict(self,X):
        X = np.hstack((np.ones((X.shape[0],1)), X))
        y_pred = np.dot(X, self.w)
        return y_pred

    def plot(self,Actual, Predicted):
        plt.scatter(Actual, Predicted , alpha=0.5)
        plt.plot([min(Actual), max(Actual)], [min(Actual), max(Actual)], color='red')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs Predicted')
        plt.show()


if __name__ == '__main__':
    diabeties_testing = True

    if diabeties_testing:
        from sklearn.datasets import load_diabetes
        X, y = load_diabetes(return_X_y=True)

        print('\n')
        print('+'*50)
        print("TESTING 1")
        print('+'*50)

        X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.2)

        linR = LRegressor()
        linR.fit(X_train, y_train)
        y_pred =  linR.predict(X_test)
        linR.plot(y_test, y_pred)

        print('\nMy \nLinear Regression')
        print('-'*50)
        print(f'r2 Score: {r2_score(y_test, y_pred)}')
        print(f'MSE: {mse(y_test, y_pred)}')
        print(f'RMSE: {rmse(y_test, y_pred)}')
        print(f'LNRMSE: {lnrmse(y_test, y_pred)}')

        from sklearn.linear_model import LinearRegression
        print("\nInbuilt Linear Regression")
        print('-'*50)
        blinR = LinearRegression()
        blinR.fit(X_train, y_train)
        y_pred = blinR.predict(X_test)
        print('r2 Score:', r2_score(y_pred, y_test))
        print(f'MSE: {mse(y_test, y_pred)}')
        print(f'RMSE: {rmse(y_test, y_pred)}')
        print(f'LNRMSE: {lnrmse(y_test, y_pred)}')


        print('\n')
        print('+'*50)
        print("TESTING 2")
        print('+'*50)

        X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.2)

        linR2 = LRegressor()
        linR2.fit(X_train, y_train)
        y_pred =  linR2.predict(X_test)
        print('\nMy \nLinear Regression')
        print('-'*50)
        print(f'r2 Score: {r2_score(y_test, y_pred)}')
        print(f'MSE: {mse(y_test, y_pred)}')
        print(f'RMSE: {rmse(y_test, y_pred)}')
        print(f'LNRMSE: {lnrmse(y_test, y_pred)}')

        print("\nInbuilt Linear Regression")
        print('-'*50)
        blinR2 = LinearRegression()
        blinR2.fit(X_train, y_train)
        y_pred2 = blinR2.predict(X_test)
        print('r2 Score:', r2_score(y_pred2, y_test))
        print(f'MSE: {mse(y_test, y_pred)}')
        print(f'RMSE: {rmse(y_test, y_pred)}')
        print(f'LNRMSE: {lnrmse(y_test, y_pred)}')

## runner 
'''
conda activate iisc
cd C:\Maha\dev\GitHub\BasicML\MahaML\
python linearRegression.py

'''
