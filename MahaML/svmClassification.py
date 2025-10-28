import numpy as np
import matplotlib.pyplot as plt
from helpers import acc_score, split_dataset
from cvxopt import matrix, solvers


class SupportVectorM_Classifier:
    def __init__(self, kernel='linear', Constraint = 1.0, epsilon = 1e-6):
        self.Constraint = Constraint
        self.kernel = kernel
        self.implemented_kernels = ['linear', 'rbf']
        self.epsilon = epsilon
        if kernel not in self.implemented_kernels:
            self.kernel = self.implemented_kernels[0]
        self.gamma = None

    def _compute_kernel(self, X1, X2):
        if self.kernel == 'linear':
            return np.dot(X1, X2.T)
        elif self.kernel == 'rbf':
            if self.gamma is None:
                std_dev = np.std(self.X_train)
                self.gamma = 1 / (2 * std_dev**2)
            X1_sq = np.sum(X1**2, axis=1)[:, None]
            X2_sq = np.sum(X2**2, axis=1)[None, :]
            sq_dists = X1_sq + X2_sq - 2 * np.dot(X1, X2.T)
            return np.exp(-self.gamma * sq_dists)

    def _svm_dual_qp(self, X, y):
        self.K = self._compute_kernel(X, X)
            
        P = matrix(np.outer(y, y) * self.K) 
        q = matrix(-np.ones(self.n))
            
        # Constraint: 0 <= alpha <= C
        G_lower_bound_std = np.diag(np.ones(self.n) * -1)
        h_lower_bound_std = np.zeros(self.n)
        G_higher_bound_slack = np.identity(self.n)
        h_higher_bound_slack = np.ones(self.n) * self.Constraint
        
        ## G alpha <= h
        
        G = matrix(np.vstack((G_lower_bound_std, G_higher_bound_slack)))
        h = matrix(np.hstack((h_lower_bound_std, h_higher_bound_slack)))
        
        #  Constraint: sum y_i alpha_i = 0
        ## A y = b
        A = matrix(y.astype(float), (1, self.n))
        b = matrix(0.0)
        
        ## cv Opt :: max P - q  ; s.t. G a <= h & A y = b
        sol = solvers.qp(P, q, G, h, A, b)
        alpha = np.ravel(sol['x'])
        return alpha
    
    def fit(self, X, y, C = None):
        y = np.where(y == 0, -1, 1)
        self.X_train = X
        self.y_train = y
        self.n, self.m = X.shape
        self.classes = np.unique(y)
        if C:
            self.Constraint = C
        self.alpha = self._svm_dual_qp(self.X_train, self.y_train)
        # print(f'{self.alpha.shape = }')
        
        sv = self.alpha > self.epsilon
        self.sv_alpha = self.alpha[sv]
        self.sv_X = self.X_train[sv]
        self.sv_y = self.y_train[sv]
        self.sv_K = self._compute_kernel(self.sv_X, self.sv_X)
        
        # print(f'{self.sv_alpha.shape = }')
        # print(f'{self.sv_X.shape = }')
        # print(f'{self.sv_y.shape = }')
        # print(f'{self.sv_K.shape = }')
        
        if self.kernel == 'linear':
            self.w = np.sum((self.sv_alpha * self.sv_y)[:, None] * self.sv_X, axis=0)
        else:
            self.w = None
            
        # print(f'{self.w.shape = }')
        self.b = np.mean([
            self.sv_y[i] - np.sum((self.sv_alpha * self.sv_y) * self.sv_K[:, i])
            for i in range(len(self.sv_alpha))
        ])
        # print(f'{self.b.shape = }')
        
        return self.w, self.b

    def predict(self, X):
        if self.kernel == 'linear':
            decision = np.dot(X, self.w) + self.b
        elif self.kernel == 'rbf':
            new_K = self._compute_kernel(X, self.sv_X)
            decision = np.dot(new_K, self.sv_alpha * self.sv_y) + self.b
        # print(decision , self.sv_alpha)
        return np.where(np.sign(decision) == -1, 0, 1)

    def acc_score(self, y_true, y_pred):
        return np.mean(y_true == y_pred)
    
    def plot(self, X_train, y_train, X_test = None, y_test = None):
        if self.w is None:
            print("[Err] Use linear kernel to see plot")
            return
        plt.figure(figsize=(10, 8))
        plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], color='red' , alpha=0.5, label='Train -ve')
        plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], color='blue' , alpha=0.5 , label='Train +ve')
        plt.scatter(self.sv_X[:, 0], self.sv_X[:, 1], color='green', s=100, alpha=0.5)
        if X_test is not None and y_test is not None:
            plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], color='red', marker='x' , alpha=0.9 , label='Test -ve')
            plt.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], color='blue', marker='x', alpha=0.9, label='Test +ve')

        plt.axline((0, -self.b / self.w[1] - 1 / self.w[1]), slope=-self.w[0] / self.w[1], color='blue', linestyle='dashed', label='$W^TX + b \geq 1 $')
        plt.axline((0, -self.b / self.w[1]), slope=-self.w[0] / self.w[1], color='black',label='$W^TX + b = 0 $')
        plt.axline((0, -self.b / self.w[1] + 1 / self.w[1]), slope=-self.w[0] / self.w[1], color='red', linestyle='dashed', label = '$W^TX + b \leq -1 $')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    iris_testing = True
    if iris_testing:
        print('\n')
        print('='*50)
        print("Data")
        print('='*50)

        n = 100
        d = 2
        ranges = 3
        postives = np.random.randn(n, d) + np.array([ranges, -ranges])
        negatives = np.random.randn(n, d) + np.array([-ranges, ranges])
        postives = np.column_stack((postives, np.ones(n)))
        negatives = np.column_stack((negatives, np.zeros(n)))

        data = np.vstack((postives, negatives))
        np.random.shuffle(data)
        X, y = data[:, :-1], data[:, -1] 
        # train_size = int(2*n*0.8)
        # X_train = X[:train_size]
        # y_train = y[:train_size]
        # X_test = X[train_size:]
        # y_test = y[train_size:]
        X_train, X_test, y_train, y_test = split_dataset(X, y, 0.2)
        
        print(f'{X_train.shape = }\n{X_test.shape = }\n{y_train.shape = }\n{y_test.shape = }\n')
        print('+'*50)
        print("TESTING 1")
        print('+'*50)
        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score

        # Create an SVM classifier
        svm_model = SVC(kernel='linear')

        # Train the model
        svm_model.fit(X_train, y_train)

        # Predict on the test data
        y_pred = svm_model.predict(X_test)
        # print(f'{y_test[:10] = }')
        # print(f'{y_pred[:10] = } {np.sum(y_pred) = } ')

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy:.2f}\n\n')
        # print(f'{svm_model.support_vectors_.shape = }')
        # print(f'{svm_model.coef_.shape = }\n\n')
        print('+'*50)
        print("TESTING 2 : Mine - linear")
        print('+'*50)
        svm_model_mine = SupportVectorM_Classifier()
        svm_model_mine.fit(X_train, y_train)
        y_pred =  svm_model_mine.predict(X_test)
        # print(f'{y_test[:10] = }')
        # print(f'{y_pred[:10] = } {np.sum(y_pred) = } ')

        print('\nMy \tSVM Classifier ')
        print('-'*50)
        print(f'Acc: {svm_model_mine.acc_score(y_test, y_pred)}\n\n')
        
        svm_model_mine.plot( X_train, y_train, X_test, y_test)
        
        print('+'*50)
        print("TESTING 3 : Mine - rbf")
        print('+'*50)
        svm_model_mine = SupportVectorM_Classifier(kernel='rbf')
        svm_model_mine.fit(X_train, y_train)
        y_pred =  svm_model_mine.predict(X_test)
        # print(f'{y_test[:10] = }')
        # print(f'{y_pred[:10] = } {np.sum(y_pred) = } ')

        print('\nMy \tSVM Classifier ')
        print('-'*50)
        print(f'Acc: {svm_model_mine.acc_score(y_test, y_pred)}\n\n')
        
        
## runner 
'''
conda activate iisc
cd C:\Maha\dev\GitHub\BasicML\MahaML\
python SVMClassification.py

'''
