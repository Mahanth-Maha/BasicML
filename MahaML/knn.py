from cProfile import label
import numpy as np
import matplotlib.pyplot as plt

class knnClassifier:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def split_data(self, X, y, valid_size=0.2):
        n = X.shape[0]
        valid_size = int(n * valid_size)
        indices = np.random.permutation(n)
        X = X[indices]
        y = y[indices]
        X_valid = X[:valid_size]
        y_valid = y[:valid_size]
        X_train = X[valid_size:]
        y_train = y[valid_size:]
        return X_train, y_train, X_valid, y_valid
    
    def subset_set_split(self, X, y, sets = 10):
        n = X.shape[0]
        indices = np.random.permutation(n)
        X = X[indices]
        y = y[indices]
        X_sets = np.array_split(X, sets)
        y_sets = np.array_split(y, sets)
        return X_sets, y_sets

    def cross_valid_accuracy(self):
        X_train_set, y_train_set = self.subset_set_split(self.X_train, self.y_train, sets=10)
        sets_acc = []
        for i in range(10):
            X_train = np.concatenate([X_train_set[j] for j in range(10) if j != i])
            y_train = np.concatenate([y_train_set[j] for j in range(10) if j != i])
            X_valid = X_train_set[i]
            y_valid = y_train_set[i]
            self.fit(X_train, y_train)
            y_pred = self.predict(X_valid)
            acc = self.acc_score(y_valid, y_pred)
            sets_acc.append(acc)
        return np.mean(sets_acc)
        
    def predict(self,X):
        y = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            distances = np.linalg.norm(self.X_train - X[i], axis=1)
            first_k = np.argsort(distances)[:self.k]
            nearest_labels = self.y_train[first_k]
            unique, counts = np.unique(nearest_labels, return_counts=True)
            y[i] = unique[np.argmax(counts)]
        return y
    
    def acc_score(self, y_true, y_pred):
        return np.mean(y_true == y_pred)
        
    def get_best_k(self, X, y, k_max = 20, plot_data=False):
        X_train, y_train, X_valid, y_valid = self.split_data(X, y)
        self.fit(X_train, y_train) 
        best_k = 0
        best_acc = 0
        data = {}
        for k in range(1,k_max,2):
            self.k = k
            y_Pred = self.predict(X_valid)
            acc = self.acc_score(y_valid, y_Pred)
            data[k] = acc
            if acc > best_acc:
                best_acc = acc
                best_k = k
        if plot_data:
            self._plot_data(data, called_by='Accuracy vs k')
        return best_k, best_acc , data
    
    def get_best_k_cross_valid(self, X, y, k_max = 20, plot_data=False):
        best_k = 0
        best_acc = 0
        data = {}
        for k in range(1,k_max,2):
            self.k = k
            acc = self.cross_valid_accuracy()
            data[k] = acc
            if acc > best_acc:
                best_acc = acc
                best_k = k
        if plot_data:
            self._plot_data(data, called_by='Cross Validation Accuracy vs k')
        return best_k, best_acc , data

    def get_best_k_n_tests(self, X, y, k_max = 20, n_tests = 5,plot_data_s = False):
        best_k_s = []
        best_acc_s = []
        data_s = []
        for i in range(n_tests):
            best_k, best_acc, data = self.get_best_k(X, y, k_max)
            best_k_s.append(best_k)
            best_acc_s.append(best_acc)
            data_s.append(data)
        if plot_data_s:
            self._plot_data_s(data_s)
        return best_k_s, best_acc_s , data_s

    def _plot_data(self, data, called_by = 'Acc vs k'):
        best_k = max(data, key=data.get)
        x = list(data.keys())
        y = list(data.values())
        plt.scatter(best_k, data[best_k], color='red',label='Best k = '+str(best_k))
        plt.plot(x, y)
        plt.xlabel('k')
        plt.ylabel('Accuracy')
        plt.title(f'KNN : {called_by}')
        plt.xticks(x)
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def _plot_data_s(self, data_s):
        for en,data in enumerate(data_s):
            best_k = max(data, key=data.get)
            x = list(data.keys())
            y = list(data.values())
            plt.scatter(best_k, data[best_k], color='red')
            plt.plot(x, y, label=f'Test {en+1}')
        plt.xlabel('k')
        plt.ylabel('Accuracy')
        plt.title('Multiple splits - KNN\nAccuracy vs k')
        plt.xticks(x)
        plt.legend()
        plt.grid(True)
        plt.show()

    def set_k(self, k):
        self.k = k

if __name__ == '__main__':
    titanic_testing = True
    iris_testing = True
    if titanic_testing :
        plot_figs = True
        knn = knnClassifier()
        folder = './data/titanic/'

        X_train = np.load(folder + 'X_train.npy')
        y_train = np.load(folder + 'y_train.npy')
        X_test = np.load(folder + 'X_test.npy')
        y_test = np.load(folder + 'y_test.npy')
        
        print(f'\nTesting Working of KNN with k = {knn.k} on Titanic Data_set')
        knn.fit(X_train, y_train)
        yPred =  knn.predict(X_test)
        print(f'\tAccuracy on test data = {knn.acc_score(y_test, yPred)}')

        stars = 40

        print('\n\n')
        print('*'*stars +f'\n\tSection : Normal Validation\n'+ '*'*stars)
        best_k, best_acc, data = knn.get_best_k(X_train, y_train, 20, plot_data=plot_figs)
        print(f'Best K = {best_k} with accuracy = {best_acc}')
        knn.set_k(best_k)
        cross_valid_acc = np.array([ knn.cross_valid_accuracy() for i in range(10)])
        print(f'Avg. Cross Validation Accuracy with best k ({best_k = }) : {cross_valid_acc.mean()}')
        print(f'Accuracy on Test data = {knn.acc_score(y_test, knn.predict(X_test))}')

        print('\n\n')
        print('*'*stars +f'\n\tSection : Cross Validation\n'+ '*'*stars)
        cv_best_k, cv_best_acc, cv_data = knn.get_best_k_cross_valid(X_train, y_train, 20 , plot_data=plot_figs)
        print(f'(CV) Best K = {cv_best_k} with accuracy = {cv_best_acc}')
        knn.set_k(cv_best_k)
        print(f'Accuracy on Test data = {knn.acc_score(y_test, knn.predict(X_test))}')

        print('\n\n')
        print('*'*stars +f'\n\tSection : Multiple Tests\n'+ '*'*stars)
        best_k_s, best_acc_s, data_s = knn.get_best_k_n_tests(X_train, y_train, 20, 5, plot_data_s=plot_figs)
        best_k_max_acc = best_k_s[np.argmax(best_acc_s)]
        print(f'Best k with max accuracy = {best_k_max_acc}')
        max_occurred = np.argmax(np.bincount(best_k_s))
        print(f'Most occurred best k = {max_occurred}')

        avg_acc_of_k = {}
        for d in data_s:
            for k,v in d.items():
                if k not in avg_acc_of_k:
                    avg_acc_of_k[k] = []
                avg_acc_of_k[k].append(v)
        avg_acc_of_k = {k:np.mean(v) for k,v in avg_acc_of_k.items()}
        print(f'Average accuracy of each k :')
        for k,v in avg_acc_of_k.items():
            print(f'{k} : {v}')
        best_avg_k = max(avg_acc_of_k, key=avg_acc_of_k.get)
        print(f'Best k with max average accuracy = {best_avg_k}')
        # knn._plot_data(avg_acc_of_k)

        knn.set_k(best_avg_k)
        print(f'Accuracy on Test data = {knn.acc_score(y_test, knn.predict(X_test))}')


    ### TESTING on iris Dataset
    if iris_testing:
        plot_figs = True

        from sklearn.datasets import load_iris
        iris = load_iris()
        X = iris.data
        y = iris.target
        knn = knnClassifier()
        X_train, y_train, X_test, y_test = knn.split_data(X, y)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        print(f'Accuracy on Testing data = {knn.acc_score(y_test, y_pred)}')
        
        stars = 40

        print('\n\n')
        print('*'*stars +f'\n\tSection : Normal Validation\n'+ '*'*stars)
        best_k, best_acc, data = knn.get_best_k(X_train, y_train, 20, plot_data=plot_figs)
        print(f'Best K = {best_k} with accuracy = {best_acc}')
        knn.set_k(best_k)
        cross_valid_acc = np.array([ knn.cross_valid_accuracy() for i in range(10)])
        print(f'Avg. Cross Validation Accuracy with best k ({best_k = }) : {cross_valid_acc.mean()}')
        print(f'Accuracy on Test data = {knn.acc_score(y_test, knn.predict(X_test))}')

        print('\n\n')
        print('*'*stars +f'\n\tSection : Cross Validation\n'+ '*'*stars)
        cv_best_k, cv_best_acc, cv_data = knn.get_best_k_cross_valid(X_train, y_train, 20, plot_data=plot_figs)
        print(f'(CV) Best K = {cv_best_k} with accuracy = {cv_best_acc}')
        knn.set_k(cv_best_k)
        print(f'Accuracy on Test data = {knn.acc_score(y_test, knn.predict(X_test))}')

        print('\n\n')
        print('*'*stars +f'\n\tSection : Multiple Tests\n'+ '*'*stars)
        best_k_s, best_acc_s, data_s = knn.get_best_k_n_tests(X_train, y_train, 20, 5, plot_data_s=plot_figs)
        best_k_max_acc = best_k_s[np.argmax(best_acc_s)]
        print(f'Best k with max accuracy = {best_k_max_acc}')
        max_occurred = np.argmax(np.bincount(best_k_s))
        print(f'Most occurred best k = {max_occurred}')

        avg_acc_of_k = {}
        for d in data_s:
            for k,v in d.items():
                if k not in avg_acc_of_k:
                    avg_acc_of_k[k] = []
                avg_acc_of_k[k].append(v)
        avg_acc_of_k = {k:np.mean(v) for k,v in avg_acc_of_k.items()}
        print(f'Average accuracy of each k :')
        for k,v in avg_acc_of_k.items():
            print(f'{k} : {v}')
        best_avg_k = max(avg_acc_of_k, key=avg_acc_of_k.get)
        print(f'Best k with max average accuracy = {best_avg_k}')
        # knn._plot_data(avg_acc_of_k)

        knn.set_k(best_avg_k)
        print(f'Accuracy on Test data = {knn.acc_score(y_test, knn.predict(X_test))}')

    ### TESTING on IMDB Review Dataset

## runner 
'''
conda activate iisc
cd C:\Maha\dev\GitHub\BasicML\MahaML\
python knn.py

'''
