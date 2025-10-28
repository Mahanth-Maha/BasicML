import numpy as np
import matplotlib.pyplot as plt
from helpers import acc_score, split_dataset

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTree_Classifier:
    def __init__(self, method = 'entropy', depth_threshold=None, min_samples_split=2):
        self.depth_threshold = depth_threshold or 1000
        self.min_samples_split = min_samples_split
        self.root = None
        self.method = method
        
    def _entropy(self, y):
        m = len(y)
        if m == 0:
            return 0
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / m
        log_probs = np.log(probs)
        p_log_p = probs * log_probs
        return -1 * np.sum(p_log_p)
    
    def _gini(self, y):
        m = len(y)
        if m == 0:
            return 0
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / m
        return 1 - np.sum(probs ** 2)
    
    def _best_split(self, X, y):
        m, n = X.shape
        best_info_gain = float('-inf')
        best_idx, best_thr = None, None

        if m >= self.min_samples_split:
            for idx in range(n):
                all_y_vals = np.unique(X[:, idx])
                for yi in all_y_vals:
                    left_mask = X[:, idx] <= yi
                    right_mask = X[:, idx] > yi
                    if left_mask.sum() == 0 or right_mask.sum() == 0:
                        continue
                    if self.method == 'entropy':
                        info_gain_left = self._entropy(y[left_mask])
                        info_gain_right = self._entropy(y[right_mask])
                    elif self.method == 'gini':
                        info_gain_left = self._gini(y[left_mask])
                        info_gain_right = self._gini(y[right_mask])
                    
                    info_gain = -1 * (left_mask.sum() * info_gain_left + right_mask.sum() * info_gain_right) / m

                    if info_gain > best_info_gain:
                        best_info_gain = info_gain
                        best_idx = idx
                        best_thr = yi

        return best_idx, best_thr

    def _grow_tree(self, X, y, depth):
        num_samples_per_class = [np.sum(y == c) for c in np.unique(y)]
        majority_class = np.unique(y)[np.argmax(num_samples_per_class)]
        
        if (depth >= self.depth_threshold or len(np.unique(y)) == 1 or len(y) < self.min_samples_split):
            return Node(value = majority_class)

        idx, thr = self._best_split(X, y)
        
        if idx is None:
            return Node(value = majority_class)

        left_mask = X[:, idx] <= thr
        right_mask =  X[:, idx] > thr
        left = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._grow_tree(X[right_mask], y[right_mask], depth + 1)
        
        return Node(idx, thr, left, right)

    def _predict(self, inputs, node):
        if node.value is not None:
            return node.value
        if inputs[node.feature] <= node.threshold:
            return self._predict(inputs, node.left)
        return self._predict(inputs, node.right)
    
    def fit(self, X, y):
        self.root = self._grow_tree(X, y, depth=0)

    def predict(self, X):
        return np.array([self._predict(inputs, self.root) for inputs in X])

    def acc_score(self, y_true, y_pred):
        return np.mean(y_true == y_pred)

    def plot(self, X_train, y_train, X_test = None, y_test = None):
        plt.figure(figsize=(10, 8))
        plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], color='red' , alpha=0.5, label='Train -ve')
        plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], color='blue' , alpha=0.5 , label='Train +ve')

        if X_test is not None and y_test is not None:
            plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], color='red', marker='x' , alpha=0.9 , label='Test -ve')
            plt.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], color='blue', marker='x', alpha=0.9, label='Test +ve')

        plt.legend()
        plt.show()

class RandomizedDecisionTree_Classifier(DecisionTree_Classifier):
    def __init__(self, method='entropy', depth_threshold=None, min_samples_split=2, k_features=None):
        super().__init__(method=method, depth_threshold=depth_threshold, min_samples_split=min_samples_split)
        self.k_features = k_features 

    def _best_split(self, X, y):
        m, n = X.shape
        best_info_gain = float('-inf')
        best_idx, best_thr = None, None

        if self.k_features is None or self.k_features > n:
            feature_indices = np.arange(n)
        else:
            feature_indices = np.random.choice(n, self.k_features, replace=False)

        if m >= self.min_samples_split:
            for idx in feature_indices:
                thresholds = np.unique(X[:, idx])
                for thr in thresholds:
                    left_mask = X[:, idx] <= thr
                    right_mask = X[:, idx] > thr
                    if left_mask.sum() == 0 or right_mask.sum() == 0:
                        continue
                    if self.method == 'entropy':
                        left_impurity = self._entropy(y[left_mask])
                        right_impurity = self._entropy(y[right_mask])
                    elif self.method == 'gini':
                        left_impurity = self._gini(y[left_mask])
                        right_impurity = self._gini(y[right_mask])
                    else:
                        raise ValueError(f"Unknown method: {self.method}")
                    
                    info_gain = -1 * (left_mask.sum() * left_impurity + right_mask.sum() * right_impurity) / m

                    if info_gain > best_info_gain:
                        best_info_gain = info_gain
                        best_idx = idx
                        best_thr = thr

        return best_idx, best_thr



if __name__ == '__main__':
    n = 100
    d = 2
    ranges = 2
    postives = np.random.randn(n, d) + np.array([ranges, -ranges])
    negatives = np.random.randn(n, d) + np.array([-ranges, ranges])
    
    postives = np.round(postives)
    negatives = np.round(negatives)
    
    postives = np.column_stack((postives, np.ones(n)))
    negatives = np.column_stack((negatives, np.zeros(n)))
    
    print(f'{np.unique(postives) = }')
    print(f'{np.unique(negatives) = }')
    
    data = np.vstack((postives, negatives))
    np.random.shuffle(data)
    X, y = data[:, :-1], data[:, -1]
    X_train, X_test, y_train, y_test = split_dataset(X, y, 0.2)
    
    print('+'*50)
    print("TESTING 1")
    print('+'*50)
    print(f'{X_train.shape = }\n{X_test.shape = }\n{y_train.shape = }\n{y_test.shape = }\n')
    
    from sklearn.tree import DecisionTreeClassifier
    
    dtree_builtin = DecisionTreeClassifier()
    dtree_builtin.fit(X_train, y_train)
    
    y_pred = dtree_builtin.predict(X_test) 
    print(f'Acc : {acc_score(y_pred, y_test):.2%}')
    
    dtree = DecisionTree_Classifier(
        depth_threshold = 5
        )
    dtree.fit(X_train, y_train)
    
    y_pred = dtree.predict(X_test) 
    print(f'\nAcc : {acc_score(y_pred, y_test):.2%}\n\n')
    # dtree.plot(X_train,y_train)
    print('+'*50)
    print("TESTING 2")
    print('+'*50)
    X = np.array([[2, 3], [1, 1], [2, 1], [1, 2], [3, 3], [4, 4]])
    y = np.array([0, 0, 0, 1, 1, 1])
    dtree_builtin = DecisionTreeClassifier()
    dtree_builtin.fit(X, y)
    
    y_pred = dtree_builtin.predict(X)
    print(f'Acc : {acc_score(y_pred, y):.2%}')
    
    dtree = DecisionTree_Classifier(depth_threshold=3)
    dtree.fit(X, y)
    y_pred = dtree.predict(X) 
    print(f'\n ENTROPY : Acc : {acc_score(y_pred, y):.2%}')

    dtree = DecisionTree_Classifier(method = 'gini',depth_threshold=3)
    dtree.fit(X, y)
    y_pred = dtree.predict(X) 
    print(f'\n GINI : Acc : {acc_score(y_pred, y):.2%}')

    # dtree.plot(X,y)
