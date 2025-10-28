import numpy as np
import matplotlib.pyplot as plt

from helpers import acc_score, split_dataset

class DecisionTree_Classifier:
    def __init__(self, feature_selector = 'entropy', depth_threshold = 10, min_samples_split = 2):
        self.feature_selectors = ['entropy', 'gini']
        if feature_selector in self.feature_selectors:
            self.feature_selector = feature_selector
        else: 
            self.feature_selector = 'entropy'
        self.depth_threshold = depth_threshold
        self.min_samples_split = min_samples_split
        self.max_depth = 0
        self.tree = None
    
    def entropy_num(self, uni_c):
        n = len(uni_c)
        probs = uni_c/np.sum(uni_c)
        log_probs = np.log(probs)
        p_log_p = probs * log_probs
        # print(f'{probs.shape = } {log_probs.shape = } {p_log_p.shape = }')
        # H = -1 * np.sum(p_log_p)
        info_gains = np.zeros(n)
        # print(f'{info_gains.shape = }')
        for i in range(1, n-1):
            info_gains[i] = ( i * np.sum(p_log_p[:i]) + (n - i) * np.sum(p_log_p[i:])) / n
            # print(f'{i = }, {info_gains[i] = }')
        info_gains = -1 * np.array(info_gains)
        max_gain = np.argmax(info_gains)
        # print(f'{max_gain = }, {info_gains[max_gain] = }')
        return max_gain, info_gains[max_gain]
    
    def _find_the_best_feature(self, X, y):
        n, feats = X.shape
        max_feat_gains = (float("-inf"), -1, -1)
        for f in range(feats):
            n_uni, n_uni_counts = np.unique(X[:,f], return_counts=True)
            # print(f'{f = } : {n_uni_counts.shape = } {n_uni_counts}')
            if self.feature_selector == 'entropy':
                idx, gain = self.entropy_num(n_uni_counts)
                # print(f'[>] {f = } : {gain = }')
                if gain > max_feat_gains[0]:
                    max_feat_gains = (gain, f, idx)
        # print(f'[>] {max_feat_gains = }')
        k = n_uni[max_feat_gains[2]]
        return (*max_feat_gains, k)
        
    def _recursive_fit(self, X, y ,depth):
        # no more features to split on
        if X.shape[0] == 0 or X.shape[1] == 0 :
           return None
        # pure node
        if len(np.unique(y)) == 1:
            return y[0]
        if self.min_samples_split > X.shape[0] or depth >= self.depth_threshold:
            vals = np.unique(y, return_counts=True)
            # print(f'[>] {vals = } , {y.shape = } , {depth = }, {X.shape = }')
            return vals[0][np.argmax(vals[0])]
        
        gain, feat, idx, thresh = self._find_the_best_feature(X,y)
        # print(f'[>] {gain = } {feat = } {idx = } {thresh = }')
        
        if gain == 0:
            vals = np.unique(y, return_counts=True)
            # print(f'[>] {vals = }')
            return vals[0][np.argmax(vals[0])]
        
        left_mask = X[:, feat] <= thresh
        right_mask = X[:, feat] > thresh
        left = self._recursive_fit(X[left_mask], y[left_mask], depth + 1)
        right = self._recursive_fit(X[right_mask], y[right_mask], depth + 1)

        return (feat, left, right, thresh)
    
    def fit(self, X, y):
        self.tree = self._recursive_fit(X, y , 0)
        return self.tree
    
    def _predict(self, x, tree = None):
        if tree is None:
            return None
        if not isinstance(tree, tuple):
            return int(tree)
        feat, left, right, thresh = tree
        if x[feat] <= thresh:
            # print(f'L' ,end='')
            return self._predict(x, left)
        # print(f'R', end ='')
        return self._predict(x, right)
        
    def predict(self, X):
        y_preds = []
        for x in X:
            # print()
            y_preds.append(self._predict(x, self.tree))
        return np.array(y_preds)

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
    
    print(f'{X_train.shape = }\n{X_test.shape = }\n{y_train.shape = }\n{y_test.shape = }\n')
    print('+'*50)
    print("TESTING 1")
    print('+'*50)
    
    from sklearn.tree import DecisionTreeClassifier
    
    dtree_builtin = DecisionTreeClassifier()
    dtree_builtin.fit(X_train, y_train)
    
    y_pred = dtree_builtin.predict(X_test) 
    print(f'Acc : {acc_score(y_pred, y_test):.2%}')
    
    dtree = DecisionTree_Classifier(
        depth_threshold = 5
        )
    dtree.fit(X_train, y_train)
    print(f"Tree : {dtree.tree}")
    # dtree.plot(X_train,y_train,X_test,y_test)
    
    y_pred = dtree.predict(X_test) 
    print(f'\nAcc : {acc_score(y_pred, y_test):.2%}')
    
    
### outPut : 
'''
np.unique(postives) = array([-5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.,  5.])
np.unique(negatives) = array([-3., -2., -1.,  0.,  1.,  2.,  3.,  4.,  5.])
X_train.shape = (160, 2)
X_test.shape = (40, 2)
y_train.shape = (160,)
y_test.shape = (40,)

++++++++++++++++++++++++++++++++++++++++++++++++++
TESTING 1
++++++++++++++++++++++++++++++++++++++++++++++++++
Acc : 0.975
(1, (0, (0, 0.0, (1, 1.0, 0.0, 0.0), -2.0), 1.0, -0.0), 0.0, 1.0)

Acc : 100.000000%
'''