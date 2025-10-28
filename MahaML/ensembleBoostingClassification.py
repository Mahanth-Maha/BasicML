import numpy as np
import matplotlib.pyplot as plt
from helpers import acc_score, split_dataset
from tqdm import tqdm
from decisionTreeClassification import DecisionTree_Classifier

class EnsembleBoostingClassification:
    def __init__(self, base_model = 'dtree', m_models = 10,learning_rate = 0.1, method = 'entropy', depth_threshold=5, min_samples_split=2):
        self.m_models = m_models 
        self.learning_rate = learning_rate
        self.base_model = base_model
        if self.base_model == 'dtree':
            self.models = [ DecisionTree_Classifier(method, depth_threshold, min_samples_split) for _ in range(self.m_models) ]
        else: 
            self.models = [ DecisionTree_Classifier(method, depth_threshold, min_samples_split) for _ in range(self.m_models) ]
        self.init_val = 0
        
    def sigmoid(self, y):
        return 1 / (1 + np.exp(-y))
    
    def loss_gradient(self, y ,y_preds):
        # return self.sigmoid(y) * (1 - self.sigmoid(y))
        return y - y_preds
    
    def fit(self, X, y):
        y = 2*y - 1
        self.init_val = np.mean(y)
        # print(f'[>] {self.init_val = }')
        if self.init_val == 1 or self.init_val == 0 : 
            log_odds = self.init_val
        else:
            log_odds = np.log(np.abs(self.init_val/(1 - self.init_val)))
        # print(f'[>] {log_odds = }')
        
        self.Fx = np.ones(y.shape[0]) * log_odds 
        # print(f'[>] {self.Fx.shape = }')
        
        X_i, y_i = X,y
        for model in tqdm(self.models, desc='Training Models',unit='Model'):
            # print(f"Training model {i+1}")
            
            y_p = self.sigmoid(self.Fx)
            pseudo_res = self.loss_gradient(y_i, y_p)
            
            model.fit(X, pseudo_res)
            
            self.Fx += self.learning_rate * model.predict(X)

    def predict(self, X):
        y_preds = np.ones(X.shape[0]) * self.init_val
        for model in tqdm(self.models, desc='Predicting',unit='Model'):
            y_pred_i = model.predict(X)
            y_preds+= self.learning_rate * np.array(y_pred_i)
            # print(f'{y_pred_i.shape = }')
        
        y_probs = self.sigmoid(y_preds)
    
        return np.where(y_probs >= 0.5, 1, 0)
                        

    def acc_score(self, y_true, y_pred):
        return np.mean(y_true == y_pred)

    def plot(self):
        pass


class GradientBoosting_Classifier(EnsembleBoostingClassification):
    def __init__(self, base_model = 'dtree', m_models = 10,learning_rate = 0.1, method = 'entropy', depth_threshold=5, min_samples_split=2):
        super().__init__(base_model=base_model, m_models=m_models, learning_rate=learning_rate, method=method, depth_threshold=depth_threshold, min_samples_split=min_samples_split)


def checkerboard_pattern(n_samples=300, n_classes=2, grid_size=5):
    X, y = [], []
    n_points = n_samples // (grid_size ** 2)
    for i in range(grid_size):
        for j in range(grid_size):
            x1 = np.random.uniform(i, i + 1, n_points)
            x2 = np.random.uniform(j, j + 1, n_points)
            X.extend(np.c_[x1, x2])
            y.extend([((i + j) % n_classes)] * n_points)
    return np.array(X), np.array(y)

if __name__ == '__main__':
    # X, y = checkerboard_pattern(700)
    
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=2500, n_features=10)
    
    print('+'*50)
    print("TESTING 1")
    print('+'*50)
    X_train, X_test, y_train, y_test = split_dataset(X, y, 0.2)
    print(f'{X_train.shape = }\n{X_test.shape = }\n{y_train.shape = }\n{y_test.shape = }\n')
    
        
        
    from sklearn.ensemble import GradientBoostingClassifier
    
    GradBoosting_builtin = GradientBoostingClassifier()
    GradBoosting_builtin.fit(X_train, y_train)
    
    y_pred = GradBoosting_builtin.predict(X_test) 
    GradBoosting_builtin_acc = acc_score(y_pred, y_test)
    print(f'SKlearn Acc : {GradBoosting_builtin_acc:.2%}')
    
    
    
    dtree = DecisionTree_Classifier(
        depth_threshold = 18
        )
    dtree.fit(X_train, y_train)
    
    y_pred = dtree.predict(X_test)
    dtree_acc = acc_score(y_pred, y_test)
    print(f'DTree Acc : {dtree_acc:.2%}')
    # dtree.plot(X_train,y_train)
    
    
    
    M_MODELS = 100
    EACH_TREE_DEPTH = 2
    
    GradBoosting_plain = EnsembleBoostingClassification(
            base_model = 'dtree',
            m_models = M_MODELS,
            learning_rate=0.1,
            method = 'gini',
            # shallow (stumps) == low var high bias 
            depth_threshold=EACH_TREE_DEPTH,
            min_samples_split=2,
        )
    GradBoosting_plain.fit(X_train, y_train)
    
    y_pred = GradBoosting_plain.predict(X_test)
    GradBoosting_plain_acc = acc_score(y_pred, y_test)
    print(f'\nGradBoosting Acc : {GradBoosting_plain_acc:.2%}\n\n')
    
    print(f'Comparison table')
    print(f'Builtin (GradBoosting) \t\t:{GradBoosting_builtin_acc:.2%}')
    print(f'plain DTree\t\t\t:{dtree_acc:.2%}')
    print(f'GradBoosting (majority) \t:{GradBoosting_plain_acc:.2%}')

'''
OUTPUT
(500,10) samples ::
Comparison table
Builtin (GradBoosting)          :94.00%
plain DTree                     :94.00%
GradBoosting (majority)         :91.00%

(2500,10) samples ::
Comparison table
Builtin (GradBoosting)          :97.20%
plain DTree                     :97.00%
GradBoosting (majority)         :97.00%

'''