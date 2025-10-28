import numpy as np
import matplotlib.pyplot as plt
from helpers import acc_score, split_dataset
from tqdm import tqdm

from logisticRegressionClassification import LogisticRegression_Classifier


class EnsembleStackingClassification:
    def __init__(self, models, use_lr_agg = False, alpha = 0.01 , max_iter = 1000 , stochastic = False , stochastic_choice = -1, epsilon = 1e-5):
        self.models = models 
        self.use_lr_agg  = use_lr_agg 
        self.aggregator = LogisticRegression_Classifier(alpha, max_iter, stochastic, stochastic_choice, epsilon) if self.use_lr_agg else None

    def fit(self, X, y):
        self.data_sampled = np.zeros(X.shape[0])
        y_preds = []
        for model in tqdm(self.models, desc='Training Models',unit='Model'):
            model.fit(X, y)
            if self.use_lr_agg:
                y_pred = model.predict(X)
                # acc = model.acc_score(y, y_pred)
                # print(f"Model {i+1} accuracy: {acc:.2%}")
                
                y_preds.append(y_pred)
                # print(f'{y_pred.shape = }')
            
        if self.use_lr_agg:
            X_new,y_new = np.array(y_preds).T,y
            print(f'[>] Training Aggregator...')
            # print(f'{X_new.shape = }, {y_new.shape = }')
            self.aggregator.fit(X_new, y_new)
        
    def predict(self, X):
        y_preds = []
        for model in tqdm(self.models ,desc='Predicting',unit='Model'):
            y_pred_i = model.predict(X)
            y_preds.append(np.array(y_pred_i))
            # print(f'{y_pred_i.shape = }')
        y_preds = np.array(y_preds)
        
        if self.use_lr_agg:
            y_pred = self.aggregator.predict(y_preds.T)
        else:
            sums = np.sum(y_preds,axis=0)
            y_pred = np.where( sums > (len(self.models)//2) ,1,0)
        
        return y_pred
                        

    def acc_score(self, y_true, y_pred):
        return np.mean(y_true == y_pred)

    def plot(self):
        pass
    
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
    X, y = make_classification(n_samples=1000, n_features=10)
    
    print('+'*50)
    print("TESTING 1")
    print('+'*50)
    X_train, X_test, y_train, y_test = split_dataset(X, y, 0.2)
    print(f'{X_train.shape = }\n{X_test.shape = }\n{y_train.shape = }\n{y_test.shape = }\n')
    
    
    from decisionTreeClassification import DecisionTree_Classifier
    dtree = DecisionTree_Classifier(depth_threshold = 12)
    dtree.fit(X_train, y_train)

    y_pred = dtree.predict(X_test)
    dtree_acc = acc_score(y_pred, y_test)
    print(f'MahaML (DTree) Acc : {dtree_acc:.2%}')   
    
    from logisticRegressionClassification import LogisticRegression_Classifier
    logReg = LogisticRegression_Classifier()
    logReg.fit(X_train, y_train)

    y_pred = logReg.predict(X_test)
    logReg_acc = acc_score(y_pred, y_test)
    print(f'MahaML (LR) Acc : {logReg_acc:.2%}')   

    from svmClassification import SupportVectorM_Classifier
    svm = SupportVectorM_Classifier()
    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)
    svm_acc = acc_score(y_pred, y_test)
    print(f'MahaML (SVM) Acc : {svm_acc:.2%}')   
 
    from naiveBayesClassification import NaiveBayes_Classifier
    naiveBayes = NaiveBayes_Classifier()
    naiveBayes.fit(X_train, y_train)

    y_pred = naiveBayes.predict(X_test)
    naiveBayes_acc = acc_score(y_pred, y_test)
    print(f'MahaML (naiveBayes) Acc : {naiveBayes_acc:.2%}')   
  
    from knnClassification import KNearestN_Classifier
    knn = KNearestN_Classifier()
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    knn_acc = acc_score(y_pred, y_test)
    print(f'MahaML (KNN) Acc : {knn_acc:.2%}')   
    
    from ensembleBaggingClassification import RandomForest_Classifier
    rforest = RandomForest_Classifier()
    rforest.fit(X_train, y_train)

    y_pred = rforest.predict(X_test)
    rforest_acc = acc_score(y_pred, y_test)
    print(f'MahaML (RForest) Acc : {rforest_acc:.2%}')   
    
    from ensembleBoostingClassification import GradientBoosting_Classifier
    gBoost = GradientBoosting_Classifier()
    gBoost.fit(X_train, y_train)

    y_pred = gBoost.predict(X_test)
    gBoost_acc = acc_score(y_pred, y_test)
    print(f'MahaML (GBoost) Acc : {gBoost_acc:.2%}')   
    
    MODELS = [
        DecisionTree_Classifier(),
        LogisticRegression_Classifier(),
        SupportVectorM_Classifier(),
        NaiveBayes_Classifier(),
        KNearestN_Classifier(),
        RandomForest_Classifier(),
        GradientBoosting_Classifier()
    ]
    
    Stacker_plain = EnsembleStackingClassification(
            models = MODELS,
            use_lr_agg = False,
        )
    Stacker_plain.fit(X_train, y_train)
    
    y_pred = Stacker_plain.predict(X_test)
    Stacker_plain_acc = acc_score(y_pred, y_test)
    print(f'\nRForest Acc : {Stacker_plain_acc:.2%}\n\n')
    
    Stacker_with_LogiR = EnsembleStackingClassification(
            models = MODELS,
            use_lr_agg = True,
            
            alpha = 0.005,
            max_iter = 10000,
            stochastic = True ,
            stochastic_choice = 5000,
            epsilon = 1e-8
        )
    Stacker_with_LogiR.fit(X_train, y_train)
    
    y_pred = Stacker_with_LogiR.predict(X_test)
    Stacker_with_LogiR_acc = acc_score(y_pred, y_test)
    print(f'\nRForest (with LR) Acc : {Stacker_with_LogiR_acc:.2%}\n\n')
    
    print()
    print('-'*50)
    print(f'Comparison table')
    print('-'*50)
    print()
    
    print(f'MahaML (DTree)   \tAcc : {dtree_acc:.2%}')   
    print(f'MahaML (LR)      \tAcc : {logReg_acc:.2%}')   
    print(f'MahaML (SVM)     \tAcc : {svm_acc:.2%}')   
    print(f'MahaML (naiveBayes)\tAcc : {naiveBayes_acc:.2%}')   
    print(f'MahaML (KNN)     \tAcc : {knn_acc:.2%}')   
    print(f'MahaML (RForest) \tAcc : {rforest_acc:.2%}')   
    print(f'MahaML (GBoost)  \tAcc : {gBoost_acc:.2%}')   
    print()
    print(f'MahaML (STACKER)  \tAcc : {Stacker_plain_acc:.2%}')   
    print(f'MahaML (STACKER W/LR)\tAcc : {Stacker_with_LogiR_acc:.2%}')   


'''
OUTPUT :: Samples (1000, 10)

--------------------------------------------------
Comparison table
--------------------------------------------------

MahaML (DTree)          Acc : 95.00%
MahaML (LR)             Acc : 93.00%
MahaML (SVM)            Acc : 90.50%
MahaML (naiveBayes)     Acc : 91.00%
MahaML (KNN)            Acc : 88.00%
MahaML (RForest)        Acc : 96.50%
MahaML (GBoost)         Acc : 96.00%

MahaML (STACKER)        Acc : 94.50%
MahaML (STACKER W LR)  Acc : 95.00%
'''