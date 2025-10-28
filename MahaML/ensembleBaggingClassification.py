import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from helpers import acc_score, split_dataset
from tqdm import tqdm
from decisionTreeClassification import DecisionTree_Classifier, RandomizedDecisionTree_Classifier
from logisticRegressionClassification import LogisticRegression_Classifier

class EnsembleBaggingClassification:
    def __init__(self, base_model = 'dtree', m_models = 10, sample_ratio = 0.5, method = 'entropy', depth_threshold=5, min_samples_split=2, use_lr_agg = False, alpha = 0.01 , max_iter = 1000 , stochastic = False , stochastic_choice = -1, epsilon = 1e-5):
        self.m_models = m_models 
        self.base_model = base_model
        self.use_lr_agg  = use_lr_agg 
        self.sample_ratio = sample_ratio
        if self.base_model == 'dtree':
            self.models = [ DecisionTree_Classifier(method, depth_threshold, min_samples_split) for _ in range(self.m_models) ]
        elif self.base_model == 'rdtree':
            self.models = [ RandomizedDecisionTree_Classifier(method, depth_threshold, min_samples_split) for _ in range(self.m_models) ]
        else: 
            self.models = [ DecisionTree_Classifier(method, depth_threshold, min_samples_split) for _ in range(self.m_models) ]
            
        self.data_sampled = None
        self.aggregator = LogisticRegression_Classifier(alpha, max_iter, stochastic, stochastic_choice, epsilon) if self.use_lr_agg else None
    
    
    def _sample_datapoints(self, X, y, split=0.5):
        n_samples = int(X.shape[0] * split)
        all_indices = np.arange(X.shape[0])
        
        in_bag_indices = np.random.choice(all_indices, size=n_samples, replace=True)
        
        out_of_bag_mask = np.ones(X.shape[0], dtype=bool)
        out_of_bag_mask[in_bag_indices] = False
        out_of_bag_indices = all_indices[out_of_bag_mask]

        self.data_sampled[in_bag_indices] += 1
        
        return X[in_bag_indices], y[in_bag_indices], in_bag_indices, out_of_bag_indices

    

    def fit(self, X, y, store_test_samples=True, report_true_test_error=False, verbose = False):
        self.data_sampled = np.zeros(X.shape[0])
        y_preds = []
        self.model_inbag_indices = []     # track which samples trained each model
        self.model_oob_indices = []       # track which samples were left out (OOB)
        self.model_oob_scores = []        # optional per-model OOB accuracy
        
        y_preds_train = []
        
        for i, model in tqdm(enumerate(self.models) , desc='Training Models',unit=' Model'):
            if verbose:
                print(f"Training model {i+1}")
            
            X_i, y_i, inbag, oob = self._sample_datapoints(X, y, self.sample_ratio)
            
            self.model_inbag_indices.append(inbag)
            self.model_oob_indices.append(oob)

            model.fit(X_i, y_i)
            y_pred_train = model.predict(X_i)
            y_preds_train.append(y_pred_train)
            if verbose:
                acc_train = model.acc_score(y_i, y_pred_train)
                print(f"Model {i+1} Train accuracy: {acc_train:.2%}")
            if report_true_test_error and len(oob) > 0:
                y_pred_oob = model.predict(X[oob])
                oob_acc = np.mean(y_pred_oob == y[oob])
                self.model_oob_scores.append(oob_acc)
                if verbose:
                    print(f"Model {i+1} Test accuracy: {oob_acc:.2%}")
            y_preds.append(y_pred)
        
        if self.use_lr_agg:
            X_new = np.array(y_preds).T
            y_new = y 
            if verbose:
                print(f'Training Aggregator...')
                # print(f'{X_new.shape = }, {y_new.shape = }')
            self.aggregator.fit(X_new, y_new)
        
        if (report_true_test_error or verbose) and len(self.model_oob_scores) > 0:
            mean_oob_acc = np.mean(self.model_oob_scores)
            print(f'\n[OOB Evaluation] Mean out-of-bag accuracy: {mean_oob_acc:.3f}')
            print(f'[OOB Evaluation] Estimated generalization error: {1 - mean_oob_acc:.3f}')

        # Optionally store test samples for later analysis
        if not store_test_samples:
            self.model_oob_indices = None
        
    def predict(self, X):
        y_preds = []
        for i, model in tqdm(enumerate(self.models) , desc='Predicting',unit=' Model',):
            y_pred_i = model.predict(X)
            y_preds.append(np.array(y_pred_i))
            # print(f'{y_pred_i.shape = }')
        y_preds = np.array(y_preds)
        
        if self.use_lr_agg:
            y_pred = self.aggregator.predict(y_preds.T)
        else:
            sums = np.sum(y_preds,axis=0)
            y_pred = np.where( sums > (self.m_models//2) ,1,0)
        
        return y_pred
                        
    def report_oob_error(self, X, y):
        if self.model_oob_indices is None:
            print("[ERR] No OOB data stored. Set store_test_samples=True in fit() and retrain !")
            return None
        
        oob_accuracies = []
        for i, model in enumerate(self.models):
            oob_idx = self.model_oob_indices[i]
            if len(oob_idx) == 0:
                continue
            y_pred = model.predict(X[oob_idx])
            acc = np.mean(y_pred == y[oob_idx])
            oob_accuracies.append(acc)

        if len(oob_accuracies) == 0:
            print("[WARN] No OOB samples found for any model.")
            return None

        mean_oob_acc = np.mean(oob_accuracies)
        print(f"[OOB Evaluation] Mean out-of-bag accuracy: {mean_oob_acc:.3f}")
        print(f"[OOB Evaluation] Estimated generalization error: {1 - mean_oob_acc:.3f}")
        return mean_oob_acc


    def acc_score(self, y_true, y_pred):
        return np.mean(y_true == y_pred)

    def plot(self):
        pass

class RandomForestDTree_Classifier(EnsembleBaggingClassification):
    def __init__(self, base_model = 'dtree', m_models = 10, sample_ratio = 0.5, method = 'entropy', depth_threshold=5, min_samples_split=2, use_lr_agg = False, alpha = 0.01 , max_iter = 1000 , stochastic = False , stochastic_choice = -1, epsilon = 1e-5):
        super().__init__(base_model=base_model, m_models=m_models, sample_ratio=sample_ratio, method=method, depth_threshold=depth_threshold, min_samples_split=min_samples_split, use_lr_agg=use_lr_agg, alpha=alpha, max_iter=max_iter, stochastic=stochastic, stochastic_choice=stochastic_choice, epsilon=epsilon)

class RandomForest_Classifier(EnsembleBaggingClassification):
    def __init__(self, base_model = 'rdtree',  m_models=10, sample_ratio=0.5, method='gini', depth_threshold=5, min_samples_split=2, use_lr_agg=False, alpha=0.01, max_iter=1000, stochastic=False, stochastic_choice=-1, epsilon=1e-5,k_features=None):
        self.m_models = m_models
        self.sample_ratio = sample_ratio
        self.use_lr_agg = use_lr_agg
        self.base_model = base_model # fixed
        self.data_sampled = None
        self.k_features = k_features
        self.models = [
            RandomizedDecisionTree_Classifier(
                method=method,
                depth_threshold=depth_threshold,
                min_samples_split=min_samples_split,
                k_features=k_features
            ) 
            for _ in range(m_models)
        ]

        self.aggregator = LogisticRegression_Classifier(
            alpha, max_iter, stochastic, stochastic_choice, epsilon
        ) if use_lr_agg else None

    def fit(self, X, y):
        if self.k_features is None:
            k_sqrt_d = int(np.sqrt(X.shape[1]))
            for model in self.models:
                model.k_features = k_sqrt_d
        super().fit(X, y)

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
    X, y = make_classification(n_samples=5000, n_features=18)
    BANNER_LEN = 80
    M_MODELS = 100
    EACH_TREE_DEPTH = None
    
    test_1_start = datetime.datetime.now()
    print( '\n' + '+'* BANNER_LEN + '\n' + f'\t\t\tTESTING 1\n' + '+'* BANNER_LEN + '\n\n')
    
    X_train, X_test, y_train, y_test = split_dataset(X, y, 0.2)
    print(f'{X_train.shape = }\n{X_test.shape = }\n{y_train.shape = }\n{y_test.shape = }\n')
    print(f'Random Forest Settings (hyper):')
    print(f'{M_MODELS = }')
    print(f'{EACH_TREE_DEPTH = }')
    
    report_columns = [
        'ModelType',
        'SubModelType',
        'Aggregator',
        'Accuracy',
        'N_Models',
        'EachModelDepth',
        'time-2-fit',
        'time-2-pred',
        'time-2-overall'
    ]

    report_df = pd.DataFrame(columns=report_columns)
    
    from sklearn.ensemble import RandomForestClassifier
    
    print( '\n' + '='* BANNER_LEN + '\n' + f'\t\tAlgo: Random Forest by sklearn\n' + '='* BANNER_LEN + '\n')
    tst = datetime.datetime.now()
    RFtree_builtin = RandomForestClassifier(
        n_estimators=M_MODELS,
        max_depth=EACH_TREE_DEPTH
    )
    RFtree_builtin.fit(X_train, y_train)
    fst = datetime.datetime.now()
    print(f'[TIME] Fit {fst - tst }')
    y_pred = RFtree_builtin.predict(X_test) 
    pst = datetime.datetime.now()
    print(f'[TIME] Pred {pst - fst }')
    RFtree_builtin_acc = acc_score(y_pred, y_test)
    print(f'[Accuracy] SKlearn - RForest\'s Acc : {RFtree_builtin_acc:.2}')
    ots = datetime.datetime.now()
    print(f'[TIME] Overall ⌚  --> {ots - tst }')
    new_entry = {'ModelType': 'SkLearn - RForest', 'Accuracy': RFtree_builtin_acc,
        'SubModelType': 'rdtree','Aggregator': 'mean',
        'EachModelDepth': EACH_TREE_DEPTH or 'inf', 'N_Models': M_MODELS,'time-2-fit': (fst - tst).seconds,'time-2-pred': (pst - fst).seconds,'time-2-overall': (ots - tst).seconds,}
    report_df = pd.concat([report_df, pd.DataFrame([new_entry])], ignore_index=True)
    
    
    
    
    print( '\n' + '='* BANNER_LEN + '\n' + f'\t\tAlgo: Pure Decision Tree (only 1 Tree)\n' + '='* BANNER_LEN + '\n')
    tst = datetime.datetime.now()
    depth_dtree = 18
    dtree = DecisionTree_Classifier(
        depth_threshold = depth_dtree
        )
    dtree.fit(X_train, y_train)
    fst = datetime.datetime.now()
    print(f'[TIME] Fit {fst - tst }')
    y_pred = dtree.predict(X_test)
    pst = datetime.datetime.now()
    print(f'[TIME] Pred {pst - fst }')
    dtree_acc = acc_score(y_pred, y_test)
    print(f'[Accuracy] 1 DTree (Depth = {depth_dtree}) Acc : {dtree_acc:.2}')
    ots = datetime.datetime.now()
    print(f'[TIME] Overall ⌚  --> {ots - tst }')
    # dtree.plot(X_train,y_train)
    new_entry = {'ModelType': 'Pure Decision Tree', 'Accuracy':dtree_acc,
        'SubModelType':'dtree','Aggregator':'',
        'EachModelDepth': depth_dtree, 'N_Models':1,'time-2-fit':(fst - tst).seconds,'time-2-pred':(pst - fst).seconds,'time-2-overall':(ots - tst).seconds,}
    report_df = pd.concat([report_df, pd.DataFrame([new_entry])], ignore_index=True)





    print( '\n' + '='* BANNER_LEN + '\n' + f'\t\tAlgo: Random Decision Tree (only 1 Tree)\n' + '='* BANNER_LEN + '\n')
    tst = datetime.datetime.now()
    depth_rdtree = 18
    rdtree = RandomizedDecisionTree_Classifier(
        depth_threshold = depth_rdtree
        )
    rdtree.fit(X_train, y_train)
    fst = datetime.datetime.now()
    print(f'[TIME] Fit {fst - tst }')
    y_pred = rdtree.predict(X_test)
    pst = datetime.datetime.now()
    print(f'[TIME] Pred {pst - fst }')
    rdtree_acc = acc_score(y_pred, y_test)
    print(f'[Accuracy] 1 Random DTree (Depth = {depth_rdtree}) Acc : {rdtree_acc:.2}')
    ots = datetime.datetime.now()
    print(f'[TIME] Overall ⌚  --> {ots - tst }')
    # rdtree.plot(X_train,y_train)
    new_entry = {'ModelType': 'Random Decision Tree', 'Accuracy': rdtree_acc,
        'SubModelType': 'rdtree','Aggregator': '',
        'EachModelDepth': depth_rdtree, 'N_Models': 1,'time-2-fit': (fst - tst).seconds,'time-2-pred': (pst - fst).seconds,'time-2-overall': (ots - tst).seconds,}
    report_df = pd.concat([report_df, pd.DataFrame([new_entry])], ignore_index=True)
    
    
    
    
    
    
    print( '\n' + '='* BANNER_LEN + '\n' + f'\t\tAlgo: RForest using Pure DTrees\n' + '='* BANNER_LEN + '\n')
    tst = datetime.datetime.now()
    RF_plain_DTree = RandomForestDTree_Classifier(
            base_model = 'dtree',
            m_models = M_MODELS,
            sample_ratio = 0.5,
            method = 'gini',
            # deeper == high var low bias 
            depth_threshold=EACH_TREE_DEPTH,
            use_lr_agg = False,
            min_samples_split=2,
        )
    RF_plain_DTree.fit(X_train, y_train)
    fst = datetime.datetime.now()
    print(f'[TIME] Fit {fst - tst }')
    
    y_pred = RF_plain_DTree.predict(X_test)
    pst = datetime.datetime.now()
    print(f'[TIME] Pred {pst - fst }')
    RF_plain_DTree_acc = acc_score(y_pred, y_test)
    print(f'\n[Accuracy] RForest Acc using Pure DTrees: {RF_plain_DTree_acc:.2%}\n')
    ots = datetime.datetime.now()
    print(f'[TIME] Overall ⌚  --> {ots - tst }')
    new_entry = {'ModelType': 'Random Forest', 'Accuracy': RF_plain_DTree_acc,
        'SubModelType': 'dtree','Aggregator': 'mean',
        'EachModelDepth': EACH_TREE_DEPTH or 'inf', 'N_Models': M_MODELS,'time-2-fit': (fst - tst).seconds,'time-2-pred': (pst - fst).seconds,'time-2-overall': (ots - tst).seconds,}
    report_df = pd.concat([report_df, pd.DataFrame([new_entry])], ignore_index=True)
    
    
    
    
    
    print( '\n' + '='* BANNER_LEN + '\n' + f'\t\tAlgo: RForest using Random DTrees\n' + '='* BANNER_LEN + '\n')
    tst = datetime.datetime.now()
    RF_plain_RDTree = RandomForest_Classifier(
            base_model = 'rdtree',
            m_models = M_MODELS,
            sample_ratio = 0.5,
            method = 'gini',
            # deeper == high var low bias 
            depth_threshold=EACH_TREE_DEPTH,
            use_lr_agg = False,
            min_samples_split=2,
        )
    RF_plain_RDTree.fit(X_train, y_train)
    fst = datetime.datetime.now()
    print(f'[TIME] Fit {fst - tst }')
    
    y_pred = RF_plain_RDTree.predict(X_test)
    pst = datetime.datetime.now()
    print(f'[TIME] Pred {pst - fst }')
    RF_plain_RDTree_acc = acc_score(y_pred, y_test)
    print(f'\n[Accuracy] RForest Acc using Random DTrees: {RF_plain_RDTree_acc:.2%}\n')
    ots = datetime.datetime.now()
    print(f'[TIME] Overall ⌚  --> {ots - tst }')
    new_entry = {'ModelType': 'Random Forest', 'Accuracy': RF_plain_RDTree_acc,
        'SubModelType': 'rdtree','Aggregator': 'mean',
        'EachModelDepth': EACH_TREE_DEPTH or 'inf', 'N_Models': M_MODELS,'time-2-fit': (fst - tst).seconds,'time-2-pred': (pst - fst).seconds,'time-2-overall': (ots - tst).seconds,}
    report_df = pd.concat([report_df, pd.DataFrame([new_entry])], ignore_index=True)
    
    
    
    
    print( '\n' + '='* BANNER_LEN + '\n' + f'\t\tAlgo: Bagg and Logistic Agg (Pure DTree)\n' + '='* BANNER_LEN + '\n')
    tst = datetime.datetime.now()
    Bagging_LogiR = EnsembleBaggingClassification(
            base_model = 'dtree',
            m_models = M_MODELS,
            sample_ratio = 0.5,
            method = 'gini',
            # deeper == high var low bias 
            depth_threshold=EACH_TREE_DEPTH,
            use_lr_agg = True,
            min_samples_split=2,
            alpha = 0.005,
            max_iter = 10000,
            stochastic = True ,
            stochastic_choice = 5000,
            epsilon = 1e-8
        )
    Bagging_LogiR.fit(X_train, y_train)
    fst = datetime.datetime.now()
    print(f'[TIME] Fit {fst - tst }')
    
    y_pred = Bagging_LogiR.predict(X_test)
    pst = datetime.datetime.now()
    print(f'[TIME] Pred {pst - fst }')
    Bagging_LogiR_acc = acc_score(y_pred, y_test)
    print(f'\n[Accuracy] RForest (Pure Dtree and LR Aggregation) Acc : {Bagging_LogiR_acc:.2%}\n')
    ots = datetime.datetime.now()
    print(f'[TIME] Overall ⌚  --> {ots - tst }')
    new_entry = {'ModelType': 'Random Forest', 'Accuracy': Bagging_LogiR_acc,
        'SubModelType': 'dtree','Aggregator': 'Logistic',
        'EachModelDepth': EACH_TREE_DEPTH or 'inf', 'N_Models': M_MODELS,'time-2-fit': (fst - tst).seconds,'time-2-pred': (pst - fst).seconds,'time-2-overall': (ots - tst).seconds,}
    report_df = pd.concat([report_df, pd.DataFrame([new_entry])], ignore_index=True)
    
    
    
    
    
    print( '\n' + '='* BANNER_LEN + '\n' + f'\t\tAlgo: Bagg and Logistic Agg (Random DTree)\n' + '='* BANNER_LEN + '\n')
    tst = datetime.datetime.now()
    Bagging_LogiR = EnsembleBaggingClassification(
            base_model = 'dtree',
            m_models = M_MODELS,
            sample_ratio = 0.5,
            method = 'gini',
            # deeper == high var low bias 
            depth_threshold=EACH_TREE_DEPTH,
            use_lr_agg = True,
            min_samples_split=2,
            alpha = 0.005,
            max_iter = 10000,
            stochastic = True ,
            stochastic_choice = 5000,
            epsilon = 1e-8
        )
    Bagging_LogiR.fit(X_train, y_train)
    fst = datetime.datetime.now()
    print(f'[TIME] Fit {fst - tst }')
    
    y_pred = Bagging_LogiR.predict(X_test)
    pst = datetime.datetime.now()
    print(f'[TIME] Pred {pst - fst }')
    Bagging_LogiR_acc = acc_score(y_pred, y_test)
    print(f'\n[Accuracy] RForest (Random Dtree and LR Aggregation) Acc : {Bagging_LogiR_acc:.2%}\n')
    ots = datetime.datetime.now()
    print(f'[TIME] Overall ⌚  --> {ots - tst }')
    new_entry = {'ModelType': 'Random Forest', 'Accuracy': Bagging_LogiR_acc,
        'SubModelType': 'rdtree','Aggregator': 'Logistic',
        'EachModelDepth': EACH_TREE_DEPTH or 'inf', 'N_Models': M_MODELS,'time-2-fit': (fst - tst).seconds,'time-2-pred': (pst - fst).seconds,'time-2-overall': (ots - tst).seconds,}
    report_df = pd.concat([report_df, pd.DataFrame([new_entry])], ignore_index=True)
    
    print(f'\n\n\tComparison table\n')
    print(f'Builtin (RF) \t\t:{RFtree_builtin_acc:.2%}')
    print(f'plain DTree \t\t:{dtree_acc:.2%}')
    print(f'RF (with Pure DTree)\t:{RF_plain_DTree_acc:.2%}')
    print(f'RF (with Random DTree)\t:{RF_plain_RDTree_acc:.2%}')
    print(f'Bagging (Logistic Reg) \t:{Bagging_LogiR_acc:.2%}')

    print(f'\n\n\nResults:\n')
    print(report_df)
    test_1_end = datetime.datetime.now()
    
    print(f'\n\nScript Time : {test_1_end - test_1_start}')
    
'''
OUTPUT

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                        TESTING 1
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


X_train.shape = (4000, 18)
X_test.shape = (1000, 18)
y_train.shape = (4000,)
y_test.shape = (1000,)

Random Forest Settings (hyper):
M_MODELS = 100
EACH_TREE_DEPTH = None

================================================================================
                Algo: Random Forest by sklearn
================================================================================

[TIME] Fit 0:00:01.050071
[TIME] Pred 0:00:00.013069
[Accuracy] SKlearn - RForest's Acc : 0.89
[TIME] Overall ⌚  --> 0:00:01.063140
C:\Maha\dev\GitHub\BasicML\MahaML\ensembleBaggingClassification.py:232: FutureWarning: The behavior of DataFrame concatenation with empty or all-NA entries is deprecated. In a future version, this will no longer exclude empty or all-NA columns when determining the result dtypes. To retain the old behavior, exclude the relevant entries before the concat operation.
  report_df = pd.concat([report_df, pd.DataFrame([new_entry])], ignore_index=True)

================================================================================
                Algo: Pure Decision Tree (only 1 Tree)
================================================================================

[TIME] Fit 0:00:45.920370
[TIME] Pred 0:00:00.002271
[Accuracy] 1 DTree (Depth = 18) Acc : 0.87
[TIME] Overall ⌚  --> 0:00:45.923174

================================================================================
                Algo: Random Decision Tree (only 1 Tree)
================================================================================

[TIME] Fit 0:00:56.706921
[TIME] Pred 0:00:00.002503
[Accuracy] 1 Random DTree (Depth = 18) Acc : 0.87
[TIME] Overall ⌚  --> 0:00:56.709424

================================================================================
                Algo: RForest using Pure DTrees
================================================================================

Training Models: 100 Model [37:36, 22.56s/ Model]
[TIME] Fit 0:37:36.429087
Predicting: 100 Model [00:00, 224.04 Model/s]
[TIME] Pred 0:00:00.448917

[Accuracy] RForest Acc using Pure DTrees: 89.20%

[TIME] Overall ⌚  --> 0:37:36.878508

================================================================================
                Algo: RForest using Random DTrees
================================================================================

Training Models: 100 Model [08:33,  5.13s/ Model]
[TIME] Fit 0:08:33.471716
Predicting: 100 Model [00:00, 248.11 Model/s]
[TIME] Pred 0:00:00.405367

[Accuracy] RForest Acc using Random DTrees: 88.30%

[TIME] Overall ⌚  --> 0:08:33.877652

================================================================================
                Algo: Bagg and Logistic Agg (Pure DTree)
================================================================================

Training Models: 100 Model [37:40, 22.61s/ Model]
[TIME] Fit 0:39:29.631611
Predicting: 100 Model [00:00, 198.82 Model/s]
[TIME] Pred 0:00:00.512364

[Accuracy] RForest (Pure Dtree and LR Aggregation) Acc : 26.70%

[TIME] Overall ⌚  --> 0:39:30.143975

================================================================================
                Algo: Bagg and Logistic Agg (Random DTree)
================================================================================

Training Models: 100 Model [34:41, 20.81s/ Model]
[TIME] Fit 0:35:31.240852
Predicting: 100 Model [00:00, 475.47 Model/s]
[TIME] Pred 0:00:00.212599

[Accuracy] RForest (Random Dtree and LR Aggregation) Acc : 23.20%

[TIME] Overall ⌚  --> 0:35:31.454022


        Comparison table

Builtin (RF)            :88.50%
plain DTree             :86.70%
RF (with Pure DTree)    :89.20%
RF (with Random DTree)  :88.30%
Bagging (Logistic Reg)  :23.20%



Results:

              ModelType SubModelType Aggregator  Accuracy N_Models EachModelDepth time-2-fit time-2-pred time-2-overall
0     SkLearn - RForest       rdtree       mean     0.885      100            inf          1           0              1
1    Pure Decision Tree        dtree                0.867        1             18         45           0             45
2  Random Decision Tree       rdtree                0.867        1             18         56           0             56
3         Random Forest        dtree       mean     0.892      100            inf       2256           0           2256
4         Random Forest       rdtree       mean     0.883      100            inf        513           0            513
5         Random Forest        dtree   Logistic     0.267      100            inf       2369           0           2370
6         Random Forest       rdtree   Logistic     0.232      100            inf       2131           0           2131


Script Time : 2:02:56.249598
'''