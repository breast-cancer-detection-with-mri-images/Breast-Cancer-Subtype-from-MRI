from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import catboost
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split


def tune_catboost_hyperparameters(train_x, train_y, val_x, val_y, parameters = None, criterion = 'accuracy'):

    param_grid = {
    'depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'iterations': [100, 200, 300],
    'l2_leaf_reg': [1, 3, 5, 7, 9]
    }

    if parameters is not None:
        param_grid = parameters
    
    split_index = [-1]*len(train_x) + [0]*len(val_x)
    X = np.concatenate((train_x, val_x), axis=0)
    y = np.concatenate((train_y, val_y), axis=0)
    pds = PredefinedSplit(test_fold = split_index)
    model = CatBoostClassifier()


    grid_search = GridSearchCV(model, param_grid = param_grid, cv = pds, scoring = criterion)
    grid_search.fit(X, y)  
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    # print("Best Hyperparameters: ", best_params)
    # print("Best Score: ", best_model.score(val_x, val_y))
    return best_model 
