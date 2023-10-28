from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

def get_best_hyperparameters(train_x, train_y, val_x, val_y, parameters = None, criterion = 'accuracy'):
    param_grid = {'n_estimators': [20,40,60,80,100,200], 
                  'criterion': ['gini', 'entropy'],
                  'max_depth': [6,10,15]} 

    if parameters is not None:
        param_grid = parameters

    
    split_index = [-1]*len(train_x) + [0]*len(val_x)
    X = np.concatenate((train_x, val_x), axis=0)
    y = np.concatenate((train_y, val_y), axis=0)
    pds = PredefinedSplit(test_fold = split_index)
    model = RandomForestClassifier()


    grid_search = GridSearchCV(model, param_grid = param_grid, cv = pds, scoring = criterion)
    grid_search.fit(X, y)  
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    # print("Best Hyperparameters: ", best_params)
    # print("Best Score: ", best_model.score(val_x, val_y))
    return best_model 