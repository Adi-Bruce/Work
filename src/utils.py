import os
import sys

import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import pickle

#sys.path.insert(0, '/home/brucewayne/Documents/Work_1/src')
from src.logger import logging
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_models(X_train,y_train,X_test,y_test,model,param):
    #models = list(model)
    report = {}
    try:
        for i in range(len(list(model))):
            c_model = list(model.values())[i]
            para = param[list(model.keys())[i]]
            
            gs = GridSearchCV(c_model,para,cv=3)
            gs.fit(X_train,y_train)

            c_model.set_params(**gs.best_params_)
            c_model.fit(X_train,y_train)

            y_train_pred= c_model.predict(X_train)
            y_test_pred = c_model.predict(X_test)
            
            train_model_score = r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)
            report[(list(model.keys()))[i]] = test_model_score
            
        return report
    except Exception as e:
        raise CustomException(e,sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)

        


        
