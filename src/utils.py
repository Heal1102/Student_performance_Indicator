import os
import sys
import numpy as np
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


from src.Exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path=os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)
    
    
# from sklearn.model_selection import GridSearchCV

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    model_report = {}

    for model_name, model in models.items():
        try:
            # Get the parameters for the current model, or use an empty dictionary
            param_grid = params.get(model_name, {})

            # Perform Grid Search if parameters are provided
            if param_grid:
                grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=2)
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
            else:
                # Directly fit the model if no parameters are provided
                model.fit(X_train, y_train)
                best_model = model

            # Evaluate the model on the test set
            score = best_model.score(X_test, y_test)
            model_report[model_name] = score

        except Exception as e:
            raise CustomException(f"Error with model '{model_name}': {str(e)}", sys)

    return model_report

        

def load_object(file_path):
    try:
        with open(file_path,"rb")as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)
            

