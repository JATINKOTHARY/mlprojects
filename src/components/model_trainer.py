import os 
import sys 
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts',"model.pkl")

class ModelTrainer():
    def __init__(self):
        self.model_trainer_path = ModelTrainerConfig()

    def initiate_model_training(self,train_arr,test_arr):
        try:
            logging.info("Spliting training and test array")
            X_train,y_train,X_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models ={
                "Linear Regressor" : LinearRegression(),
                "KNN Regressor" : KNeighborsRegressor(),
                "XGboost Regressor" : XGBRegressor(),
                "Random Forest Regressor" : RandomForestRegressor(),
                "Decision Tree Regressor" : DecisionTreeRegressor(),
                "Adaboost Regressor" : AdaBoostRegressor(),
                "Gradient Boost Regressor": GradientBoostingRegressor()
            }

            model_report:dict = evaluate_models(X_train= X_train,y_train = y_train,X_test = X_test, y_test = y_test,models = models)

            ## To get best model score 

            best_model_score = max(sorted(model_report.values()))

            ## To get best model name 
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No Best Model Found",sys)
            
            logging.info(f"Best Training model found for both training and testing datasets")


            save_object(
                file_path= self.model_trainer_path.trained_model_file_path,
                obj= best_model
            )

            predicted = best_model.predict(X_test)
            r2score = r2_score(y_test,predicted)

            return r2score

        except Exception as e:
            raise CustomException(e,sys)
