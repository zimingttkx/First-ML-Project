import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import *
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
    """
    模型训练配置类，定义了模型训练的路径和相关参数。
    """
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_arr):
        """
        启动模型训练流程。
        1. 从训练数据中分离特征和目标变量。
        2. 训练多个模型并评估其性能。
        3. 保存最佳模型。
        """
        try:
            logging.info("Splitting training and testing input data")
            # 划分数据集 将数据集的最后一列作为目标变量，其余列作为特征
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            # 创建即将训练的Models
            models = {
                "LinearRegression":LinearRegression(),
                "DecisionTreeRegressor":DecisionTreeRegressor(),
                "RandomForestRegressor":RandomForestRegressor(),
                "GradientBoostingRegressor":GradientBoostingRegressor(),
                "XGBRegressor":XGBRegressor(),
                "KNeighborsRegressor":KNeighborsRegressor(),
                "AdaBoostRegressor":AdaBoostRegressor(),
                "CatBoostRegressor":CatBoostRegressor(verbose=False)

            }

            # 获取model performance - R2 score
            model_report:dict = evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)
            # 获取最佳模型的得分
            best_model_score = max(sorted(model_report.values()))
            # 获取最佳模型名称
            best_model_name = list(model_report.keys())[list(
                model_report.values()).index(best_model_score)]
            # 获取最佳模型
            best_model = models[best_model_name]

            # 设置一个模型分数表现阈值
            if best_model_score < 0.6:
                raise CustomException("No best model found with sufficient performance.", sys)
            logging.info("Best model found with sufficient performance {}".format(best_model_score))

            # 保存最佳模型
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # 查看一下预测情况
            predicted = best_model.predict(X_test)
            r2_squared = r2_score(y_test,predicted)

            return r2_squared

        except Exception as e:
            raise CustomException(e,sys)
