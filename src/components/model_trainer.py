import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass

# 模型导入
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# 评估指标和工具
from sklearn.metrics import r2_score

# 自定义模块导入
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    """
    模型训练配置类 (Model Training Configuration Class)
    定义了训练后模型的保存路径。
    """
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    """
    模型训练器类 (Model Trainer Class)
    负责执行整个模型训练、评估和保存的流程。
    """

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        # 将性能阈值定义为类属性，避免“魔法数字”
        self.PERFORMANCE_THRESHOLD = 0.6

    def initiate_model_trainer(self, train_array, test_array):
        """
        启动模型训练流程 (Initiate Model Training Process)
        """
        try:
            logging.info("Splitting training and test data into features and target")
            # 1. 从训练和测试数组中分离特征(X)和目标变量(y)
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # 2. 定义待评估的模型
            models = {
                "LinearRegression": LinearRegression(),
                "KNeighborsRegressor": KNeighborsRegressor(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "RandomForestRegressor": RandomForestRegressor(),
                "AdaBoostRegressor": AdaBoostRegressor(),
                "GradientBoostingRegressor": GradientBoostingRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoostRegressor": CatBoostRegressor(verbose=False)
            }

            # 3. 定义与模型对应的超参数网格 (键名已完全修正)
            # 更加全面和复杂的超参数网格
            params = {
                "LinearRegression": {},  # 线性回归通常没有需要调优的超参数

                "KNeighborsRegressor": {
                    'n_neighbors': [3, 5, 7, 9, 11],  # 邻居数
                    'weights': ['uniform', 'distance'],  # 权重函数: uniform-等权重, distance-距离越近权重越高
                    'p': [1, 2]  # 距离度量: 1-曼哈顿距离(Manhattan), 2-欧氏距离(Euclidean)
                },

                "DecisionTreeRegressor": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],  # 衡量切分质量的函数
                    'max_depth': [5, 10, 20, 30, None],  # 树的最大深度，None表示不限制
                    'min_samples_split': [2, 5, 10],  # 节点继续切分所需的最少样本数
                    'min_samples_leaf': [1, 2, 4],  # 叶子节点最少需要的样本数
                    'max_features': ['sqrt', 'log2', None]  # 寻找最佳切分时考虑的特征数量
                },

                "RandomForestRegressor": {
                    'n_estimators': [64, 128, 256],  # 森林中树的数量
                    'max_depth': [10, 20, 30, None],  # 树的最大深度
                    'min_samples_split': [2, 5, 10],  # 节点继续切分所需的最少样本数
                    'min_samples_leaf': [1, 2, 4],  # 叶子节点最少需要的样本数
                    'max_features': ['sqrt', 'log2', 1.0],  # 每棵树使用的特征比例/数量
                    'bootstrap': [True, False]  # 是否在构建树时使用有放回的样本抽样
                },

                "AdaBoostRegressor": {
                    # 注意: 在新版scikit-learn中, 'base_estimator' 已更名为 'estimator'
                    'estimator': [DecisionTreeRegressor(max_depth=3), DecisionTreeRegressor(max_depth=5),
                                  DecisionTreeRegressor(max_depth=7)],  # 要迭代的基学习器
                    'n_estimators': [32, 64, 128, 256],  # 基学习器的数量
                    'learning_rate': [0.1, 0.05, 0.01, 0.5, 1.0],  # 学习率，控制每个学习器的贡献
                    'loss': ['linear', 'square', 'exponential']  # 损失函数
                },

                "GradientBoostingRegressor": {
                    'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],  # 损失函数
                    'learning_rate': [0.1, 0.05, 0.01],  # 学习率
                    'n_estimators': [64, 128, 256],  # 提升阶段的数量
                    'subsample': [0.7, 0.8, 0.9],  # 用于拟合每个基学习器的样本比例
                    'max_depth': [3, 5, 8],  # 每个决策树的最大深度
                    'min_samples_leaf': [1, 2, 5],
                    'min_samples_split': [2, 5, 10]
                },

                "XGBRegressor": {
                    'n_estimators': [64, 128, 256],
                    'learning_rate': [0.05, 0.1, 0.2, 0.3],  # 学习率
                    'max_depth': [3, 5, 7, 9],  # 树的最大深度
                    'subsample': [0.7, 0.8],  # 训练样本的子采样比例
                    'colsample_bytree': [0.7, 0.8],  # 构建每棵树时特征的子采样比例
                    'gamma': [0, 0.1, 0.2],  # 节点分裂所需的最小损失降低，用于剪枝
                    'reg_alpha': [0, 0.1, 0.5, 1],  # L1 正则化项
                    'reg_lambda': [1, 1.5, 2]  # L2 正则化项
                },

                "CatBoostRegressor": {
                    'depth': [6, 8, 10],  # 树的深度
                    'learning_rate': [0.03, 0.05, 0.1],  # 学习率
                    'iterations': range(1000),  # 提升迭代次数
                    'l2_leaf_reg': [1, 3, 5],  # L2 正则化系数
                    'border_count': [32, 64, 128]  # 用于数值特征分箱的数量
                }
            }

            logging.info("Starting model evaluation")
            # 4. 调用评估函数，获取模型报告
            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                models=models, param=params
            )

            # 5. 从报告中找出最佳模型
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            # 6. 检查最佳模型性能是否达到阈值
            if best_model_score < self.PERFORMANCE_THRESHOLD:
                raise CustomException("No best model found with sufficient performance.", sys)

            logging.info(f"Best model found: {best_model_name} with R2 Score: {best_model_score}")

            # 7. 保存最佳模型对象
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # 8. 在测试集上进行预测并返回最终分数
            predicted = best_model.predict(X_test)
            r2 = r2_score(y_test, predicted)
            logging.info(f"Final R2 score on the test set for the best model is {r2}")

            return r2

        except Exception as e:
            logging.error(f"Exception occurred in model training: {e}")
            raise CustomException(e, sys)