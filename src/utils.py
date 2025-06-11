import sys
import os
import pandas as pd
import numpy as np
import dill
from sklearn.metrics import r2_score
from src.exception import CustomException


def save_object(file_path,obj):
    """
    保存对象到指定的文件路径。
    :param file_path: 文件路径，包含文件名和扩展名。
    :param obj: 要保存的对象，可以是任何Python对象。
    :return:
    """

    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e,sys)


def evaluate_model(X_train,y_train,X_test,y_test, models):
    """
    评估模型性能。
    :param X_train: 训练集特征。
    :param y_train: 训练集目标变量。
    :param X_test: 测试集特征。
    :param y_test: 测试集目标变量。
    :param models: 要评估的模型字典。
    :return: 包含模型名称和其对应的评估指标的DataFrame。
    """
    try:
        report = {}
        # 遍历每个模型进行训练和评估
        for i in range(len(list(models))):
            # 获取模型
            model = list(models.values())[i]
            # 训练模型
            model.fit(X_train,y_train)
            # 预测测试集
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            # 计算评估指标r2
            train_model_score = r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)
            # 将模型名称和评估指标添加到报告中
            report[list(models.keys())[i]] = test_model_score
            # 打印模型名称和评估指标
            return report

    except Exception as e:
        raise CustomException(e, sys)
