import sys
import os
from dataclasses import dataclass
import pandas as pd
import numpy as np
from jupyter_server.auth import passwd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from src.utils import save_object
from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    """
    数据转换配置类，定义了数据转换的路径和相关参数。
    """
    preprocessed_object_file_path: str = os.path.join("artifacts", "preprocessor.pkl")
    transformed_train_data_path: str = os.path.join("artifacts", "transformed_train.csv")
    transformed_test_data_path: str = os.path.join("artifacts", "transformed_test.csv")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        创建数据转换管道 根据不同的数据类型进行转换
        包括数值特征的标准化和类别特征的独热编码。
        """
        try:
            numerical_columns = ["writing_score","reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]
            # 创建数据的特征管道 : 填充missing-value和数据标准化
            num_pipeline = Pipeline(
                steps =[("imputer",SimpleImputer(strategy="median")),
                        ("scaler",StandardScaler())
                ])

            # 创建类别特征的管道 : 填充missing-value和独热编码
            cat_pipeline = Pipeline(
                steps=[("imputer",SimpleImputer(strategy="most_frequent")),
                       ("one_hot_encoder",OneHotEncoder()),
                       ("scaler",StandardScaler(with_mean=False))
                ])
            logging.info("Numerical columns are standardized")
            logging.info("Category  columns are encoded")

            # 创建列转换器，将数值特征和类别特征的管道组合在一起
            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        启动数据转换流程。
        1. 读取训练集和测试集数据。
        2. 创建数据转换器对象。
        3. 对训练集和测试集进行转换。
        4. 保存转换后的数据到指定路径。
        """
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("获取预处理对象")

            preprocessor_obj = self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df = train_df.drop(columns = [target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns = [target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            logging.info("Applying preprocessing object on training and testing dataframes")
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)
            # 使用np.c_将转换后的特征和目标变量合并为一个数组
            train_arr = np.c_[input_feature_train_arr,input_feature_train_df]
            test_arr = np.c_[input_feature_test_arr,input_feature_test_df]

            logging.info("Saved preprocessing object")

            # 保存预处理对象到指定路径
            save_object(
                file_path = self.data_transformation_config.preprocessed_object_file_path,
                obj = preprocessor_obj
            )



            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessed_object_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)
