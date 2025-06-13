"""从特定的数据源读取数据"""
import sys
import os
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.model_trainer import ModelTrainer
from src.utils import *
from src.components.model_trainer import ModelTrainerConfig
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

@dataclass
class DataIngestionConfig:
    """
    数据摄取配置类，定义了数据摄取的路径和相关参数。
    """
    train_data_path:str = os.path.join("../../artifacts", "train.csv")
    test_data_path:str = os.path.join("../../artifacts", "test.csv")
    raw_data_path:str = os.path.join("../../artifacts", "raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        启动数据摄取流程。
        1. 读取数据集。
        2. 分割数据集为训练集和测试集。
        3. 保存数据到指定路径。
        """
        logging.info("开始进行数据摄取")
        try:
            # 读取数据集
            # 将所有的 \ 替换为 /

            df = pd.read_csv("/Users/apple/PycharmProjects/PythonProject2/notebook/data/stud.csv")  # 替换为实际的数据源路径
            logging.info("Read the data as DataFrame")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            # 分割数据集
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info("Data split into train and test sets")

            # 保存数据到指定路径
            train_set.to_csv(self.ingestion_config.train_data_path, index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False,header=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Data ingestion completed successfully")

            # 返回训练集、测试集和原始数据的路径 以便为数据转换提供信息
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.raw_data_path
            )

        except Exception as e:
            raise CustomException(e, sys) from e

if __name__== "__main__":
    obj = DataIngestion()
    train_data,test_data,_ = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data,test_data)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))