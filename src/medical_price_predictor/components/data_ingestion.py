import os
import sys
from src.medical_price_predictor.logger import logging
from src.medical_price_predictor.exception import CustomException
from src.medical_price_predictor.utils import read_sql_data
import pandas as pd
from sklearn.model_selection import train_test_split

from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts', 'train.csv')
    test_data_path: str=os.path.join('artifacts', 'test.csv')
    raw_data_path: str=os.path.join('artifacts', 'raw.csv')


class DataIngestion:
    def __init__(self, ):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            ##reading the data from mysql
            df = pd.read_csv(os.path.join('artifacts', 'raw.csv'))
            logging.info("Reading completed from MySql Database")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
