from src.medical_price_predictor.logger import logging
from src.medical_price_predictor.exception import CustomException
from src.medical_price_predictor.components.data_ingestion import DataIngestion
from src.medical_price_predictor.components.data_ingestion import DataIngestionConfig
from src.medical_price_predictor.components.data_transformation import DataTransformationConfig, DataTransformer

import sys

if __name__=="__main__":
    logging.info("The Execution has started.")
    
    try:
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

        #data_transformation_config = DataTransformationConfig()
        data_transformation = DataTransformer()
        data_transformation.initiate_data_transformation(train_data_path, test_data_path)


    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)

