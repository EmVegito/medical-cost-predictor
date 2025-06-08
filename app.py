from src.medical_price_predictor.logger import logging
from src.medical_price_predictor.exception import CustomException
from src.medical_price_predictor.components.data_ingestion import DataIngestion
from src.medical_price_predictor.components.data_ingestion import DataIngestionConfig

import sys

if __name__=="__main__":
    logging.info("The Execution has started.")
    
    try:
        data_ingestion = DataIngestion()
        data_ingestion.initiate_data_ingestion()
    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)

