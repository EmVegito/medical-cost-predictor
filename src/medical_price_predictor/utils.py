import os
import sys
from typing import List
from src.medical_price_predictor.logger import logging
from src.medical_price_predictor.exception import CustomException
import pandas as pd
import numpy as np
import pymysql
from dotenv import load_dotenv
import pickle

load_dotenv()

host=os.getenv("host")
user=os.getenv("user")
password=os.getenv("password")
db=os.getenv("db")

def read_sql_data():
    logging.info("Reading SQL Database Started.")
    try:
        mydb = pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db,
        )
        logging.info(f"Connection Established {mydb}")
        df = pd.read_sql_query('Select * from insurance', mydb)
        print(df.head())

        return df
    except Exception as ex:
        raise CustomException(ex)
    
def encode_smoker(df: pd.DataFrame) -> pd.DataFrame:

    '''
    This function encodes smoker variable to binary 1('yes') and 0('no')
    '''
    df_smoker_binary = df.copy()

    logging.info('Binary encoder of feature smoker started')
    df_smoker_binary['smoker'] = df['smoker'].map({
        'yes':1,
        'no':0,
    })

    logging.info('Smoker Binary encoded')

    return df_smoker_binary

def log_transformer(df: pd.DataFrame, features: List['str']) -> pd.DataFrame:
    df_log_transformed = df.copy()

    logging.info('Log transformation Started')

    for feature in features:
        df_log_transformed[feature] = np.log1p(
            df[feature]
        )

    logging.info('Log transformation Completed')
    
    return df_log_transformed


def interaction_term(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    df_with_interaction_term = df.copy()

    logging.info(f'Multiplying {features[0]} and {features[1]}')

    df_with_interaction_term[f"{features[0]}_{features[1]}"] = df[features[0]] * df[features[1]]

    logging.info(f'interaction term created.')
    return df_with_interaction_term

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)