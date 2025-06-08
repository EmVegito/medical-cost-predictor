import os
import sys
from src.medical_price_predictor.logger import logging
from src.medical_price_predictor.exception import CustomException
import pandas as pd
import pymysql
from dotenv import load_dotenv

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