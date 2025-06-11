import pandas as pd
import numpy as np
from dataclasses import dataclass
import os
import sys
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.medical_price_predictor.exception import CustomException
from src.medical_price_predictor.logger import logging
from src.medical_price_predictor.utils import encode_smoker, log_transformer, interaction_term, save_object
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformer:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    
    def get_transformation_obj(self, df: pd.DataFrame):
        '''
        this function is responsible for data transformation
        '''

        try:

            numerical_columns = df.select_dtypes(include="number").columns
            categorical_columns = df.select_dtypes(include='object').columns

            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy = 'median')),
                ("scalar", StandardScaler())
            ])
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot", OneHotEncoder()),
                ("scaler", StandardScaler(with_mean=False))
            ])
            logging.info(f'Categorical Columns: {categorical_columns}')
            logging.info(f'Numerical Columns: {numerical_columns}')

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipleine', cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor    

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Reading the train and test file and applying transformations")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            train_df_encoded = encode_smoker(train_df)
            test_df_encoded = encode_smoker(test_df)

            train_df_log_transformed =log_transformer(train_df_encoded, ['bmi', 'charges'])
            test_df_log_transformed =  log_transformer(test_df_encoded, ['bmi', 'charges'])

            train_df_int_term = interaction_term(train_df_log_transformed, ['smoker', 'age'])
            test_df_int_term = interaction_term(test_df_log_transformed, ['smoker', 'age'])
            print(train_df_int_term.head())
            logging.info("Transformation Applied")

            preprocessing_obj = self.get_transformation_obj(train_df_int_term.drop(columns=["charges"]))

            target_column = 'charges'

            target_feature_train_df = train_df_int_term[target_column]
            input_features_train_df = train_df_int_term.drop(columns=[target_column], axis=1)

            target_feature_test_df = test_df_int_term[target_column]
            input_features_test_df = test_df_int_term.drop(columns=[target_column], axis=1)


            logging.info('Applying Preprocessing on training and testing dataframe')

            input_feature_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_features_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info("Saving preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)