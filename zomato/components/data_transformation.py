import os , sys
from zomato.exception import CustomException
from zomato.constant import *
from zomato.logger import logging
#from zomato.components.data_ingestion import DataIngestion
from zomato.config.configuration import PREPROCESSING_OBJ_PATH


from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder,OneHotEncoder
from sklearn.pipeline import Pipeline
from zomato.utils import save_object



@dataclass
class DataTransformationConfig():
    preprocessor_obj_file_path=PREPROCESSING_OBJ_PATH


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')


            # defining the ordinal data ranking
            Road_traffic_density=['Low','Medium','High','Jam']
            Weather_conditions=['Sunny','Cloudy','Windy','Fog','Sandstorms','Stormy']
            

            # defining the categorical and numerical column
            categorical_column=['Type_of_order','Type_of_vehicle','Festival','City','Delivery_city']
            ordinal_encod=['Road_traffic_density','Weather_conditions']
            numerical_column=['Delivery_person_Age','Delivery_person_Ratings','Vehicle_condition','multiple_deliveries','distance']
            

            # numerical pipeline

            numerical_pipeline=Pipeline(steps=[
                ('impute',SimpleImputer(strategy='constant',fill_value=0)),
                ('scaler',StandardScaler(with_mean=False))
                ])



            # categorical pipeline

            categorical_pipeline=Pipeline(steps=[
                ('impute',SimpleImputer(strategy='most_frequent')),
                ('onehot',OneHotEncoder(handle_unknown='ignore')),
                ('scaler',StandardScaler(with_mean=False))
                ])




            # ordinal pipeline

            ordianl_pipeline=Pipeline(steps=[
                ('impute',SimpleImputer(strategy='most_frequent')),
                ('ordinal',OrdinalEncoder(categories=[Road_traffic_density,Weather_conditions])),
                ('scaler',StandardScaler(with_mean=False))   
                ])
            
            # preprocessor

            preprocessor =ColumnTransformer([
                ('numerical_pipeline',numerical_pipeline,numerical_column),
                ('categorical_pipeline',categorical_pipeline,categorical_column),
                ('ordianl_pipeline',ordianl_pipeline,ordinal_encod)
                ])


            # returning preprocessor

            return preprocessor
        
            logging.info('Pipeline Completed')



        except Exception as e:
            logging.info("error in data transformation")
            raise CustomException(e,sys)
    ''' 
    def distance(self,lat1, lon1, lat2, lon2):
    # Convert latitude and longitude to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        R = 6371.0 # Earth's radius in km
        dist = R * c
    
        return dist
    '''

    def distance_numpy(self,df,lat1, lon1, lat2, lon2):
        p = np.pi/180
        a = 0.5 - np.cos((df[lat2]-df[lat1])*p)/2 + np.cos(df[lat1]*p) * np.cos(df[lat2]*p) * (1-np.cos((df[lon2]-df[lon1])*p))/2
        df['distance'] = 12742 * np.arcsin(np.sqrt(a))
    




    def initaite_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')


            logging.info("Creating feature on latitude nad longitude")
            self.distance_numpy(train_df,'Restaurant_latitude','Restaurant_longitude', 
                                'Delivery_location_latitude','Delivery_location_longitude')
            self.distance_numpy(test_df,'Restaurant_latitude','Restaurant_longitude', 
                                'Delivery_location_latitude','Delivery_location_longitude')


# adding delivery city
            train_df['Delivery_city']=train_df['Delivery_person_ID'].str.split('RES',expand=True)[0]
            test_df['Delivery_city']=test_df['Delivery_person_ID'].str.split('RES',expand=True)[0]


            logging.info('Obtaining preprocessing object')

# preprocessing object

            preprocessing_obj = self.get_data_transformation_object()

# column to drop
            train_df.drop(['Delivery_person_ID','Restaurant_latitude','Restaurant_longitude','Delivery_location_latitude','Delivery_location_longitude',
        'Order_Date','Time_Orderd','Time_Order_picked'],axis=1,inplace=True)
            
            test_df.drop(['Delivery_person_ID','Restaurant_latitude','Restaurant_longitude','Delivery_location_latitude','Delivery_location_longitude',
        'Order_Date','Time_Orderd','Time_Order_picked'],axis=1,inplace=True)


            logging.info(f'Train Dataframe Head: \n{train_df.head().to_string()}')
            logging.info(f'Train Dataframe Head: \n{test_df.head().to_string()}')

# target column

            target_column_name = 'Time_taken (min)'
            #drop_columns = [target_column_name,'id']

            X_train = train_df.drop(columns=target_column_name,axis=1)
            y_train=train_df[target_column_name]

            X_test=test_df.drop(columns=target_column_name,axis=1)
            y_test=test_df[target_column_name]

            logging.info(f"shape of {X_train.shape} and {X_test.shape}")
            logging.info(f"shape of {y_train.shape} and {y_test.shape}")

            # Transforming using preprocessor obj
            
            X_train=preprocessing_obj.fit_transform(X_train)            
            X_test=preprocessing_obj.transform(X_test)
            logging.info("Applying preprocessing object on training and testing datasets.")
            logging.info(f"shape of {X_train.shape} and {X_test.shape}")
            logging.info(f"shape of {y_train.shape} and {y_test.shape}")
            

            logging.info("transformation completed")
            


            train_arr = np.c_[X_train, np.array(y_train)]
            test_arr = np.c_[X_test, np.array(y_test)]

            logging.info("train_arr, test_arr completed")


            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj)
            
            logging.info("Preprocessor file saved")
            
            return(train_arr,
                   test_arr,
                   self.data_transformation_config.preprocessor_obj_file_path)



        except Exception as e:
            logging.info("error in data transformation")
            raise CustomException(e,sys)

