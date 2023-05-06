from zomato.constant import *
from zomato.config.configuration import *
from zomato.logger import logging
from zomato.exception import CustomException
from zomato.components.data_transformation import DataTransformation
import os,sys
import pandas as pd
import numpy as np

from zomato.components import data_transformation
from sklearn.model_selection import train_test_split
from dataclasses import dataclass




@dataclass
class DataIngestionconfig:
    train_data_path:str=TRAIN_FILE_PATH
    test_data_path:str=TEST_FILE_PATH
    raw_data_path:str= RAW_FILE_PATH

## create a class for Data Ingestion
class DataIngestion:
    def __init__(self):
        self.data_ingestion_config=DataIngestionconfig()

    def initiate_data_ingestion(self):
        logging.info("="*50)
        logging.info("Initiate Data Ingestion config")
        logging.info("="*50)
        
        
        try:
            df=pd.read_csv(DATASET_PATH)
            #df=pd.read_csv(os.path.join("D:\DATASET\delivery_dataset\finalTrain.csv"))
            logging.info(f"Download data {DATASET_PATH}")

            logging.info('Dataset read as pandas Dataframe')

            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path),exist_ok=True)  
            df.to_csv(self.data_ingestion_config.raw_data_path,index=False)   

            df.drop(['ID'],axis=1,inplace=True) 

            logging.info('Train test split')
            #train_set = None
            #test_set = None

            train_set,test_set=train_test_split(df,test_size=0.30,random_state=42)

            train_set.to_csv(self.data_ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.data_ingestion_config.test_data_path,index=False,header=True)

            logging.info('Ingestion of Data is completed')


            return(
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )
  
            
        except Exception as e:
            logging.info('Exception occured at Data Ingestion stage')
            raise CustomException(e,sys)



# run data ingestion 
#if __name__ == "__main__":
 #   obj = DataIngestion()
  #  train_data,test_data=obj.initiate_data_ingestion()


# to run this
# pyhton zomato/components/data_ingestion.py

# run data_tranasformation

#if __name__ == "__main__":
 #   obj = DataIngestion()
  #  train_data_path,test_data_path=obj.initiate_data_ingestion()
   # data_transformation = DataTransformation()
    #train_arr,test_arr,_ = data_transformation.initaite_data_transformation(train_data_path,test_data_path)


# to run this
# pyhton zomato/components/data_ingestion.py
   