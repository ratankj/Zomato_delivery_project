import os
import sys
from zomato.logger import logging
from zomato.exception import CustomException
import pandas as pd

from zomato.components.data_ingestion import DataIngestion
from zomato.components.data_transformation import DataTransformation
from zomato.components.model_trainer import ModelTrainer

class Train:
    def __init__(self):
        self.c = 0 
        print(f"-----------{self.c}------------")


    def main(self):
#if __name__=='__main__':
        obj=DataIngestion()
        train_data_path,test_data_path=obj.initiate_data_ingestion()
        data_transformation = DataTransformation()
        train_arr,test_arr,_=data_transformation.initaite_data_transformation(train_data_path,test_data_path)
        model_trainer=ModelTrainer()
        model_trainer.initate_model_training(train_arr,test_arr)





# to run this code use
#python zomato/pipeline/training_pipeline.py