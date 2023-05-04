import os
import sys
import pickle
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score

from zomato.exception import CustomException
from zomato.logger import logging

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
            
    except Exception as e:
        logging.info('Exception occured while saving an object')
        raise CustomException(e,sys)
    
