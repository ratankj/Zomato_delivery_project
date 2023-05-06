import os 

# Root Directory 
ROOT_DIR_KEY = os.getcwd()

# Data file path 
DATA_DIR = "Data"
DATA_DIR_KEY = 'finalTrain.csv'

# Artifact folder
ARTIFACT_DIR_KEY = 'artifact' 

DATASET_URL= 'dataset_url'

# Data Ingestion constants
DATA_INGESTION_KEY = 'data ingestion'
TRAIN_DATA_DIR_KEY = 'train.csv'
TEST_DATA_DIR_KEY = 'test.csv' 
RAW_DATA_DIR_KEY = 'raw.csv'

DATA_TRANSFORMATION_ARTIFACT = 'data_transformation'
DATA_TRANSFORMATION_PREPROCESSING_OBJ = 'preprocessor.pkl'

MODEL_ARTIFACT = 'model_trainer'
MODEL_OBJECT = 'model.pkl'