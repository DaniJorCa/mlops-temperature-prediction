import os

"""
defining common constant variable for training pipeline
"""
TARGET_COLUMN = "target_t_max"
PIPELINE_NAME: str = "Temperatures"
ARTIFACT_DIR: str = "artifacts"
FILE_NAME: str = "temperatures.csv"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")

SAVED_MODEL_DIR =os.path.join("saved_models")
MODEL_FILE_NAME = "model.pkl"

KNN_IMPUTER_PARAMS: dict = { 
    "n_neighbors": 5, 
    "weights":'distance', 
    "add_indicator":True
    }


"""
Constants to extract pipeline
"""

MONGO_DATABASE = 'MLOps'
MONGO_COLLECTION = 'temperatures'
DATA_EXTRACTION_DIR = 'data_extraction'


"""
Constants to ingestion pipeline
"""

DATA_INGESTION_PATH: str = 'data_ingestion'
FEATURE_STORE_PATH: str = 'feature_store'
INGESTED_PATH: str = 'ingested'
FEATURE_STORE_NAME: str = 'temperatures.csv'
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2
RANDOM_STATE: int = 42


"""
Constants to Validation pipeline
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"


"""
Constants to Transformation pipeline
"""
DATA_TRANSFORMATION_PATH: str = "data_transformation"
PREPROCESSOR_PATH: str = "preprocessor"
DATA_TRANSFORMED_PATH: str = "transformed"
TRAIN_TRANSFORMED_FILE: str = "test.npy"
TEST_TRANSFORMED_FILE: str = "train.npy"
PREPROCESSOR_NAME: str = 'processor.pkl'



   
