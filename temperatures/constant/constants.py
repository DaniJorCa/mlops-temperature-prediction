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
SCALER_PATH: str = 'scalers'
FEATURE_STORE_NAME: str = 'temperatures.csv'
SCALER_FEATURES_NAME: str = 'X_scaler.pkl'
SCALER_TARGET_NAME: str = 'y_scaler.pkl'
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2
RANDOM_STATE: int = 42