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

#SAVED_MODEL_DIR =os.path.join("saved_models")
#MODEL_FILE_NAME = "model.pkl"


"""
Constants to extract pipeline
"""

MONGO_DATABASE = 'MLOps'
MONGO_COLLECTION = 'temperatures'
DATA_EXTRACTION_DIR = 'data_extraction'