import os
from datetime import datetime
from temperatures.constant import constants


class TrainingPipelineConfig:
    def __init__(self,timestamp=datetime.now()):
        timestamp=timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.pipeline_name=constants.PIPELINE_NAME
        self.artifact_name=constants.ARTIFACT_DIR
        self.artifact_dir=os.path.join(self.artifact_name,timestamp)
        self.model_dir=os.path.join("final_model")
        self.timestamp: str=timestamp

class DataExtractionConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        self.file_name = constants.FILE_NAME
        self.raw_data_path = os.path.join(training_pipeline_config.artifact_dir, constants.DATA_EXTRACTION_DIR)
        self.mongo_database = constants.MONGO_DATABASE
        self.mongo_collection = constants.MONGO_COLLECTION


class DataIngestionConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        self.data_ingestion_path = os.path.join(training_pipeline_config.artifact_dir, constants.DATA_INGESTION_PATH)
        self.feature_store_path = os.path.join(self.data_ingestion_path, constants.FEATURE_STORE_PATH)
        self.ingested_path = os.path.join(self.data_ingestion_path, constants.INGESTED_PATH)
        self.scaler_path = os.path.join(self.data_ingestion_path,constants.SCALER_PATH)


class DataValidationConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        self.data_validation_path = os.path.join(training_pipeline_config.artifact_dir, constants.DATA_VALIDATION_DIR_NAME)
        self.data_validation_dir = os.path.join(self.data_validation_path, constants.DATA_VALIDATION_VALID_DIR)
        self.data_invalid_dir = os.path.join(self.data_validation_path, constants.DATA_VALIDATION_INVALID_DIR)
        self.drift_report_dir = os.path.join(self.data_validation_path, constants.DATA_VALIDATION_DRIFT_REPORT_DIR)
        self.drift_report_filename = os.path.join(self.drift_report_dir, constants.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME)