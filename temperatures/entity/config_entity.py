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
        self.raw_data_path = constants.DATA_EXTRACTION_DIR
        self.mongo_database = constants.MONGO_DATABASE
        self.mongo_collection = constants.MONGO_COLLECTION