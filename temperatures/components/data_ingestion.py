import os
import pandas as pd

from sklearn.model_selection import train_test_split

from temperatures.utils.utils import save_pickle_object

from temperatures.constant import constants

from temperatures.entity.config_entity import DataIngestionConfig

from temperatures.entity.artifact_entity import DataExtractionArtifact

from temperatures.entity.artifact_entity import DataIngestionArtifact


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig, data_extraction_artifact : DataExtractionArtifact):
        try:
            self.data_ingestion_config = data_ingestion_config
            self.data_extraction_artifact = data_extraction_artifact
        except Exception as e:
            raise Exception(f"Error in DataIngestion Component {e}")
        
    def train_test_split(self, feature_store_df = pd.DataFrame):
        ratio_split = constants.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
        random_state = constants.RANDOM_STATE
        test, train = train_test_split(feature_store_df, test_size=float(ratio_split), random_state=int(random_state))

        # Save df
        os.makedirs(os.path.join(self.data_ingestion_config.ingested_path), exist_ok=True)

        train.to_csv(os.path.join(self.data_ingestion_config.ingested_path, constants.TRAIN_FILE_NAME), index = False)
        test.to_csv(os.path.join(self.data_ingestion_config.ingested_path, constants.TEST_FILE_NAME), index = False)

    def initiate_data_ingestion(self):
        df = pd.read_csv(self.data_extraction_artifact.raw_data_file_path)
        self.train_test_split(df)
        data_ingestion_artifact = DataIngestionArtifact(
            trained_file_path= os.path.join(self.data_ingestion_config.ingested_path, constants.TEST_FILE_NAME),
            test_file_path = os.path.join(self.data_ingestion_config.ingested_path, constants.TRAIN_FILE_NAME)
        )

        return data_ingestion_artifact

        

        