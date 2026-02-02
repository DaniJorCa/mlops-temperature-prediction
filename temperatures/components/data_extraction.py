import os
import pymongo
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from temperatures.entity.config_entity import DataExtractionConfig
from temperatures.constant.constants import DATA_EXTRACTION_DIR
from temperatures.entity.artifact_entity import DataExtractionArtifact

load_dotenv()

MONGO_URI = os.getenv("URI_MONGO_DB", '')


class DataExtraction:
    def __init__(self, data_extraction_config: DataExtractionConfig):
        try:
            self.data_extraction_config = data_extraction_config
        except Exception as e:
            raise(f"No se ha podido cargar la configuracion {e}")
        
    def export_collection_as_df(self):
        """
        Read data from MongoDB
        """
        try:
            db_name = self.data_extraction_config.mongo_database
            collection = self.data_extraction_config.mongo_collection
            self.mongo_client = pymongo.MongoClient(MONGO_URI)
            collection = self.mongo_client[db_name][collection]

            df = pd.DataFrame(list(collection.find()))
            if '_id' in df.columns.to_list():
                df = df.drop(columns=["_id"], axis=1)

            df.replace({"na":np.nan}, inplace=True)

            return df

        except Exception as e:
            raise(f"Not possible load data from MongoDB {e}")

    def save_raw_data(self, df:pd.DataFrame):
        try:
            os.makedirs(self.data_extraction_config.raw_data_path, exist_ok=1)
            df.to_csv(os.path.join(self.data_extraction_config.raw_data_path, self.data_extraction_config.file_name))
        except Exception as e:
            raise(f"Not possible save raw data from MongoDB to raw data folder {e}")

    def initiate_data_extraction(self):
        try:
            dataframe = self.export_collection_as_df()
            self.save_raw_data(dataframe)
            data_extraction_artifact = DataExtractionArtifact(
                raw_data_file_path = os.path.join(self.data_extraction_config.raw_data_path,self.data_extraction_config.file_name)
            )
        except Exception as e:
            raise(f"Not possible to initiate data extraction {e}")