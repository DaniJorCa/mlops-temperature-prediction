from temperatures.entity.config_entity import (
    TrainingPipelineConfig, 
    DataExtractionConfig,
    DataIngestionConfig
)

from temperatures.components.data_extraction import DataExtraction 
from temperatures.components.data_ingestion import DataIngestion


if __name__=='__main__':
    try:
        trainingPipelineConfig=TrainingPipelineConfig()
        dataExtractionConfig=DataExtractionConfig(trainingPipelineConfig)
        data_extraction=DataExtraction(dataExtractionConfig)
        data_extraction_artifact=data_extraction.initiate_data_extraction()
        print(f"Raw data extracted succesfully from MongoDB {data_extraction_artifact}")

        dataIngestionConfig=DataIngestionConfig(trainingPipelineConfig)
        data_ingestion = DataIngestion(dataIngestionConfig,data_extraction_artifact)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        print(data_ingestion_artifact)

    except Exception as e:
        print(f"Imposible to load raw data {e}")
