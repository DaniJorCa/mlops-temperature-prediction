from temperatures.entity.config_entity import (
    TrainingPipelineConfig, 
    DataExtractionConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig
)

from temperatures.components.data_extraction import DataExtraction 
from temperatures.components.data_ingestion import DataIngestion
from temperatures.components.data_validation import DataValidation
from temperatures.components.data_transformation import DataTransformation
from temperatures.components.model_trainer import ModelTrainer



if __name__=='__main__':
    try:
        trainingPipelineConfig=TrainingPipelineConfig()
        data_extraction_config=DataExtractionConfig(trainingPipelineConfig)
        data_extraction=DataExtraction(data_extraction_config)
        data_extraction_artifact=data_extraction.initiate_data_extraction()
        print(f"Raw data extracted succesfully from MongoDB {data_extraction_artifact}")

        data_ingestion_config=DataIngestionConfig(trainingPipelineConfig)
        data_ingestion = DataIngestion(data_ingestion_config,data_extraction_artifact)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        print(data_ingestion_artifact)

        data_validation_config = DataValidationConfig(trainingPipelineConfig)
        data_validation = DataValidation(data_ingestion_artifact, data_validation_config)
        data_validation_artifact = data_validation.initiate_data_validation()
        print(data_validation_artifact)

        data_transformation_config = DataTransformationConfig(trainingPipelineConfig)
        data_transformation = DataTransformation(data_validation_artifact, data_transformation_config)
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        print(data_transformation_artifact)

        model_trainer_config = ModelTrainerConfig(trainingPipelineConfig)
        model_trainer = ModelTrainer(data_transformation_artifact, model_trainer_config)
        model_train_artifact = model_trainer.initiate_train_model()
        print(model_train_artifact)


    except Exception as e:
        print(f"Imposible to load raw data {e}")
