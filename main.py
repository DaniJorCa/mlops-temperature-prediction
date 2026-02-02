from temperatures.entity.config_entity import TrainingPipelineConfig

from temperatures.components.data_extraction import DataExtraction 

from temperatures.components.data_extraction import DataExtractionConfig


if __name__=='__main__':
    try:
        trainingpipelineconfig=TrainingPipelineConfig()
        dataextractionconfig=DataExtractionConfig(trainingpipelineconfig)
        data_extraction=DataExtraction(dataextractionconfig)
        data_extraction_artifact=data_extraction.initiate_data_extraction()
    except Exception as e:
        print("Imposible to load raw data")
