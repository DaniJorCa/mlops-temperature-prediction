from dataclasses import dataclass


@dataclass
class DataExtractionArtifact:
    raw_data_file_path: str


@dataclass
class DataIngestionArtifact:
    trained_file_path:str
    test_file_path: str

@dataclass
class DataValidationArtifact:
    validation_status: str
    valid_train_file_path: str
    valid_test_file_path: str
    invalid_train_file_path: str
    invalid_test_file_path: str
    drift_report_file_path: str
    
@dataclass
class DataTransformationArtifact:
    processor_file_path: str
    transformed_train_path: str
    transformed_test_path: str


@dataclass
class ClassificationMetricArtifact:
    mean_absolute_error: float
    mean_squared_error: float
    r2_score: float


@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    train_metric_artifact: ClassificationMetricArtifact
    test_metric_artifact: ClassificationMetricArtifact
  
