from dataclasses import dataclass


@dataclass
class DataExtractionArtifact:
    raw_data_file_path: str


@dataclass
class DataIngestionArtifact:
    trained_file_path:str
    test_file_path: str
