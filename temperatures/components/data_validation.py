import os
import pandas as pd
from scipy.stats import ks_2samp

from temperatures.constant.constants import SCHEMA_FILE_PATH

from temperatures.utils.utils import read_yaml_file, write_yaml_file

from temperatures.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from temperatures.entity.config_entity import DataValidationConfig


class DataValidation:
    def __init__(self, data_ingestion_artifact = DataIngestionArtifact,
                 data_validation_config = DataValidationConfig):
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_validation_config = data_validation_config
        self.schema = read_yaml_file(SCHEMA_FILE_PATH)

    
    def check_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            if len(self.schema.columns) == len(dataframe.columns):
                return True
            else:
                print("Train columns are not the same of the schema")
                return False
        except Exception as e:
            raise Exception("An error ocurred while checking number of columns")



    def detect_dataset_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, threshold=0.05, drift_ratio_threshold=0.2) -> bool:

        try:
            report = {}
            drift_count = 0
            total_columns = len(base_df.columns)

            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]
                
                # Statistic test
                is_same_dist = ks_2samp(d1, d2)
                
                # If p.value is minus than threshold
                if is_same_dist.pvalue < threshold:
                    is_found = True
                    drift_count += 1
                else:
                    is_found = False

                report.update({column: {
                    "p_value": float(is_same_dist.pvalue),
                    "drift_status": is_found
                }})

            # CÁLCULO DE ROBUSTEZ GLOBAL
            drift_ratio = drift_count / total_columns
            # El dataset está bien (status=True) si el ratio de drift es menor al umbral tolerado
            dataset_drift_status = drift_ratio < drift_ratio_threshold

            # Agregar métricas globales al reporte
            report["_global_metrics"] = {
                "total_columns": total_columns,
                "drift_columns_count": drift_count,
                "drift_ratio": drift_ratio,
                "threshold_limit": drift_ratio_threshold
            }

            # Guardar reporte
            drift_report_file_path = self.data_validation_config.drift_report_filename
            os.makedirs(os.path.dirname(drift_report_file_path), exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path, content=report)

            print(f"Drift detectado en el {drift_ratio*100:.2f}% de las columnas.")
            
            # Retornamos el estado global: True si está sano, False si hay demasiado drift
            return dataset_drift_status

        except Exception as e:
            raise Exception(f"An error ocurred while is doing drift detection {e}")



    def initiate_data_validation(self):

        try:
            train = pd.read_csv(self.data_ingestion_artifact.trained_file_path)
            test = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            # Check number of columns
            bool_train_columns = self.check_number_of_columns(train)
            bool_test_columns = self.check_number_of_columns(test)

            if not bool_test_columns:
                print("Test dataset does not contain the same columns than schema")

            if not bool_train_columns:
                print("Train dataset does not contain the same columns than schema")

            # Drift detect
            bool_drift = self.detect_dataset_drift(train, test)
        
            # If columns are correct in both df
            if bool_train_columns and bool_test_columns:
                os.makedirs(self.data_validation_config.data_validation_dir, exist_ok = True)
                train.to_csv(os.path.join(self.data_validation_config.data_validation_dir, 'train.csv'))
                test.to_csv(os.path.join(self.data_validation_config.data_validation_dir, 'test.csv'))
            else:
                os.makedirs(self.data_validation_config.data_invalid_dir, exist_ok = True)
                train.to_csv(os.path.join(self.data_validation_config.data_invalid_dir, 'train.csv'))
                test.to_csv(os.path.join(self.data_validation_config.data_invalid_dir, 'test.csv'))


            overall_bool = True if bool_test_columns and bool_train_columns and bool_drift else False

            data_validation_artifact = DataValidationArtifact(
                validation_status= overall_bool,
                valid_train_file_path=self.data_ingestion_artifact.trained_file_path,
                valid_test_file_path=self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path= self.data_validation_config.data_invalid_dir,
                invalid_test_file_path=self.data_validation_config.data_invalid_dir,
                drift_report_file_path=self.data_validation_config.drift_report_filename,
            )

            return data_validation_artifact
        except Exception as e:
            print(f"An error ocurred while trying to initiate data Validation. {e}")
