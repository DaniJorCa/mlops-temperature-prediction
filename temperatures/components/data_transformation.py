import os
import pandas as pd
import numpy as np

from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler



from temperatures.constant import constants
from temperatures.utils.utils import save_pickle_object
from temperatures.entity.artifact_entity import DataTransformationArtifact
from temperatures.entity.config_entity import DataTransformationConfig
from temperatures.entity.artifact_entity import DataValidationArtifact



class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        self.data_validation_artifact = data_validation_artifact
        self.data_transformation_config = data_transformation_config

    def get_data_transformer_object(self):
        """
        Crea el objeto de transformación puro. 
        Primero Imputa y luego Escala.
        """
        try:
            imputer = KNNImputer(**constants.KNN_IMPUTER_PARAMS)
            scaler = StandardScaler()

            preprocessor = Pipeline(steps=[
                ("imputer", imputer),
                ("scaler", scaler)
            ])
            
            preprocessor.set_output(transform="pandas")
            return preprocessor
            
        except Exception as e:
            raise Exception(f"Error al crear el objeto preprocessor: {e}")

    def initiate_data_transformation(self):
        try:
            train_df = pd.read_csv(self.data_validation_artifact.valid_train_file_path)
            test_df = pd.read_csv(self.data_validation_artifact.valid_test_file_path)
            target_column = constants.TARGET_COLUMN

            # Separar Features y Target
            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            preprocessing_obj = self.get_data_transformer_object()
            
            X_train_final = preprocessing_obj.fit_transform(X_train)
            
            X_test_final = preprocessing_obj.transform(X_test)

            train_arr = pd.concat([X_train_final, y_train.reset_index(drop=True)], axis=1)
            test_arr = pd.concat([X_test_final, y_test.reset_index(drop=True)], axis=1)

            os.makedirs(self.data_transformation_config.data_transformed_path, exist_ok=True)
            
            # Guardar numpy arrays
            np.save(os.path.join(self.data_transformation_config.data_transformed_path, constants.TRAIN_TRANSFORMED_FILE), train_arr)
            np.save(os.path.join(self.data_transformation_config.data_transformed_path, constants.TEST_TRANSFORMED_FILE), test_arr)

            os.makedirs(self.data_transformation_config.preprocessor_path, exist_ok=True)
            save_pickle_object(
                obj=preprocessing_obj,
                obj_name= os.path.join(self.data_transformation_config.preprocessor_path, constants.PREPROCESSOR_NAME)
            )

            return DataTransformationArtifact(
                processor_file_path = self.data_transformation_config.preprocessor_path,
                transformed_test_path= os.path.join(self.data_transformation_config.data_transformation_path, constants.TEST_TRANSFORMED_FILE),
                transformed_train_path= os.path.join(self.data_transformation_config.data_transformation_path, constants.TRAIN_TRANSFORMED_FILE)
            )

        except Exception as e:
            raise Exception(f"Error en initiate_data_transformation: {e}")