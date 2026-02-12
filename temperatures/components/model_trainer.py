import os
import dagshub
import mlflow
from urllib.parse import urlparse
from dotenv import load_dotenv

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor


from temperatures.entity.artifact_entity import ModelTrainerArtifact
from temperatures.utils.ml_utils import TemperaturePredictorModel
from temperatures.utils.utils import evaluate_models, load_pickle_object, save_pickle_object, load_numpy_array_data
from temperatures.metric.metric import get_classification_score
from temperatures.entity.artifact_entity import DataTransformationArtifact
from temperatures.entity.config_entity import ModelTrainerConfig


#dagshub.init(repo_owner='DaniJorCa', repo_name='mlops-temperature-prediction', mlflow=True)

load_dotenv()


MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")


class ModelTrainer():
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 data_model_trainer_config: ModelTrainerConfig):
        self.data_transformation_artifact = data_transformation_artifact
        self.data_model_trainer_config = data_model_trainer_config

    

    def track_mlflow(self, best_model, model_name, train_metrics, test_metrics):
        mlflow.set_registry_uri(MLFLOW_TRACKING_URI)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        with mlflow.start_run():
            train_mean_absolute_error=train_metrics.mean_absolute_error
            train_mean_squared_error=train_metrics.mean_squared_error
            train_r2_score=train_metrics.r2_score

            test_mean_absolute_error=test_metrics.mean_absolute_error
            test_mean_squared_error=test_metrics.mean_squared_error
            test_r2_score=test_metrics.r2_score

            mlflow.log_metric("train_mean_absolute_error",train_mean_absolute_error)
            mlflow.log_metric("train_mean_squared_error",train_mean_squared_error)
            mlflow.log_metric("train_r2_score",train_r2_score)

            mlflow.log_metric("test_mean_absolute_error",test_mean_absolute_error)
            mlflow.log_metric("test_mean_squared_error",test_mean_squared_error)
            mlflow.log_metric("test_r2_score",test_r2_score)

            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the modelgoogle.es\
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.sklearn.log_model(best_model, "model", registered_model_name=model_name)
            else:
                mlflow.sklearn.log_model(best_model, "model")


    def train_models(self, X_train, y_train, X_test, y_test):

        models_and_grids = {

            "random_forest": {
                "model": RandomForestRegressor(),
                "params": {
                    "n_estimators": [100, 300],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2]
                }
            },

            # "gradient_boosting": {
            #     "model": GradientBoostingRegressor(),
            #     "params": {
            #         "n_estimators": [100, 300],
            #         "learning_rate": [0.01, 0.05, 0.1],
            #         "max_depth": [3, 5]
            #     }
            # },

            # "knn": {
            #     "model": KNeighborsRegressor(),
            #     "params": {
            #         "n_neighbors": [3, 5, 7, 11],
            #         "weights": ["uniform", "distance"],
            #         "p": [1, 2]
            #     }
            # }

        }

        model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test, models_and_grids=models_and_grids)

        ## To get best model score from dict
        best_model_score = max(sorted([x["final_score"] for x in model_report.values()]))

        best_model_name = list(model_report.keys())[[x["final_score"] for x in model_report.values()].index(best_model_score)]
        
        best_model = model_report[best_model_name]["best_model"]

        if best_model:
            y_train_pred=best_model.predict(X_train)
            y_test_pred=best_model.predict(X_test)

            train_metrics, test_metrics=get_classification_score(X_train, y_train, y_train_pred, X_test, y_test, y_test_pred)

            ## Track the experiements with mlflow
            self.track_mlflow(best_model, best_model_name, train_metrics=train_metrics, test_metrics=test_metrics)

            preprocessor = load_pickle_object(obj_path=self.data_transformation_artifact.processor_file_path)

            model_dir_path = os.path.dirname(self.data_model_trainer_config.model_trainer_path)
            os.makedirs(model_dir_path,exist_ok=True)


            os.makedirs(self.data_model_trainer_config.predictor_object_path,exist_ok=True)
            temperature_predictor_model=TemperaturePredictorModel(preprocessor=preprocessor,model=best_model)
            save_pickle_object(obj=temperature_predictor_model, obj_name=self.data_model_trainer_config.predictor_object_file)
            
            
            #model pusher
            os.makedirs(self.data_model_trainer_config.model_trained_path,exist_ok=True)
            save_pickle_object(obj=best_model, obj_name=self.data_model_trainer_config.model_trained_file)

            ## Model Trainer Artifact
            model_trainer_artifact=ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_metrics,
                test_metric_artifact=test_metrics
            )

            return model_trainer_artifact
        else:
            print(f"Not possible to get best model from grid search")
            raise Exception("Not possible to get Best model from GridSearch")

    def initiate_train_model(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_path
            test_file_path = self.data_transformation_artifact.transformed_test_path

            #loading training array and testing array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            print(train_arr)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            model_trainer_artifact=self.train_models(x_train,y_train,x_test,y_test)
            return model_trainer_artifact

            
        except Exception as e:
            raise Exception(f"Error while initiate train model {e}")