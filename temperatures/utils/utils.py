import os
import pickle
import yaml
import numpy as np
from box import ConfigBox
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV



def save_pickle_object(obj, obj_name):
    try:
        with open(obj_name, 'wb') as f:
            pickle.dump(obj, f)
    except Exception as e:
        raise Exception(f"Unable to save pickle object {e}")
    

def load_pickle_object(obj_path):
    try:
        with open(obj_path, 'rb') as f:
            obj = pickle.load(f)

        return obj
    except Exception as e:
        raise Exception(f"Unable to load pickle object {e}")
    

def read_yaml_file(file_path: str) -> ConfigBox:
    """
    Read a yaml file and return it.
    """
    try:
        with open(file_path, "rb") as yaml_file:
            return ConfigBox(yaml.safe_load(yaml_file))
    except Exception as e:
        raise Exception(f"Not possible read yaml file {file_path}")
    

def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise Exception(f"Not possible write yaml file {file_path}")


def evaluate_models(X_train, y_train,X_test,y_test,models_and_grids):
    try:
        report = {}

        for name_model, model_param in models_and_grids.items():

            para = model_param["params"]
            model = model_param["model"]

            tscv = TimeSeriesSplit(n_splits=3)

            gs = GridSearchCV(
                model, 
                para, 
                cv=tscv,
                scoring="r2",
                n_jobs=1
            )

            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_

            y_test_pred = best_model.predict(X_test)

            # Métricas
            r2 = r2_score(y_test, y_test_pred)
            mae = mean_absolute_error(y_test, y_test_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

            # Convertir errores a scores
            mae_score = 1 / (1 + mae)
            rmse_score = 1 / (1 + rmse)

            # Media final
            final_score = np.mean([r2, mae_score, rmse_score])

            report[name_model] = {
                "r2": r2,
                "mae": mae,
                "rmse": rmse,
                "final_score": final_score,
                "best_model": gs.best_estimator_
            }

        return report

    except Exception as e:
        raise Exception(f"Not possible evaluate model {e}")
    

def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise Exception("Not possible load numpy arrays. {e}")