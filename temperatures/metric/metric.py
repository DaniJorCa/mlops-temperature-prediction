import sys
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


from temperatures.entity.artifact_entity import ClassificationMetricArtifact


def get_classification_score(X_train, y_train, y_train_pred, X_test, y_test, y_test_pred)->ClassificationMetricArtifact:


    try:
        # ---- TRAIN METRICS ----
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_r2 = r2_score(y_train, y_train_pred)

        train_metrics = ClassificationMetricArtifact(
            mean_absolute_error=train_mae,
            mean_squared_error=train_rmse,
            r2_score=train_r2
        )

        # ---- TEST METRICS ----
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_r2 = r2_score(y_test, y_test_pred)

        test_metrics = ClassificationMetricArtifact(
            mean_absolute_error=test_mae,
            mean_squared_error=test_rmse,
            r2_score=test_r2
        )

        return train_metrics, test_metrics

    except Exception as e:
        raise Exception(f"Not possible to retrieve metrics: {e}")
    