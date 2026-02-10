from temperatures.entity.artifact_entity import ClassificationMetricArtifact
import sys

def get_classification_score(best_model_metrics:dict)->ClassificationMetricArtifact:
    try:
            
        mean_absolute_error = best_model_metrics["mae"]
        mean_squared_error = best_model_metrics["rmse"]
        r2_score=best_model_metrics["r2"]

        classification_metric =  ClassificationMetricArtifact(mean_absolute_error=mean_absolute_error,
                    mean_squared_error=mean_squared_error, 
                    r2_score=r2_score)
        return classification_metric
    except Exception as e:
        raise Exception(f"Not possible retrive metrics.")