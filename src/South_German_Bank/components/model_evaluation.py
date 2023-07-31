import os 
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from South_German_Bank.entity.config_entity import ModelEvaluationConfig
from urllib.parse import urlparse
from South_German_Bank.constants import * 
from South_German_Bank.utils.common import read_yaml, create_directories, save_json
import mlflow
import mlflow.sklearn
import numpy as np 
import joblib


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        acc = accuracy_score(actual, pred)
        f1 = f1_score(actual, pred)
        p_score = precision_score(actual, pred)
        r_score = recall_score(actual, pred)
        ras = roc_auc_score(actual, pred)

        return acc, f1, p_score, r_score, ras
    
    def log_into_mlflow(self):

        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        X_test = test_data.drop(self.config.target_column, axis=1)
        y_test = test_data[[self.config.target_column]]

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme


        with mlflow.start_run():

            predicted_qualities = model.predict(X_test)

            (acc, f1, p_score, r_score, ras) = self.eval_metrics(y_test, predicted_qualities)


            # Saving metrics as local

            scores = {"accuracy_score": acc, "f1_score": f1, "precision_score": p_score, "recall_score": r_score, "roc_auc_score": ras}

            save_json(path=Path(self.config.metric_file_name), data=scores)

            mlflow.log_params(self.config.all_params)

            mlflow.log_metric("accuracy_score", acc)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("precision_score", p_score)
            mlflow.log_metric("recall_score", r_score)
            mlflow.log_metric("roc_auc_score", ras)


            # Model registry does not work with file store
            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(model, "model", registered_model_name="XGBClassifier")
            else:
                mlflow.sklearn.log_model(model, "model")