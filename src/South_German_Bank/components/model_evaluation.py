import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from South_German_Bank.entity.config_entity import ModelEvaluationConfig
import json
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

    def log_metrics_locally(self, acc, f1, p_score, r_score, ras):
        scores = {
            "accuracy_score": acc,
            "f1_score": f1,
            "precision_score": p_score,
            "recall_score": r_score,
            "roc_auc_score": ras
        }
        with open(self.config.metric_file_name, "w") as f:
            json.dump(scores, f)

    def evaluate_model(self):
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        X_test = test_data.drop(self.config.target_column, axis=1)
        y_test = test_data[[self.config.target_column]]

        predicted_qualities = model.predict(X_test)

        acc, f1, p_score, r_score, ras = self.eval_metrics(y_test, predicted_qualities)

        self.log_metrics_locally(acc, f1, p_score, r_score, ras)


