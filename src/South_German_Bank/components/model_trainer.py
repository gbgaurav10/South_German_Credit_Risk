import pandas as pd
import os 
from South_German_Bank.logging import logger
from xgboost import XGBClassifier
import joblib
from South_German_Bank.entity.config_entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        X_train = train_data.drop([self.config.target_column],axis=1)
        X_test = test_data.drop([self.config.target_column], axis=1)
        y_train = train_data[[self.config.target_column]]
        y_test = test_data[[self.config.target_column]]

        xgb = XGBClassifier(n_estimators=self.config.n_estimators, max_depth=self.config.max_depth, min_child_weight=self.config.min_child_weight, random_state=42)
        xgb.fit(X_train, y_train)

        joblib.dump(xgb, os.path.join(self.config.root_dir, self.config.model_name))