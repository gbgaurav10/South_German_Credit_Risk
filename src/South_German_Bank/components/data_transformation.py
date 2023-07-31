import os
from South_German_Bank.logging import logger
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, RobustScaler
from South_German_Bank.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.preprocessor = None  # Initialize preprocessor as None

    def get_data_transformation(self):
        try:
            # Load the data
            df = pd.read_csv(self.config.data_path)

            numerical_features = df.select_dtypes(exclude="object").columns
            categorical_features = df.select_dtypes(include="object").columns

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("robustscaler", RobustScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("ordinalEncoder", OrdinalEncoder()),
                    ("robustscaler", RobustScaler())
                ]
            )

            logger.info(f"Categorical columns: {categorical_features}")
            logger.info(f"Numerical columns: {numerical_features}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_features),
                    ("cat_pipeline", cat_pipeline, categorical_features)
                ]
            )

            self.preprocessor = preprocessor  # Store the preprocessor for later use
            logger.info("Data preprocessing done")

        except Exception as e:
            raise e

    def train_test_split(self):
        if self.preprocessor is None:
            raise ValueError("Preprocessor is not available. Please call get_data_transformation first.")

        data = pd.read_csv(self.config.data_path)

        # Transform the data using the preprocessor
        transformed_data = self.preprocessor.fit_transform(data)

        # Convert the transformed data back to a DataFrame
        transformed_df = pd.DataFrame(transformed_data, columns=data.columns)

        # Split the data into train and test sets
        train, test = train_test_split(transformed_df)

        # Save the encoded train and test sets to CSV files
        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        logger.info("Splited data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)