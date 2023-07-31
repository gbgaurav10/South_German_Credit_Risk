import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import os
from South_German_Bank.logging import logger
from South_German_Bank.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.preprocessor = None  # Initialize preprocessor as None
        self.transformed_df = None  # Initialize transformed DataFrame as None

    def get_data_transformation(self):
        try:
            # Load the data
            df = pd.read_csv(self.config.data_path)

            # Separate target variable
            X = df.drop("credit_risk", axis=1)
            y = df["credit_risk"]

            # Map target variable "credit_risk" to 1 and 0
            y.replace({"good": 1, "bad": 0}, inplace=True)

            numerical_features = X.select_dtypes(exclude="object").columns
            categorical_features = X.select_dtypes(include="object").columns

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("robustscaler", RobustScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("ordinalencoder", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
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

            # Transform the whole data using the preprocessor
            X_transformed = preprocessor.fit_transform(X)

            # Get the updated column names after ordinal encoding
            column_names = numerical_features.tolist() + categorical_features.tolist()

            # Combine X_transformed and y back into one DataFrame
            self.transformed_df = pd.DataFrame(X_transformed, columns=column_names)
            self.transformed_df['credit_risk'] = y

            logger.info("Data preprocessing done")

        except Exception as e:
            raise e

    def train_test_split(self):
        if self.preprocessor is None:
            raise ValueError("Preprocessor is not available. Please call get_data_transformation first.")

        # Split the data into train and test sets
        train, test = train_test_split(self.transformed_df)

        # Save the encoded train and test sets to CSV files
        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        logger.info("Splited data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)