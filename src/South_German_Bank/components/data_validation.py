import os
import urllib.request as request
import zipfile
from South_German_Bank.logging import logger
from South_German_Bank.utils.common import get_size
from pathlib import Path 
from South_German_Bank.entity.config_entity import (DataIngestionConfig, 
                                                    DataValidationConfig)
import pandas as pd





class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config


    def validate_all_columns(self)->bool:
        try:
            validation_status = None 

            data = pd.read_csv(self.config.unzip_data_dir)
            all_cols = list(data.columns)

            all_schema = self.config.all_schema.keys()


            for col in all_cols:
                if col not in all_schema:
                    validation_status = False 
                    with open(self.config.STATUS_FILE, "w") as f:
                        f.write(f"Validation Status: {validation_status}")
                else:
                    validation_status = True 
                    with open(self.config.STATUS_FILE, "w") as f:
                        f.write(f"Validation Status: {validation_status}")

            return validation_status
        
        except Exception as e:
            raise e