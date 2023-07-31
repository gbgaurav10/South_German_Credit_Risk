from South_German_Bank.config.configuration import ConfigurationManager
from South_German_Bank.components.data_ingestion import DataIngestion
from South_German_Bank.components.data_transformation import DataTransformation
from South_German_Bank.logging import logger

STAGE_NAME = "Data Transformation Stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass 

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.get_data_transformation()
        data_transformation.train_test_split()


if __name__ == "__main__":
    try:
        logger.info(f">>>> Stage {STAGE_NAME} started <<<<")
        obj = DataTransformationTrainingPipeline()
        obj.main()
        logger.info(f">>>> Stage {STAGE_NAME} completed <<<<")
    except Exception as e:
        logger.exception(e)
        raise e