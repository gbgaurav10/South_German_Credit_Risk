
from South_German_Bank.logging import logger
from South_German_Bank.pipeline.stage01_data_ingestion import DataIngestionTrainingPipeline
from South_German_Bank.pipeline.stage02_data_validation import DataValidationTrainingPipeline
from South_German_Bank.pipeline.stage03_data_transformation import DataTransformationTrainingPipeline

STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f">>>> Stage {STAGE_NAME} started <<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>> Stage {STAGE_NAME} completed <<<<")
except Exception as e:
    logger.exception(e)
    raise e



STAGE_NAME = "Data Validation Stage"
try:
    logger.info(f">>>> Stage {STAGE_NAME} started <<<<")
    obj = DataValidationTrainingPipeline()
    obj.main()
    logger.info(f">>>> Stage {STAGE_NAME} completed <<<<")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Data Transformation Stage"
try:
    logger.info(f">>>> Stage {STAGE_NAME} started <<<<")
    obj = DataTransformationTrainingPipeline()
    obj.main()
    logger.info(f">>>> Stage {STAGE_NAME} completed <<<<")
except Exception as e:
    logger.exception(e)
    raise e