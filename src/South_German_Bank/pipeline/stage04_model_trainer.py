from South_German_Bank.config.configuration import ConfigurationManager
from South_German_Bank.components.model_trainer import ModelTrainer
from South_German_Bank.logging import logger

STAGE_NAME = "Model Training Stage"

class ModelTrainingTrainingPipeline:
    def __init__(self):
        pass


    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer_config = ModelTrainer(config=model_trainer_config)
        model_trainer_config.train()


if __name__ == "__main__":
    try:
        logger.info(f">>>> Stage {STAGE_NAME} started <<<<")
        obj = ModelTrainingTrainingPipeline()
        obj.main()
        logger.info(f">>>> Stage {STAGE_NAME} completed <<<<")
    except Exception as e:
        logger.exception(e)
        raise e