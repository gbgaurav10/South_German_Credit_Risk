from South_German_Bank.config.configuration import ConfigurationManager
from South_German_Bank.components.model_trainer import ModelTrainer
from South_German_Bank.components.model_evaluation import ModelEvaluation
from South_German_Bank.logging import logger

STAGE_NAME = "Model Evaluation Stage"

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass


    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation_config = ModelEvaluation(config=model_evaluation_config)
        model_evaluation_config.log_into_mlflow()


if __name__ == "__main__":
    try:
        logger.info(f">>>> Stage {STAGE_NAME} started <<<<")
        obj = ModelEvaluationTrainingPipeline()
        obj.main()
        logger.info(f">>>> Stage {STAGE_NAME} completed <<<<")
    except Exception as e:
        logger.exception(e)
        raise e