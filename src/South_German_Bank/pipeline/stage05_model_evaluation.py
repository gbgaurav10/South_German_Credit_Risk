from South_German_Bank.config.configuration import ConfigurationManager
from South_German_Bank.components.model_evaluation import ModelEvaluation  
from South_German_Bank.logging import logger

STAGE_NAME = "Model Evaluation Stage"

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        
        # Instantiate ModelEvaluation with the provided configuration
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        
        # Perform model evaluation
        model_evaluation.evaluate_model()

if __name__ == "__main__":
    try:
        logger.info(f">>>> Stage {STAGE_NAME} started <<<<")
        obj = ModelEvaluationTrainingPipeline()
        obj.main()
        logger.info(f">>>> Stage {STAGE_NAME} completed <<<<")
    except Exception as e:
        logger.exception(e)
        raise e
