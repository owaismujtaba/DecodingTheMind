import src.config as config




if __name__ == "__main__":

    if config.preprocessData:
        from src.preprocessing import perceptionImaginationPreProcessingPipeline

        perceptionImaginationPreProcessingPipeline()

