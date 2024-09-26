import src.config as config




if __name__ == "__main__":

    if config.preprocessData:
        from src.preprocessing import perception_imagination_preprocessingPipeline

        perception_imagination_preprocessingPipeline()

