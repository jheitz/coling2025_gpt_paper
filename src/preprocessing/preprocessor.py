from abc import abstractmethod
from dataloader.dataset import Dataset

class Preprocessor:
    name = "Generic preprocessor"

    def __init__(self, config, constants, run_parameters=None):
        self.config = config
        self.CONSTANTS = constants
        self.run_parameters = run_parameters

    @abstractmethod
    def preprocess_dataset(self, dataset: Dataset) -> Dataset:
        pass