"""Abstract base model"""

from abc import ABC, abstractmethod
from config.config import Config
from config.constants import Constants
from config.run_parameters import RunParameters

from dataloader.dataset import Dataset


class BaseModel(ABC):
    """Abstract Model class that is inherited to all models"""

    def __init__(self, name, run_parameters, config, constants):
        self.name = name
        print(f"Initializing model {self.name}")
        self.config = None
        if config is None or not isinstance(config, Config):
            raise Exception("Model should have a config")
        if run_parameters is None or not isinstance(run_parameters, RunParameters):
            raise Exception("Model should have a config")
        if constants is None or not isinstance(constants, Constants):
            raise Exception("Model should have CONSTANTS")

        self.config = config
        self.run_parameters = run_parameters
        self.CONSTANTS = constants

    @abstractmethod
    def prepare_data(self):
        pass

    @abstractmethod
    def set_train(self, dataset: Dataset):
        pass

    @abstractmethod
    def set_test(self, dataset: Dataset):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass

    def train_test(self):
        self.train()
        self.test()
