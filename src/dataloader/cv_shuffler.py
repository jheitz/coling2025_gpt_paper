from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import os

class CrossValidationShuffler:
    """
    We have multiple different datasets. For each dataset, we want to ensure that the splits in cross validation are
    consistent, i.e. the same samples should go into train / test splits, no matter if there was any preprocessing or
    other details involved. This is important e.g. for ADReSS vs. ADReSS_from_PITT. The PITT-version is conceptually
    the same, but it is missing one element and thus the splits are completely different if done independently
    """

    def __init__(self, constants, config):
        self.mapping: pd.DataFrame = None
        self.CONSTANTS = constants
        self.config = config

    def create_mapping(self, n_splits):
        pass

    def get_mapping(self):
        assert self.mapping is not None, "self.mapping not initialized"
        return self.mapping

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, CrossValidationShuffler):
            if self.mapping is not None:
                # if mapping is available -> compare it
                return self.mapping.equals(other.mapping)
            elif other.mapping is not None:
                # other has mapping, self has not
                return False
            else:
                # both mappings are still None -> equivalence if the same (sub)class
                return self.__class__ == other.__class__
        return False

class ADReSSCrossValidationShuffler(CrossValidationShuffler):
    """
    ADReSS dataset
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_mapping(self, n_splits):
        # load sample names and corresponding labels (for stratified CV) from disk
        current_dir = os.path.dirname(os.path.realpath(__file__))
        self.mapping = pd.read_csv(
            os.path.join(current_dir, os.path.join(self.CONSTANTS.RESOURCES_DIR, "ADReSS_sample_name_label.csv")))

        # new column: test_split - we store the split where this is part of the test set
        #                          it will be used for training in all other splits
        self.mapping['test_split'] = -np.ones(self.mapping.shape[0])

        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=134)
        for split_idx, (train_indices, test_indices) in enumerate(kfold.split(X=self.mapping.sample_name, y=self.mapping.label)):
            self.mapping.iloc[test_indices, self.mapping.columns.get_indexer(['test_split'])[0]] = split_idx

        assert np.min(self.mapping['test_split']) == 0

