import pandas as pd
import numpy as np
import os

class CustomDataSplitter:
    """
    In some cases, we might want to split the dataset in a custom, but reproducible way
    """

    def __init__(self, constants):
        self.mapping: pd.DataFrame = None
        self.CONSTANTS = constants
        self.create_mapping()

    def create_mapping(self):
        pass

    def get_mapping(self):
        assert self.mapping is not None, "self.mapping not initialized"
        return self.mapping

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, CustomDataSplitter):
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

class ADReSSTrainValTestSplitter(CustomDataSplitter):
    """
    ADReSS dataset, equally split into train, val, test, where test is the original test set,
    train & val are stratified
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_mapping(self):
        # load custom
        current_dir = os.path.dirname(os.path.realpath(__file__))
        self.mapping = pd.read_csv(
            os.path.join(current_dir, os.path.join(self.CONSTANTS.RESOURCES_DIR,
                                                   "ADReSS_train_validation_test_splits.csv")))

        # some data checks
        assert self.mapping.shape[0] == self.mapping['sample_name'].drop_duplicates().shape[0]
        assert set(self.mapping.split.drop_duplicates()) == {'Train', 'Validation', 'Test'}

        # check that splits are of roughly equal size (test is a bit smaller)
        assert (np.sum(self.mapping['split'] == 'Train') == np.sum(self.mapping['split'] == 'Validation')
                and np.sum(self.mapping['split'] == 'Train') * 0.8 < np.sum(self.mapping['split'] == 'Test') < np.sum(self.mapping['split'] == 'Train')), \
            f"Split sizes: train: {np.sum(self.mapping['split'] == 'Train')}, val: {np.sum(self.mapping['split'] == 'Validation')}, test: {np.sum(self.mapping['split'] == 'Test')}"

        # check that sample names are all present
        sample_name_label = pd.read_csv(os.path.join(current_dir, os.path.join(self.CONSTANTS.RESOURCES_DIR,
                                                                               "ADReSS_sample_name_label.csv")))
        assert set(sample_name_label.sample_name) == set(self.mapping.sample_name)

        # check that labels are present equally in each split
        merged = self.mapping.merge(sample_name_label, on="sample_name", how="inner")
        for s in self.mapping.split.drop_duplicates():
            pos = merged.query(f"split == '{s}' and label == 1")
            neg = merged.query(f"split == '{s}' and label == 0")
            assert pos.shape[0] > 0
            assert pos.shape[0] == neg.shape[0], \
                f"Split {s} has invalid distribution of labels: 1: {pos.shape[0]}, 0: {neg.shape[0]}"



