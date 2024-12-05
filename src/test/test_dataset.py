import pytest
import numpy as np
import pandas as pd

from dataloader.dataset import TabularDataset

class TestTabularDatasetMerge:
    def _init_data(self):
        self.one_sample_names = np.array(['a', 'b', 'c', 'd'])
        self.one_labels = np.array([1, 1, 0, 0])
        self.one_data = pd.DataFrame({'feature1': [10, 11, 12, 13], 'feature2': [20, 21, 22, 23]})
        self.one = TabularDataset(sample_names=self.one_sample_names, labels=self.one_labels, data=self.one_data,
                                  name="DS1")

        self.two_sample_names = np.array(['a', 'b', 'c', 'd'])
        self.two_labels = np.array([1, 1, 0, 0])
        self.two_data = pd.DataFrame({'feature3': [30, 31, 32, 33]})
        self.two = TabularDataset(sample_names=self.two_sample_names, labels=self.two_labels, data=self.two_data,
                                  name="DS2")
    def test_basic(self):
        # basic functionality
        self._init_data()
        merged = self.one.merge(self.two)
        assert np.all(merged.sample_names == np.array(['a', 'b', 'c', 'd']))
        assert np.all(merged.labels == np.array([1, 1, 0, 0]))
        assert merged.data.equals(pd.DataFrame({'feature1': [10, 11, 12, 13], 'feature2': [20, 21, 22, 23],
                                                'feature3': [30, 31, 32, 33]}))

    def test_reordered(self):
        # reorder samples in second dataset
        self._init_data()
        self.two_sample_names = np.array(['b', 'c', 'd', 'a'])  # reordered
        self.two_labels = np.array([1, 0, 0, 1])  # reordered
        self.two = TabularDataset(sample_names=self.two_sample_names, labels=self.two_labels, data=self.two_data,
                                  name="DS2")

        merged = self.one.merge(self.two)
        assert np.all(merged.sample_names == np.array(['a', 'b', 'c', 'd']))
        assert np.all(merged.labels == np.array([1, 1, 0, 0]))
        assert merged.data.equals(pd.DataFrame({'feature1': [10, 11, 12, 13], 'feature2': [20, 21, 22, 23],
                                                'feature3': [33, 30, 31, 32]}))

    def test_missing_row(self):
        # missing sample in two
        self._init_data()
        self.two_sample_names = self.two_sample_names[:-1]
        self.two_labels = self.two_labels[:-1]
        self.two_data = self.two_data.iloc[:-1, :]
        self.two = TabularDataset(sample_names=self.two_sample_names, labels=self.two_labels, data=self.two_data,
                                  name="DS2")

        # should raise exception
        with pytest.raises(Exception):
            merged = self.one.merge(self.two)

    def test_incompatible_sample_name(self):
        # different sample name
        self._init_data()
        self.two_sample_names = np.array(['a', 'b', 'c', 'f'])
        self.two = TabularDataset(sample_names=self.two_sample_names, labels=self.two_labels, data=self.two_data,
                                  name="DS2")

        # should raise exception
        with pytest.raises(Exception):
            merged = self.one.merge(self.two)

    def test_incompatible_label(self):
        # different order of labels
        self._init_data()
        self.two_labels = np.array([1, 0, 1, 0])
        self.two = TabularDataset(sample_names=self.two_sample_names, labels=self.two_labels, data=self.two_data,
                                  name="DS2")

        # should raise exception
        with pytest.raises(Exception):
            merged = self.one.merge(self.two)

    def test_incompatible_vars(self):
        # additional paths, should not work
        self._init_data()
        self.two = TabularDataset(sample_names=self.two_sample_names, labels=self.two_labels, data=self.two_data,
                                  name="DS2", paths=self.two_sample_names)

        # should raise exception
        with pytest.raises(Exception):
            merged = self.one.merge(self.two)

    def test_compatible_vars(self):
        # additional paths in both, should work
        self._init_data()
        self.one = TabularDataset(sample_names=self.one_sample_names, labels=self.one_labels, data=self.one_data,
                                  name="DS1", paths=self.one_sample_names)

        self.two = TabularDataset(sample_names=self.two_sample_names, labels=self.two_labels, data=self.two_data,
                                  name="DS2", paths=self.two_sample_names)


        merged = self.one.merge(self.two)
        assert np.all(merged.sample_names == np.array(['a', 'b', 'c', 'd']))
        assert np.all(merged.labels == np.array([1, 1, 0, 0]))
        assert merged.data.equals(pd.DataFrame({'feature1': [10, 11, 12, 13], 'feature2': [20, 21, 22, 23],
                                                'feature3': [30, 31, 32, 33]}))
        assert np.all(merged.paths == np.array(['a', 'b', 'c', 'd']))




class TestDatasetConcatenation:
    def _init_data(self):
        self.one_sample_names = np.array(['a', 'b', 'c', 'd'])
        self.one_labels = np.array([1, 1, 0, 0])
        self.one_data = pd.DataFrame({'feature1': [10, 11, 12, 13]})
        self.one = TabularDataset(sample_names=self.one_sample_names, labels=self.one_labels,
                                  data=self.one_data,
                                  name="DS1")

        self.two_sample_names = np.array(['x', 'y', 'z'])
        self.two_labels = np.array([1, 1, 0])
        self.two_data = pd.DataFrame({'feature1': [30, 31, 32]})
        self.two = TabularDataset(sample_names=self.two_sample_names, labels=self.two_labels,
                                  data=self.two_data,
                                  name="DS2")
    def test_basic(self):
        # basic functionality
        self._init_data()
        concatenated = self.one.concatenate(self.two)
        assert np.all(concatenated.sample_names == np.array(['a', 'b', 'c', 'd', 'x', 'y', 'z']))
        assert np.all(concatenated.labels == np.array([1, 1, 0, 0, 1, 1, 0]))
        assert concatenated.data.equals(pd.DataFrame({'feature1': [10, 11, 12, 13, 30, 31, 32]}))

    def test_incompatible_vars(self):
        # additional paths, should not work
        self._init_data()
        self.two = TabularDataset(sample_names=self.two_sample_names, labels=self.two_labels, data=self.two_data,
                                  name="DS2", paths=self.two_sample_names)

        # should raise exception
        with pytest.raises(Exception):
            concatenated = self.one.concatenate(self.two)


class TestSubsetSelection:

    def _init_data(self):
        self.sample_names = np.array(['a', 'b', 'c', 'd'])
        self.labels = np.array([1, 1, 0, 0])
        self.data = pd.DataFrame({'feature1': [10, 11, 12, 13]})
        self.dataset = TabularDataset(sample_names=self.sample_names, labels=self.labels,
                                      data=self.data, name="DS1")

    def test_basic(self):
        self._init_data()
        new = self.dataset.subset_from_indices([2, 3])
        assert np.all(new.sample_names == np.array(['c', 'd'])), f"{new.sample_names} vs {np.array(['c', 'd'])}"
        assert np.all(new.labels == np.array([0, 0])), f"{new.labels} vs {np.array([0, 0])}"
        assert new.data.equals(pd.DataFrame({'feature1': [12, 13]}))

    def test_invalid_indices(self):
        # check invalid indices
        self._init_data()
        with pytest.raises(Exception):
            new = self.dataset.subset_from_indices(['a', 'b'])

        with pytest.raises(Exception):
            new = self.dataset.subset_from_indices([-1, 0])

        with pytest.raises(Exception):
            new = self.dataset.subset_from_indices([0, 4])