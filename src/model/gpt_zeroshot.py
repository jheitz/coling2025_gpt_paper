from sklearn.model_selection import StratifiedKFold
import os
import numpy as np
import pandas as pd
from sklearn import metrics
from openai import OpenAI

from model.base_model import BaseModel
from dataloader.dataset import TextDataset
from util.helpers import python_to_json, plot_roc, plot_roc_cv
from dataloader.custom_data_splitter import ADReSSTrainValTestSplitter


class GPTZeroShot(BaseModel):
    """
    GPT zero-shot predition
    """
    def __init__(self, *args, **kwargs):
        super().__init__("GPT3", *args, **kwargs)
        self._train: TextDataset = None
        self._test: TextDataset = None

        # Use cross validation if cv_splits > 1. If cv_splits == 1, use predefined train and test sets
        try:
            self.cv_splits = self.config.cv_splits
        except AttributeError:
            self.cv_splits = 10

        # the gpt model
        try:
            self.gpt_model = self.config.config_gpt.model
        except AttributeError:
            self.gpt_model = 'gpt-4-1106-preview'

        print(f"Using gpt_model {self.gpt_model}")

        self.openai_key = open("../keys/openai-key.txt", "r").read()
        self.openai_client = OpenAI(api_key=self.openai_key)

    def set_train(self, dataset: TextDataset):
        self._train = dataset

    def set_test(self, dataset: TextDataset):
        self._test = dataset

    def prepare_data(self):
        # Prepare for openai fine-tuning
        def preprocess_function(ds: TextDataset):
            ds.tokens = np.array([text + "\n\n###\n\n" for text in ds.data])
            return ds

        self._train = preprocess_function(self._train)
        if self._test:
            self._test = preprocess_function(self._test)

    def _get_messages(self, row, train=True):
        text = """
Please decide whether the following transcription (enclosed with ```) comes from a person with Alzheimer's dementia or a healthy person. Just answer "Dementia" or "Healthy", without explanation.

```
""" + row['data'] + """
```
""".strip()
        answer = "Dementia" if row['labels'] == 1 else "Healthy"
        messages = [{"role": "user", 'content': text.strip()}]
        if train:
            messages.append({'role': 'assistant', 'content': answer})
        return messages

    def train(self):
        # zero-shot prediction, there's nothing to train
        pass

    def test(self, test_set=None, split_idx=None):
        assert test_set is not None, "Error, Test set is None"
        print("Start testing", f"split {split_idx}..." if split_idx is not None else "")

        test_messages = [self._get_messages(row, train=False) for row in test_set]

        def get_prediction(messages):
            res = self.openai_client.chat.completions.create(model=self.gpt_model,
                                                             messages=messages,
                                                             temperature=0,
                                                             top_logprobs=3,
                                                             logprobs=True)

            logprob_for_first_token = res.choices[0].logprobs.content[0]
            probab = np.exp(logprob_for_first_token.logprob)
            # get probability for dementia (i.e. fist token D)
            probab_of_1 = probab if logprob_for_first_token.token == 'D' else 1 - probab

            return probab_of_1

        predictions = []
        for messages in test_messages:
            predictions.append(get_prediction(messages))
        predictions = np.array(predictions)

        computed_metrics = {
            "accuracy": metrics.accuracy_score(test_set.labels, np.round(predictions)),
            "roc_auc": metrics.roc_auc_score(test_set.labels, predictions),
            "predictions": predictions,
            "labels": test_set.labels,
            "sample_names": test_set.sample_names
        }

        print(f"Test metrics", f"split {split_idx}..." if split_idx is not None else "", ":",
              python_to_json(computed_metrics))

        return computed_metrics

    def run_cv_split(self, train_set, test_set, split_idx):
        computed_metrics = self.test(test_set, split_idx=split_idx)
        return split_idx, computed_metrics

    def train_test(self):
        test_metrics = {}
        all_predictions, all_labels, all_sample_names = [], [], []

        if self.cv_splits > 1:
            # We use cross validation, i.e. we first combine the train and test set and then randomly split
            # it into train / test using CV
            dataset = self._train.concatenate(self._test)

            kfold = StratifiedKFold(n_splits=self.cv_splits, shuffle=True, random_state=134)

            computed_metrics = []
            for split_idx, (train_indices, test_indices) in enumerate(
                    kfold.split(X=np.zeros(len(dataset)), y=dataset.labels)):
                train_set = dataset.subset_from_indices(train_indices)
                test_set = dataset.subset_from_indices(test_indices)
                computed_metrics.append(self.run_cv_split(train_set, test_set, split_idx))

            for split_idx, computed_metrics in computed_metrics:
                test_metrics[f'split_{split_idx}'] = computed_metrics
                all_predictions.append(computed_metrics['predictions'])
                all_labels.append(computed_metrics['labels'])
                all_sample_names.append(computed_metrics['sample_names'])

            print("CV test metrics:", test_metrics)
            print("CV test metrics aggregated", {})

        elif self.cv_splits == -1:
            # Train on part of train and test, validate on validation set
            # This is used to compare to some GPT experiment I'm doing, where I don't want to touch the
            # test set yet
            data_splitter = ADReSSTrainValTestSplitter(self.CONSTANTS)
            dataset = self._train.concatenate(self._test)
            test_indices = np.where(
                np.isin(dataset.sample_names, data_splitter.get_mapping().query("split == 'Validation'").sample_name))
            train_indices = np.where(
                np.isin(dataset.sample_names, data_splitter.get_mapping().query("split != 'Validation'").sample_name))
            train_set = dataset.subset_from_indices(train_indices)
            test_set = dataset.subset_from_indices(test_indices)
            computed_metrics = [self.run_cv_split(train_set, test_set, None)]
            _, test_metrics = computed_metrics[0]
            all_predictions.append(test_metrics['predictions'])
            all_labels.append(test_metrics['labels'])
            all_sample_names.append(test_metrics['sample_names'])

        else:
            # No cross validation, use existing train / test split
            train_set = self._train
            test_set = self._test
            computed_metrics = [self.run_cv_split(train_set, test_set, None)]
            _, test_metrics = computed_metrics[0]
            all_predictions.append(test_metrics['predictions'])
            all_labels.append(test_metrics['labels'])
            all_sample_names.append(test_metrics['sample_names'])

            print("Test metrics:", test_metrics)

        # write metrics to file (as json)
        with open(os.path.join(self.run_parameters.results_dir, "metrics.txt"), "w") as file:
            file.write(python_to_json(test_metrics))

        # write predictions and labels to file
        with open(os.path.join(self.run_parameters.results_dir, "predictions.txt"), "w") as file:
            file.write(python_to_json(all_predictions))
        with open(os.path.join(self.run_parameters.results_dir, "labels.txt"), "w") as file:
            file.write(python_to_json(all_labels))
        with open(os.path.join(self.run_parameters.results_dir, "sample_names.txt"), "w") as file:
            file.write(python_to_json(all_sample_names))

        # write roc curve to results dir
        plot_roc(os.path.join(self.run_parameters.results_dir, "roc.png"), all_predictions, all_labels)
        plot_roc_cv(os.path.join(self.run_parameters.results_dir, "roc_cv.png"), all_predictions, all_labels)