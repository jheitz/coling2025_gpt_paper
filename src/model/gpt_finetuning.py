from sklearn.model_selection import StratifiedKFold
import os
import numpy as np
import json
import pandas as pd
import random
from sklearn import metrics
import time, datetime
import hashlib
import asyncio
from openai import OpenAI

from model.base_model import BaseModel
from dataloader.dataset import TextDataset
from util.helpers import python_to_json, plot_roc, plot_roc_cv
from dataloader.custom_data_splitter import ADReSSTrainValTestSplitter


class GPTFineTuning(BaseModel):
    """
    GPT fine-tuning, based on https://platform.openai.com/docs/guides/fine-tuning/
    """
    def __init__(self, *args, **kwargs):
        super().__init__("GPTFineTuning", *args, **kwargs)
        self._train: TextDataset = None
        self._test: TextDataset = None

        self.batch_size = 8
        ## Some configuration

        # number of epochs to train
        try:
            self.num_epochs = self.config.num_epochs
        except AttributeError:
            self.num_epochs = 25  # best in hyperparameter testing
            self.num_epochs = None

        # Whether to use a validation set for validation after each epoch.
        # Not relevant for this model, assert it's false
        try:
            self.use_val_set = self.config.use_val_set
        except AttributeError:
            self.use_val_set = False
        assert not self.use_val_set

        # Use cross validation if cv_splits > 1. If cv_splits == 1, use predefined train and test sets
        try:
            self.cv_splits = self.config.cv_splits
        except AttributeError:
            self.cv_splits = 10

        # learning rate multiplier, see https://platform.openai.com/docs/guides/fine-tuning/advanced-usage
        try:
            self.learning_rate_multiplier = self.config.learning_rate_multiplier
        except AttributeError:
            self.learning_rate_multiplier = None

        # the gpt model
        try:
            self.gpt_model = self.config.config_gpt.model
        except AttributeError:
            self.gpt_model = "gpt-3.5-turbo-1106"  # 'gpt-4-1106-preview'

        print(f"Using num_epochs {self.num_epochs}, use_val_set {self.use_val_set}, "
              f"cv_splits {self.cv_splits}, "
              f"learning_rate_multplier {self.learning_rate_multiplier}, "
              f"gpt_model {self.gpt_model}")

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

    async def train(self, train_set=None, split_idx=None):
        """
        One training using train_set, if given, or self._train otherwise

        :param train_set: train set of this CV split
        :param split_idx: CV split Number
        :return:
        """
        assert train_set is not None, "Error: train_set is None"

        # save train set to file for openai
        train_set_df = []

        # Print some identifiers of items in sets
        # This is to be able to manually check that the splits are identical over different runs
        def subset_identifer(data_set):
            # get the sum of the tensor elements of the first three samples in each split
            return [row['data'][:10] for row in list(data_set)][:3]
        if self.use_val_set:
            raise NotImplementedError()
        else:
            print(f"Train set: {subset_identifer(train_set)}...")

        print(f"Fine-tuning for split {split_idx}")

        # prepare training samples
        training_messages = [self._get_messages(row, train=True) for row in train_set]
        random.Random(4).shuffle(training_messages)

        train_hash = hashlib.sha1(f'{"-".join([str(t) for t in training_messages])}'.encode("utf-8")).hexdigest()

        # uniquely identify one fine-tuned model. If finetune_id is already available in openai, it will not be trained again
        hash_config = {**vars(self.config), 'train_hash': train_hash}
        config_hash = hashlib.sha1(str(hash_config).encode("utf-8")).hexdigest()[:6]
        finetune_id = f"{config_hash}-split{split_idx if split_idx is not None else ''}-{self.cv_splits}"

        # check if finetuned already
        res = self.openai_client.models.list()
        matching_finetuned_models = [model.id for model in res.data if f"{finetune_id}" in str(model.id)]
        if len(matching_finetuned_models) > 0:
            print(f"Already finetuned: Model with ID {finetune_id} (config {hash_config}, split {split_idx}) already fine-tuned. Nothing to do...")
            return matching_finetuned_models[-1]

        else:
            print(f"Running fine-tuning with finetune_id {finetune_id} (config {hash_config}, split {split_idx})")

            # write fine-tuning (training) samples to jsonl files for gpt3
            path_jsonl_for_finetuning = os.path.join(self.CONSTANTS.CACHE_DIR,
                                                     f"train_{self.name}_{config_hash}_{split_idx}.jsonl")
            print(f"Writing to {path_jsonl_for_finetuning}")
            with open(path_jsonl_for_finetuning, "w") as f:
                f.writelines([json.dumps({'messages': row})+"\n" for row in training_messages])

            # upload training file
            training_file = self.openai_client.files.create(
              file=open(path_jsonl_for_finetuning, "rb"),
              purpose="fine-tune"
            )
            print("Uploaded training file", training_file)

            create_args = {
                "training_file": training_file.id,
                "model": self.gpt_model,
                "suffix": finetune_id,
                "hyperparameters": {
                    "n_epochs": self.num_epochs,
                    #"batch_size": 3,
                    "learning_rate_multiplier": self.learning_rate_multiplier,
                }
            }

            num_retries = 5
            for retry in range(num_retries):
                try:
                    response = self.openai_client.fine_tuning.jobs.create(**create_args)
                    break
                except Exception as e:
                    if retry + 1 == num_retries:
                        raise e
                    print("Error while creating the fine_tuning job:", e)
                    print("Waiting 10min and retrying")
                    time.sleep(60 * 10)

            job_id = response.id

            print(f'Split {split_idx} Fine-tunning model with jobID: {job_id}.')
            def format_time(seconds):
                return f"{int(np.floor(seconds / 60))}:{int(np.mod(seconds, 60)):02d}min"

            start_time = time.time()
            messages_length = 0
            fine_tuned_model = None
            while True:
                await asyncio.sleep(10)
                job = self.openai_client.fine_tuning.jobs.retrieve(fine_tuning_job_id=job_id)
                status = job.status
                fine_tuned_model = job.fine_tuned_model
                messages = [f"Split {split_idx}: {format_time(event.created_at - job.created_at)} ({status}): {event.message}" for
                            event in self.openai_client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id)]
                # print only new messages, i.e. the ones exceeding messages_length
                if len(messages) > messages_length:
                    print("\n".join(messages[0:len(messages)-messages_length]))
                messages_length = len(messages)
                if status in ["succeeded", "failed", "cancelled"]:
                    break

            print(f'Split {split_idx}: Finetune job {job_id} finished with status: {status}')
            current_time = time.time()
            elapsed_time = current_time - start_time
            print(f"Split {split_idx}: Run time: ", format_time(elapsed_time))
            print('Checking other finetune jobs in the subscription.')
            result = self.openai_client.fine_tuning.jobs.list()
            print(f'Found {len(result.data)} finetune jobs.')

        print(f"Finished training", f"split {split_idx}..." if split_idx is not None else "", f"Fine-tuned model: {fine_tuned_model}")

        return fine_tuned_model

    def test(self, fine_tuned_model=None, test_set=None, split_idx=None):
        assert test_set is not None, "Error, Test set is None"
        assert fine_tuned_model is not None, "Error: fine_tuned_model is None"
        print("Start testing", f"split {split_idx}..." if split_idx is not None else "")

        test_messages = [self._get_messages(row, train=False) for row in test_set]

        def get_prediction(messages):
            res = self.openai_client.chat.completions.create(model=fine_tuned_model,
                                          messages=messages,
                                          temperature=0,
                                          top_logprobs=3,
                                          logprobs=True)

            logprob_for_first_token = res.choices[0].logprobs.content[0]
            probab = np.exp(logprob_for_first_token.logprob)
            # get probability for dementia (i.e. fist token D)
            probab_of_1 = probab if logprob_for_first_token.token == 'D' else 1 - probab

            #print("first token", logprob_for_first_token.token, "logprob", logprob_for_first_token.logprob, "probab", probab, "prediction", probab_of_1)
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

    async def run_cv_split(self, train_set, test_set, split_idx):
        fine_tuned_model = await self.train(train_set, split_idx)
        computed_metrics = self.test(fine_tuned_model, test_set, split_idx=split_idx)
        return split_idx, computed_metrics

    async def async_train_test(self):
        test_metrics = {}
        all_predictions, all_labels, all_sample_names = [], [], []

        if self.cv_splits > 1:
            # We use cross validation, i.e. we first combine the train and test set and then randomly split
            # it into train / test using CV
            dataset = self._train.concatenate(self._test)

            kfold = StratifiedKFold(n_splits=self.cv_splits, shuffle=True, random_state=134)

            tasks = []
            for split_idx, (train_indices, test_indices) in enumerate(kfold.split(X=np.zeros(len(dataset)), y=dataset.labels)):
                train_set = dataset.subset_from_indices(train_indices)
                test_set = dataset.subset_from_indices(test_indices)
                tasks.append(self.run_cv_split(train_set, test_set, split_idx))

            computed_metrics_gathered = await asyncio.gather(*tasks)
            for split_idx, computed_metrics in computed_metrics_gathered:
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
            test_indices = np.where(np.isin(dataset.sample_names, data_splitter.get_mapping().query("split == 'Validation'").sample_name))
            train_indices = np.where(np.isin(dataset.sample_names, data_splitter.get_mapping().query("split != 'Validation'").sample_name))
            train_set = dataset.subset_from_indices(train_indices)
            test_set = dataset.subset_from_indices(test_indices)
            tasks = [self.run_cv_split(train_set, test_set, None)]
            computed_metrics_gathered = await asyncio.gather(*tasks)
            _, test_metrics = computed_metrics_gathered[0]
            all_predictions.append(test_metrics['predictions'])
            all_labels.append(test_metrics['labels'])
            all_sample_names.append(test_metrics['sample_names'])

        else:
            # No cross validation, use existing train / test split
            train_set = self._train
            test_set = self._test
            tasks = [self.run_cv_split(train_set, test_set, None)]
            computed_metrics_gathered = await asyncio.gather(*tasks)
            _, test_metrics = computed_metrics_gathered[0]
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

    def train_test(self):
        asyncio.run(self.async_train_test())