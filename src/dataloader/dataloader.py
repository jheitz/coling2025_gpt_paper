from abc import abstractmethod
import os, re
import numpy as np
import pandas as pd
# import pylangacq
import time
from sklearn import model_selection

from dataloader.dataset import AudioDataset, TextDataset
from dataloader.cv_shuffler import ADReSSCrossValidationShuffler
from config.constants import Constants
from util.helpers import create_directory, get_sample_names_from_paths, hash_from_dict, dataset_name_to_url_part
from dataloader.chat_parser import ChatTranscriptParser
from dataloader.metadata_loader import ADReSSParticipantMetadataLoader


class DataLoader:
    def __init__(self, name, debug=False, local=None, constants=None, config=None):
        self.debug = debug
        self.name = name
        self.preprocessors = []
        if constants is None:
            self.CONSTANTS = Constants(local=local)
        else:
            self.CONSTANTS = constants
        self.config = config
        print(f"Initializing dataloader {self.name}")

        # Allow option to use only train data, no test data, useful for e.g. hyperparameter tuning, where test
        # set should not be touched
        try:
            self.only_train = self.config.config_data.only_train
        except (AttributeError, KeyError):
            self.only_train = False

    @abstractmethod
    def _load_train(self):
        pass

    @abstractmethod
    def _load_test(self):
        pass

    def load_data(self):
        print(f"Loading data using dataloader {self.name}")
        train = self._load_train()
        test = self._load_test()

        if self.only_train:
            print("Using only train data (dropping test), splitting into new train / test split")
            indices = np.arange(len(train))
            new_train_indices, new_test_indices = model_selection.train_test_split(indices, test_size=0.3,
                                                                                   shuffle=True, random_state=123,
                                                                                   stratify=train.labels)
            test = train.subset_from_indices(new_test_indices)
            train = train.subset_from_indices(new_train_indices)
            train_label_distribution = {label: np.sum(np.where(train.labels == label, 1, 0)) for label in set(train.labels)}
            test_label_distribution = {label: np.sum(np.where(test.labels == label, 1, 0)) for label in set(test.labels)}
            print(f"New train: {len(train)} (labels {train_label_distribution})")
            print(f"New test: {len(test)} (labels {test_label_distribution})")
        return train, test


class ADReSSDataLoader(DataLoader):

    def __init__(self, name="ADReSS audio", *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.preprocessors = []
        self.dir_train_ad = self.CONSTANTS.DATA_ADReSS_TRAIN_AD
        self.dir_train_cc = self.CONSTANTS.DATA_ADReSS_TRAIN_CONTROL
        self.dir_test = self.CONSTANTS.DATA_ADReSS_TEST

    def _load_train(self):
        paths = []
        labels = []
        for dir_path in [self.dir_train_ad, self.dir_train_cc]:
            label = 1 if os.path.basename(dir_path) == 'cd' else 0  # cc = control or cd = dementia
            for i, file_name in enumerate(os.listdir(dir_path)):
                if file_name.endswith('.wav'):
                    file_path = os.path.join(dir_path, file_name)
                    paths.append(file_path)
                    labels.append(label)
                if self.debug and i >= 1:
                    break

        dataset = AudioDataset(data=np.array(paths), labels=np.array(labels),
                               sample_names=get_sample_names_from_paths(np.array(paths)), name=f"{self.name} (train)",
                               config={'preprocessors': self.preprocessors, 'debug': self.debug,
                                       'cv_shuffler': ADReSSCrossValidationShuffler(constants=self.CONSTANTS, config=self.config)})
        return dataset

    def _load_test(self):
        paths = []
        for i, file_name in enumerate(os.listdir(self.dir_test)):
            if file_name.endswith('.wav'):
                file_path = os.path.join(self.dir_test, file_name)
                file_id = re.sub(r'\.wav', '', os.path.basename(file_path))
                paths.append((file_id, file_path))
            if self.debug and i > 1:
                break

        paths = pd.DataFrame(paths, columns=['ID', 'path'])

        assert os.path.exists(self.CONSTANTS.DATA_ADReSS_TEST_METADATA), "Test label / metadata file not available"

        metadata = pd.read_csv(self.CONSTANTS.DATA_ADReSS_TEST_METADATA, delimiter=';')
        metadata.columns = [c.strip() for c in metadata.columns]
        metadata['ID'] = metadata['ID'].str.strip()

        data = metadata.merge(paths, on="ID", how="inner")
        # labels = np.squeeze(metadata['Label'])

        dataset = AudioDataset(data=np.array(data['path']), labels=np.array(data['Label']),
                               sample_names=get_sample_names_from_paths(np.array(data['path'])), name=f"{self.name} (test)",
                               config={'preprocessors': self.preprocessors, 'debug': self.debug,
                                       'cv_shuffler': ADReSSCrossValidationShuffler(constants=self.CONSTANTS, config=self.config)})
        return dataset


class ADReSSWithPITTDataLoader(ADReSSDataLoader):
    """
    ADReSS dataset, but using the non-preprocessed corresponding files from the PITT corpus
    Also called ADReSS_RAW
    """

    def __init__(self, *args, **kwargs):
        name = "ADReSS PITT audio"
        super().__init__(name, *args, **kwargs)
        self.dir_train_ad = os.path.join(self.CONSTANTS.ADReSS_ORIGINAL_PITT_FILES, "train/cd")
        self.dir_train_cc = os.path.join(self.CONSTANTS.ADReSS_ORIGINAL_PITT_FILES, "train/cc")
        self.dir_test = os.path.join(self.CONSTANTS.ADReSS_ORIGINAL_PITT_FILES, "test")


class ADReSSTranscriptDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        # for this dataloader, we consider different configurations to load / preprocess CHAT format differently
        # in different situations, we sometimes want the configurations to be explicitly passed in the constructor,
        # while other times we want to use config given in the run yaml file

        config_settings = ['only_PAR', 'keep_pauses', 'keep_terminators', 'keep_unintelligable_speech',
                           'insert_pauses_automatic']

        name = "ADReSS manual transcripts"
        kwargs_without_explicit_config = {k: kwargs[k] for k in kwargs if k not in config_settings}
        super().__init__(name, *args, **kwargs_without_explicit_config)

        # back to the config: first, let's load the explicit parameters from kwargs
        transcript_config_dict = {k: kwargs[k] for k in kwargs if k in config_settings}

        # then, let's load the remaining ones from self.config.config_data
        if hasattr(self.config, 'config_data'):
            new_vals = {k: getattr(self.config.config_data, k) for k in vars(self.config.config_data)
                        if k in config_settings and k not in transcript_config_dict}
            transcript_config_dict = {**transcript_config_dict, **new_vals}

        # lastly, let's get default values for what is not defined yet
        self.transcript_config = {
            'only_PAR': transcript_config_dict.get('only_PAR', True), # only keep participant (PAR) parts if only_PAR==True, otherwise also include interviewer INV
            'keep_pauses': transcript_config_dict.get('keep_pauses', False),
            'keep_terminators': transcript_config_dict.get('keep_terminators', False),
            'keep_unintelligable_speech': transcript_config_dict.get('keep_unintelligable_speech', False),
            'insert_pauses_automatic': transcript_config_dict.get('insert_pauses_automatic', False),
        }
        self.transcript_config_hash = hash_from_dict(self.transcript_config, 6)

        self.method = None  # Choose method = 'pylangacq' for word extraction based on pylangacq library
        print("ADReSSTranscriptDataLoader chat_config:", self.transcript_config)
        self.preprocessors = []
        self.chat_transcript_parser = ChatTranscriptParser(config=self.transcript_config, constants=self.CONSTANTS)

        # self.label = the target variable to predict. "group" for AD / Ctrl binary classification, "mmse" for regression
        try:
            self.label = self.config.label
        except:
            self.label = "group"

        metadata_loader = ADReSSParticipantMetadataLoader(local=self.CONSTANTS.local, constants=self.CONSTANTS)
        self.metadata = metadata_loader.load_metadata()

    def _save_transcription(self, group, split, file_path, transcription):
        """ Save preprocessed transcript to file so it can be looked at """
        metadata = {"file": file_path, "group": group, "split": split}

        try:
            # from /path/to/S123.wav -> extract S123
            identifier = re.match(r".*/([a-zA-Z0-9_-]+.)\.(cha)", file_path).group(1)
            metadata["identifier"] = identifier
        except:
            print("Failed identifier of transcription, cannot save")
            return False

        def save_with_metadata():
            # save including some metadata
            content = "\n".join(f"{key}: {value}" for key, value in metadata.items())
            content += "\n\n"
            content += transcription
            transcriptions_base_dir = os.path.join(self.CONSTANTS.PREPROCESSED_DATA, "transcriptions", "manual",
                                                   dataset_name_to_url_part(self.name),
                                                   self.transcript_config_hash)
            transcriptions_dir = os.path.join(transcriptions_base_dir, split if split != "NA" else "", group)
            create_directory(transcriptions_dir)

            config_file = os.path.join(transcriptions_base_dir, "config.txt")
            with open(config_file, 'w') as f:
                f.write(str({**self.transcript_config, 'runtime': time.strftime("%Y-%m-%d %H:%M")}))

            file_to_store = os.path.join(transcriptions_dir, identifier + ".txt")
            with open(file_to_store, 'w') as f:
                return f.write(content)

        def save_raw():
            # save raw content, for further analysis
            transcriptions_base_dir = os.path.join(self.CONSTANTS.PREPROCESSED_DATA, "transcriptions", "manual",
                                                   "_raw_" + dataset_name_to_url_part(self.name),
                                                   self.transcript_config_hash)
            transcriptions_dir = os.path.join(transcriptions_base_dir, split if split != "NA" else "", group)
            create_directory(transcriptions_dir)

            file_to_store = os.path.join(transcriptions_dir, identifier + ".txt")
            with open(file_to_store, 'w') as f:
                return f.write(transcription)


        save_raw()
        save_with_metadata()


    def _load_train(self):
        transcripts = []
        paths = []
        labels = []
        disfluency_metrics = []
        for dir_path in [self.CONSTANTS.DATA_ADReSS_TRAIN_TRANSCRIPTS_AD,
                         self.CONSTANTS.DATA_ADReSS_TRAIN_TRANSCRIPTS_CONTROL]:
            label = 1 if os.path.basename(dir_path) == 'cd' else 0  # cc = control or cd = dementia
            for i, file_name in enumerate(os.listdir(dir_path)):
                if file_name.endswith('.cha'):
                    file_path = os.path.join(dir_path, file_name)
                    with open(file_path, 'r') as file:
                        transcript = file.read()
                        preprocessed = self.chat_transcript_parser.preprocess_transcript(transcript, file_path)
                        disfluency_metrics.append(self.chat_transcript_parser.extract_disfluency_metrics(transcript))
                        self._save_transcription(os.path.basename(dir_path), "train", file_path, preprocessed)
                    transcripts.append(preprocessed)
                    labels.append(label)
                    paths.append(file_path)
                if self.debug and i >= 1:
                    break

        dataset = TextDataset(data=np.array(transcripts), labels=np.array(labels),
                              sample_names=get_sample_names_from_paths(np.array(paths)), name=f"{self.name} (train)",
                              paths=np.array(paths), config={'preprocessors': self.preprocessors, 'debug': self.debug,
                                                   'cv_shuffler': ADReSSCrossValidationShuffler(constants=self.CONSTANTS, config=self.config),
                                                   'transcript_config': self.transcript_config,
                                                   'transcript_config_hash': self.transcript_config_hash},
                              # Add disfluency metric as own piece of data, to be used by linguistic feature extractor
                              disfluency_metrics=pd.DataFrame(disfluency_metrics))
        return dataset

    def _load_test(self):
        info = []
        for i, file_name in enumerate(os.listdir(self.CONSTANTS.DATA_ADReSS_TEST_TRANSCRIPTS)):
            if file_name.endswith('.cha'):
                file_path = os.path.join(self.CONSTANTS.DATA_ADReSS_TEST_TRANSCRIPTS, file_name)
                file_id = re.sub(r'\.cha', '', os.path.basename(file_path)).strip()
                with open(file_path, 'r') as file:
                    transcript = file.read()
                    preprocessed = self.chat_transcript_parser.preprocess_transcript(transcript, file_path)
                    disfluency_metrics = self.chat_transcript_parser.extract_disfluency_metrics(transcript)
                    self._save_transcription("NA", "test", file_path, preprocessed)
                info.append((file_id, preprocessed, file_path, disfluency_metrics))
            if self.debug and i > 1:
                break

        transcript = pd.DataFrame(info, columns=['sample_name', 'transcript', 'path', 'disfluency_metrics'])
        data = self.metadata.merge(transcript, on="sample_name")

        dataset = TextDataset(data=np.array(data['transcript']), labels=np.array(data['Label']),
                              sample_names=np.array(data['sample_name']), name=f"{self.name} (test)",
                              paths=np.array(data['path']),
                              config={'preprocessors': self.preprocessors, 'debug': self.debug,
                                      'cv_shuffler': ADReSSCrossValidationShuffler(constants=self.CONSTANTS, config=self.config),
                                      'transcript_config': self.transcript_config,
                                      'transcript_config_hash': self.transcript_config_hash},
                              # Add disfluency metric as own piece of data, to be used by linguistic feature extractor
                              disfluency_metrics=data['disfluency_metrics'].apply(pd.Series))
        return dataset
