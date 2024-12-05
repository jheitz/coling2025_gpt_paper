from abc import abstractmethod
import torchaudio
import re, os
import torch
import datetime

from util.decorators import cache_to_file_decorator
from util.helpers import create_directory, hash_from_dict, dataset_name_to_url_part, python_to_json
from dataloader.dataset import TextDataset, AudioDataset
from preprocessing.preprocessor import Preprocessor

class Transcriber(Preprocessor):
    name = "Generic transcriber"

    def __init__(self, input_sample_rate=44100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_sample_rate = 44100  # sample rate of the ADReSS data set
        self.sample_rate = 16000  # 16Khz needed for most model
        self.resampler = torchaudio.transforms.Resample(self.input_sample_rate, self.sample_rate)
        self.current_date = str(datetime.date.today())
        self.current_dataset = ""  # Dataset currently transcribing
        self.transcriber_config = {}  # any config for the transcriber = different versions
        self.version = 1  # version of the transcriber's code -> if significant logic changes, change this
        print(f"Initializing audio transcriber {self.name}")


    def _resample(self, waveform, old_sample_rate):
        if old_sample_rate == self.input_sample_rate:
            resampler = self.resampler
        else:
            print(
                f"Unexpected input sample rate {old_sample_rate}. Expecting a default of {self.input_sample_rate}.",
                "If this default is wrong, change it in the class settings.",
                "For now, we create a new resampler for this sample")
            resampler = torchaudio.transforms.Resample(old_sample_rate, self.sample_rate)
        waveform_resampled = resampler(waveform)
        return waveform_resampled, self.sample_rate

    def _preprocess_waveform(self, waveform):
        assert len(waveform.shape) == 2, "Input waveform must be two-dimensional (channels x samples)"
        assert waveform.shape[0] == 1 or waveform.shape[0] == 2, "Input waveform must be mono or stereo"
        if waveform.shape[0] == 2:
            # stereo sound, take mean to get mono
            waveform = torch.mean(waveform, dim=0)
            return waveform
        elif waveform.shape[0] == 1:
            # mono
            return waveform.squeeze(0)

    @property
    def _version_config(self):
        # a short string representing the version and config hash.
        # this should make a particular version of the code and config of the transcriber unique
        # used for saving transcriptions to files and handling the caches
        config_hash = hash_from_dict(self.transcriber_config, 6)
        version_config = f"v{self.version}_{config_hash}"
        return version_config

    def _transcription_save_base_dir(self, is_raw=False):
        """
        Directory to save transcriptions to
        """
        if is_raw:
            dir = os.path.join(self.CONSTANTS.PREPROCESSED_DATA, "transcriptions", self.name, self._version_config,
                               "_raw_" + dataset_name_to_url_part(self.current_dataset))
        else:
            dir = os.path.join(self.CONSTANTS.PREPROCESSED_DATA, "transcriptions", self.name, self._version_config,
                               dataset_name_to_url_part(self.current_dataset))

        create_directory(dir)
        return dir

    def _initialize_transcription(self, dataset: AudioDataset):
        """
        Called once on each dataset: This writes information on the dataset, time point, config file, etc. to
        disk, so it's clear where the transcriptions are coming from
        """
        self.current_dataset = dataset.name
        base_dir = self._transcription_save_base_dir()
        info_file = os.path.join(base_dir, "info.txt")
        with open(info_file, 'a') as f:
            text = {
                'dataset': str(dataset),
                'datetime': str(datetime.datetime.now()),
                'class': self.__class__.__name__,
                'name': self.name,
                'version': self.version,
                'transcriber_config': python_to_json(self.transcriber_config)
            }
            f.write("\n".join([f"{key}: {text[key]}" for key in text]))
            f.write("\n\n")


    def _save_transcription(self, file_path, transcription, transcription_extended=None, sample_name=None):
        """
        Save transcription to file so it can be looked at
        transcription_extended can be a dict or similar with extended information
        """
        metadata = {"file": file_path, "transcriber": self.name}

        # train or test
        try:
            split = re.match(r".*/(train|test|test-dist)/.*", file_path).group(1)
            metadata["split"] = split
        except:
            pass

        # dementia or control
        try:
            group = re.match(r".*/(cd|cc|ad)/.*", file_path).group(1)
            metadata["group"] = group
        except:
            pass

        if sample_name is not None:
            identifier = sample_name
            metadata["identifier"] = sample_name
        else:
            print("Sample_name not passed for transcription, cannot save transcription.")
            return False

        def save_with_metadata():
            # save including some metadata
            content = "\n".join(f"{key}: {value}" for key, value in metadata.items())
            content += "\n\n"
            content += transcription
            if transcription_extended is not None:
                content += "\n\n"
                content += python_to_json(transcription_extended)

            transcriptions_dir = os.path.join(self._transcription_save_base_dir(is_raw=False),
                                              metadata['split'] if 'split' in metadata else "",
                                              metadata['group'] if 'group' in metadata else "")
            create_directory(transcriptions_dir)
            file_to_store = os.path.join(transcriptions_dir, f"{identifier}.txt")
            with open(file_to_store, 'w') as f:
                return f.write(content)

        def save_raw():
            # save raw content, for further analysis
            transcriptions_dir = os.path.join(self._transcription_save_base_dir(is_raw=True),
                                              metadata['split'] if 'split' in metadata else "",
                                              metadata['group'] if 'group' in metadata else "")
            create_directory(transcriptions_dir)
            file_to_store = os.path.join(transcriptions_dir, f"{identifier}.txt")
            with open(file_to_store, 'w') as f:
                return f.write(transcription)

        save_raw()
        save_with_metadata()


    @cache_to_file_decorator()
    @abstractmethod
    def transcribe_file(self, file_path: str, version_config: str) -> str:
        # version_config should identify the current relevant code version and potential config
        pass

    def transcribe_dataset(self, dataset: AudioDataset) -> TextDataset:
        print(f"Transcribing dataset {dataset.name} using audio transcriber {self.name}")

    def preprocess_dataset(self, dataset: AudioDataset) -> TextDataset:
        return self.transcribe_dataset(dataset)
