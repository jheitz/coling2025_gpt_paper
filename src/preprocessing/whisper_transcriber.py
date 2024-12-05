import numpy as np
import torch
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
import torchaudio

from dataloader.dataset import TextDataset, AudioDataset
from util.decorators import cache_to_file_decorator
from preprocessing.audio_transcriber import Transcriber


class WhisperTranscriber(Transcriber):

    def __init__(self, *args, **kwargs):
        self.name = "whisper-large"
        super().__init__(*args, **kwargs)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Whisper model
        try:
            self.whisper_model = self.config.config_whisper.model
        except AttributeError:
            self.whisper_model = 'whisper-large-v3'

        model_id = f"openai/{self.whisper_model}"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        #torch_dtype = torch.float32

        if self.whisper_model == 'whisper-large-v2':
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=model_id,
                chunk_length_s=30,
                device=device,
            )

        elif self.whisper_model == 'whisper-large-v3':
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id, torch_dtype=torch_dtype, use_safetensors=True, low_cpu_mem_usage=True,
            )
            model.to(device)

            processor = AutoProcessor.from_pretrained(model_id)

            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                max_new_tokens=128,
                chunk_length_s=30,
                batch_size=2,
                return_timestamps=True,
                torch_dtype=torch_dtype,
                device=device,
            )
        else:
            raise ValueError(f"Invalid whisper model: {self.whisper_model}")

        self.transcriber_config = {'whisper_model': self.whisper_model}  # any config for the transcriber = different versions
        self.version = 2  # version of the transcriber's code -> if significant logic changes, change this

        print(f"Using whisper_model {self.whisper_model} (model_id {model_id})")

    def _transcribe_file_whisper2(self, file_path: str):
        # load file
        waveform, sample_rate = torchaudio.load(file_path)

        # resample to target sample rate
        waveform, sample_rate = self._resample(waveform, sample_rate)

        # preprocess
        waveform = self._preprocess_waveform(waveform)

        transcription = self.pipe(np.array(waveform), batch_size=4)['text']
        return transcription

    def _transcribe_file_whisper3(self, file_path: str):
        return self.pipe(file_path, generate_kwargs={"language": "english"})['text']

    @cache_to_file_decorator()
    def transcribe_file(self, file_path: str, sample_name: str, version_config: str) -> str:
        print(f"Transcribing file {file_path}")

        if self.whisper_model == 'whisper-large-v2':
            transcription = self._transcribe_file_whisper2(file_path)
        elif self.whisper_model == 'whisper-large-v3':
            transcription = self._transcribe_file_whisper3(file_path)

        self._save_transcription(file_path, transcription, sample_name=sample_name)

        return transcription

    def transcribe_dataset(self, dataset: AudioDataset) -> TextDataset:
        # create new TextDataset with transcribed files
        super().transcribe_dataset(dataset)
        self._initialize_transcription(dataset)

        transcribed_data = np.array(
            [self.transcribe_file(file, sample_name=sample_name, version_config=self._version_config)
             for file, sample_name in zip(dataset.data, dataset.sample_names)]
        )

        config_without_preprocessors = {key: dataset.config[key] for key in dataset.config if key != 'preprocessors'}
        new_config = {
            'preprocessors': [*dataset.config['preprocessors'], self.name],
            **config_without_preprocessors
        }
        return TextDataset(data=transcribed_data, labels=np.array(dataset.labels),
                           sample_names=np.array(dataset.sample_names),
                           name=f"{dataset.name} - {self.name} transcribed", config=new_config)
