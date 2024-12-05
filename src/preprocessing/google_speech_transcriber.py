import numpy as np
import os
import datetime
from google.cloud import storage
from google.api_core.exceptions import GoogleAPICallError
from google.api_core.exceptions import NotFound
from google.api_core import client_options
from google.cloud import speech_v2

from dataloader.dataset import TextDataset, AudioDataset
from util.decorators import cache_to_file_decorator
from preprocessing.audio_transcriber import Transcriber

class GoogleSpeechTranscriber(Transcriber):

    def __init__(self, *args, **kwargs):
        self.name = "google_speech"
        super().__init__(*args, **kwargs)

        self._set_auth()

        # Settings for GCS bucket and GCP project_id
        self.project_id = 'sodium-pager-388408'
        self.bucket_name = "jheitz_dementia"

        # Google Speech model
        try:
            self.model = self.config.config_google_speech.model
        except AttributeError:
            self.model = 'chirp'

        # Initialize the Google Cloud clients
        self.storage_client = storage.Client()

        self.recognizer_location = 'europe-west4' if self.model == 'chirp' else 'europe-west3'
        client_options_var = client_options.ClientOptions(
            api_endpoint=f"{self.recognizer_location}-speech.googleapis.com"
        )
        self.speech_client = speech_v2.SpeechClient(client_options=client_options_var)
        self.recognition_config = {
            'auto_decoding_config': speech_v2.types.cloud_speech.AutoDetectDecodingConfig(),
            'model': self.model,  # 'chirp' for lastest model, 'long' for default
            'language_codes': ["en-US"],
            'features': {
                'enable_automatic_punctuation': True,  # default false
                'max_alternatives': 1,  # default 1
                'enable_word_time_offsets': True,  # default False,
            }
        }

        # any config for the transcriber = different versions --> This is relevant for caching
        self.transcriber_config = {
            'recognition_config': self.recognition_config,
        }

        self.version = 1  # version of the transcriber's code -> if significant logic changes, change this

        self.current_date = str(datetime.date.today())

        print(f"Using model {self.model}")


    def _request_v2(self, file_metadata):
        """
        Request for the v2 API. Note that the recognizer needs to change depending on the location used
        """
        config = speech_v2.types.cloud_speech.RecognitionConfig(**self.recognition_config)
        recognizer = f"projects/{self.project_id}/locations/{self.recognizer_location}/recognizers/_"
        request = speech_v2.types.cloud_speech.BatchRecognizeRequest(
            recognizer=recognizer,
            config=config,
            files=[file_metadata],
            recognition_output_config=speech_v2.types.cloud_speech.RecognitionOutputConfig(
                inline_response_config=speech_v2.types.cloud_speech.InlineOutputConfig(),
            ),
        )
        return request


    def _set_auth(self):
        """
        Set Google Cloud credentials.
        This is done using Service Account Keys (RSA authentification for server-to-server)
        Check the documentation
        - https://cloud.google.com/docs/authentication/provide-credentials-adc#local-key
        - Create service account key: https://console.cloud.google.com/iam-admin/serviceaccounts/create?authuser=3&walkthrough_id=iam--create-service-account-keys&project=sodium-pager-388408&supportedpurview=project
        """
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../keys/gcloud-sodium-pager-388408-95aa25d97ab0.json")

    @cache_to_file_decorator(n_days=365)  # only recalculate after 1 year
    def transcribe_file(self, file_path: str, sample_name: str, version_config: str) -> str:

        try:
            # Upload the audio file to Google Cloud Storage
            folder_path = f"api_access/{self._version_config}/{self.current_date}/"
            basename = os.path.basename(file_path)
            blob_name = folder_path + basename
            blob = self.storage_client.bucket(self.bucket_name).blob(blob_name)
            blob.upload_from_filename(file_path)

            print(f"Transcribing file {file_path} for version_config {version_config}")

            # Get the GCS URI of the uploaded audio file
            gcs_uri = f"gs://{self.bucket_name}/{blob_name}"

            # Configure the speech recognition request for asynchronous recognition
            file_metadata = speech_v2.types.cloud_speech.BatchRecognizeFileMetadata(uri=gcs_uri)
            request = self._request_v2(file_metadata)
            operation = self.speech_client.batch_recognize(request=request)

            print(f"Waiting for operation to complete for {basename}...")
            operation_result = operation.result()

            # note that there's only one result (operation_result.results) because only one file is given
            operation_result = list(operation_result.results.values())[0]

            results_parsed = {
                'total_billed_time': str(operation_result.metadata.total_billed_duration),
                'output_error': str(operation_result.error),
                'results': [{
                    'alternatives': [{
                        'transcript': alt.transcript,
                        'confidence': alt.confidence,
                        'words': [{
                            'word': word.word,
                            'confidence': word.confidence,
                        } for word in alt.words]
                    } for alt in result.alternatives],
                    'language_code': result.language_code
                } for result in operation_result.transcript.results],
                'combined_transcript': "\n".join(
                    [result.alternatives[0].transcript for result in operation_result.transcript.results if len(result.alternatives) > 0]),
                'combined_confidences': [result.alternatives[0].confidence for result in operation_result.transcript.results if len(result.alternatives) > 0]
            }

            print(f"Transcription completed for {basename}")

        except GoogleAPICallError as e:
            print(f"An error occurred while processing {basename}: {str(e)}")
        except NotFound as e:
            print(f"Audio file not found: {str(e)}")
        except Exception as e:
            print(f"An unexpected error occurred while processing {basename}: {str(e)}")

        self._save_transcription(file_path, results_parsed['combined_transcript'], results_parsed, sample_name=sample_name)

        return results_parsed['combined_transcript']

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
