import sys
import argparse
from typing import Type
import os

# Set environment variable for api key, to be used globally (needed for import openai)
os.environ["OPENAI_API_KEY"] = open("../keys/openai-key.txt", "r").read()

from model.base_model import BaseModel
from model.gpt_finetuning import GPTFineTuning
from model.gpt_zeroshot import GPTZeroShot
from model.tree_based_classifier import RandomForest, GradientBoosting
from dataloader.dataloader import ADReSSDataLoader, ADReSSTranscriptDataLoader, ADReSSWithPITTDataLoader
from dataloader.dataset import TabularDataset
from preprocessing.whisper_transcriber import WhisperTranscriber
from preprocessing.google_speech_transcriber import GoogleSpeechTranscriber
from preprocessing.audio_cutter import AudioCutter
from preprocessing.linguistic_features import LinguisticFeatures
from config.config import Config
from config.run_parameters import RunParameters
from config.constants import Constants
from util.helpers import create_directory



def run_path(run_parameters: RunParameters, path_config: Config, CONSTANTS: Constants):
    dataloader = None
    try:
        debug = path_config.debug
    except AttributeError:
        debug = False

    if path_config.data == "ADReSS_manual_transcripts":
        dataloader = ADReSSTranscriptDataLoader(debug=debug, config=path_config)
    elif path_config.data == "ADReSS_audio":
        dataloader = ADReSSDataLoader(debug=debug, config=path_config)
    elif path_config.data == "ADReSS_with_PITT_audio":
        dataloader = ADReSSWithPITTDataLoader(debug=debug, config=path_config)
    else:
        raise ValueError("Invalid data config", path_config.data)

    preprocessors = []
    if path_config.preprocessors is not None:
        for p in path_config.preprocessors:
            if p == "ASR whisper":
                preprocessors.append(WhisperTranscriber(config=path_config, constants=CONSTANTS))
            elif p == "ASR google_speech":
                preprocessors.append(GoogleSpeechTranscriber(config=path_config, constants=CONSTANTS))
            elif p == 'PAR segmentation':
                preprocessors.append(AudioCutter(config=path_config, constants=CONSTANTS))
            elif p == "Linguistic Features":
                preprocessors.append(LinguisticFeatures(config=path_config, constants=CONSTANTS, run_parameters=run_parameters))
            else:
                raise ValueError("Invalid preprocessor:", p)

    data_train, data_test = dataloader.load_data()
    print("Train data before preprocessing: ", data_train)
    print("Test data before preprocessing: ", data_test)
    for p in preprocessors:
        print(f"Running preprocessor {p}...")
        data_train = p.preprocess_dataset(data_train)
        data_test = p.preprocess_dataset(data_test)
    print("Train data after preprocessing:", data_train)
    print("Test data after preprocessing:", data_test)

    return data_train, data_test

def run(run_parameters: RunParameters, config: Config, CONSTANTS: Constants):
    """Builds model, loads data, trains and evaluates"""

    print("Running pipeline...")
    print("Run Parameters:")
    print(run_parameters, end="\n\n")
    print("Config:")
    print(config, end="\n\n")

    # A pipeline is a dataset and some preprocessors applied on top
    # The result of each path is then concatenated and the model is applied to it
    try:
        assert isinstance(config.pipeline, list)
        pipeline = config.pipeline
    except AttributeError:
        # just pass all config attributes to the path
        pipeline = [config]

    data_train, data_test = None, None
    for i, path_config in enumerate(pipeline):
        print(f"\nRunning path {i+1} / {len(pipeline)}: {path_config}", end="\n\n")
        data_train_path, data_test_path = run_path(run_parameters, path_config, CONSTANTS)
        if i == 0:
            data_train, data_test = data_train_path, data_test_path
        else:
            print("Merging features of different paths")
            assert isinstance(data_train, TabularDataset), \
                "Can only merge TabularDatasets, make sure each path on pipeline produces one"
            data_train = data_train.merge(data_train_path)
            data_test = data_test.merge(data_test_path)

    print("Train data for model:", data_train)
    print("Test data for model:", data_test)

    ModelClass: Type[BaseModel]
    if config.model == "GPT Finetuning":
        ModelClass = GPTFineTuning
    elif config.model == "GPTZeroShot":
        ModelClass = GPTZeroShot
    elif config.model == "RandomForest":
        ModelClass = RandomForest
    elif config.model == "GradientBoosting":
        ModelClass = GradientBoosting
    elif config.model is None:
        # exit, this run is only for preprocessing (e.g. storing transcriptions), no model training
        print("No model specified, exiting...")
        sys.exit()
    else:
        raise ValueError("Invalid model", config.model)
    model: BaseModel = ModelClass(run_parameters=run_parameters, config=config, constants=CONSTANTS)
    model.set_train(data_train)
    model.set_test(data_test)
    model.prepare_data()
    model.train_test()




if __name__ == '__main__':
    # run parameters from command line arguments
    run_parameters = RunParameters.from_command_line_args()

    # configuration based on config file
    config = Config.from_yaml(run_parameters.config)

    # constants for e.g. directory paths
    CONSTANTS = Constants(local=run_parameters.local)

    # create results file
    create_directory(run_parameters.results_dir)

    run(run_parameters, config, CONSTANTS)
