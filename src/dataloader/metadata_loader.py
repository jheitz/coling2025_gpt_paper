import os
import pandas as pd
import numpy as np

from config.constants import Constants

class ADReSSParticipantMetadataLoader:
    def __init__(self, local=None, constants=None):
        self.name = "ADReSS Participant Metadata"
        if constants is None:
            self.CONSTANTS = Constants(local=local)
        else:
            self.CONSTANTS = constants
        print(f"Initializing metadata loader {self.name}")

        self.metadata = None

    def load_metadata(self):
        metadata_raw = [
            {'path': os.path.join(self.CONSTANTS.DATA_ADReSS_ROOT, "test", "meta_data_with_labels.csv"), 'label': None},
            {'path': os.path.join(self.CONSTANTS.DATA_ADReSS_ROOT, "train", "cd_meta_data.txt"), 'label': 1},
            {'path': os.path.join(self.CONSTANTS.DATA_ADReSS_ROOT, "train", "cc_meta_data.txt"), 'label': 0}]

        dfs = []
        for d in metadata_raw:
            path, label = d['path'], d['label']
            df = pd.read_csv(path, sep=";")
            if label is not None:
                df['Label'] = label
            df = df.rename(columns={c: c.strip() for c in df.columns}).rename(columns={'ID': 'sample_name'})
            dfs.append(df)

        self.metadata = pd.concat(dfs)
        self.metadata.sample_name = self.metadata.sample_name.str.strip()
        self.metadata.gender = np.where(self.metadata.gender.str.strip().isin(['female', "1", 1]), 1,
                                               np.where(self.metadata.gender.str.strip().isin(['male', "0", 0]),
                                                        0, self.metadata.gender))  # 1 female, 0 male

        self.metadata.mmse = self.metadata.mmse.astype('str').str.strip().replace('NA', ).dropna().astype(int)

        return self.metadata
