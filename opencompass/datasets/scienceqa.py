import json
from glob import glob
import os.path as osp
import numpy as np
import pandas as pd

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class ScienceQADataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        dataset = DatasetDict()
        rng = np.random.default_rng(42)
        for split in ['train', 'validation', 'test']:
            raw_data = []
            filenames = glob(osp.join(path, f'data/{split}-*.parquet'))

            letters = ["A", "B", "C", "D"]
            for filename in filenames:
                data = pd.read_parquet(filename)
                for item in data.iloc:
                    choices = list(item)
                    choices += (4 - len(choices)) * ["This is not an option"]
                    raw_data.append({
                        'lecture': item['lecture'],
                        'input': item['question'],
                        'A': choices[0],
                        'B': choices[1],
                        'C': choices[2],
                        'D': choices[3],
                        'target': letters[item['answer']],
                    })

            dataset[split] = Dataset.from_list(raw_data)
        return dataset
