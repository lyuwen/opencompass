import json
import os.path as osp
import numpy as np
import pandas as pd

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class SciQDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        dataset = DatasetDict()
        rng = np.random.default_rng(42)
        for split in ['train', 'test']:
            raw_data = []
            filename = osp.join(path, f'data/{split}-00000-of-00001.parquet')
            data = pd.read_parquet(filename)

            choices = ["A", "B", "C", "D"]
            for item in data.iloc:
                rng.shuffle(choices)
                raw_data.append({
                    'support': item['support'],
                    'input': item['question'],
                    choices[0]: item['distractor1'],
                    choices[1]: item['distractor2'],
                    choices[2]: item['distractor3'],
                    choices[3]: item['correct_answer'],
                    'target': choices[3],
                })

            dataset[split] = Dataset.from_list(raw_data)
        return dataset
