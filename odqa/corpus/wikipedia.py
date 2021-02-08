import os

from json import dump
from os.path import dirname
from pathlib import Path
from typing import List

from datasets import load_dataset, load_from_disk

from odqa.config import Config
from odqa.corpus.corpus import Corpus

class Wikipedia(Corpus):
    def __init__(self, tokenizer):
        # try:
        #     self.corpus = load_from_disk(os.path.join(Config.cache_dir, "wikipedia/"))
        #     print("\nReusing previously saved the dataset")
        # except FileNotFoundError:
        print("\nDownloading the dataset")
        super().__init__(tokenizer)
        corpus = load_dataset(
            'wikipedia',
            '20200501.en',
            split='train[:10000]',
            cache_dir=Config.cache_dir
        )

        corpus = corpus.map(
            lambda data: {
                "title": data['title'],
                "context": data['text']
            },
            remove_columns=corpus.column_names,
            num_proc=Config.max_proc_to_use
        )

        self.corpus = corpus.map(
            self.generate_block_info,
            batched=True,
            remove_columns=corpus.column_names,
            num_proc=Config.max_proc_to_use
        )

        self.corpus.save_to_disk(os.path.join(Config.cache_dir, "wikipedia/"))

    def get_corpus(self):
        return self.corpus
        