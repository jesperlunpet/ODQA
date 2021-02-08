import os

from datasets import load_dataset

from ..config import Config
from odqa.corpus.corpus import Corpus

class Squad2(Corpus):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        corpus = load_dataset(
            "squad",
            split='validation',
            cache_dir=Config.cache_dir,
        )

        print(corpus[0])

        corpus = corpus.map(
            lambda data: {
                "title": data['title'],
                "context": data['context']
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

        self.corpus.save_to_disk(os.path.join(Config.cache_dir, "dprsquad/"))

    def get_corpus(self):
        return self.corpus
        