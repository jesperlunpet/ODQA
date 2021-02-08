import sys
import nltk
import os
from ..config import Config
from abc import ABC, abstractclassmethod
from typing import List, Union

from datasets import Dataset

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

class Corpus(ABC):
    """
    Abstract class for all the corpus
    """
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        nltk.download("punkt", download_dir=Config.cache_dir)
        self._sentence_splitter =  nltk.data.load(
            os.path.join(Config.cache_dir, "tokenizers/punkt/english.pickle"))
        self.counter = 0

    @abstractclassmethod
    def get_corpus(self):
        raise NotImplementedError
    
    def chunk_examples(self, examples):
        chunks = []
        for sentence in examples['context']:
            chunks += [sentence[i:i + 50] for i in range(0, len(sentence), 50)]
        return {'context': chunks}
    
    def split_sentences(self, title, text, max_block_length=100):
        """Split sentences in each block from text."""
        title_length = len(self._tokenizer.tokenize(title))
        current_token_count = 0
        current_block_sentences = []
        final_list = []
        for sentence in self._sentence_splitter.tokenize(text):
            num_tokens = len(self._tokenizer.tokenize(sentence))
            # Too long sequence
            if num_tokens > max_block_length:
                continue
            # Hypothetical sequence [CLS] <title> [SEP] <current> <next> [SEP].
            hypothetical_length = 3 + title_length + current_token_count + num_tokens
            if hypothetical_length <= max_block_length:
                current_token_count += num_tokens
                current_block_sentences.append(sentence)
            else:
                final_list.append([title, " ".join(current_block_sentences)])
                current_token_count = num_tokens
                current_block_sentences = []
                current_block_sentences.append(sentence)
        if current_block_sentences:
            final_list.append([title, " ".join(current_block_sentences)])
        return final_list

    def generate_block_info(self, title_text):
        sentences = []
        for index, context in enumerate(title_text["context"]):
            sentences += self.split_sentences(title_text["title"][index], context)
        return {"context": sentences}
        # return {"context": self.split_sentences(title_text["title"], title_text["context"])}

    def index_answers(self, sentences, answer):
        ol = []
        for sentence in sentences:
            il = []
            for i in range(len(self._tokenizer.tokenize(sentence["sentence"])) - len(self._tokenizer.tokenize(answer)) + 1):
                if self._tokenizer.tokenize(sentence["sentence"])[i:i+len(self._tokenizer.tokenize(answer))] == self._tokenizer.tokenize(answer):
                    il.append([i,i+len(self._tokenizer.tokenize(answer))])
            if il == []:
                il.append([-1,-1])
            ol.append(il)

        return 0

    def generate_block_info2(self, title_text):
        return self.split_sentences(title_text["title"], title_text["context"])
        