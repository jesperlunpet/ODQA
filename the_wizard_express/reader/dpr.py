from transformers import (
    DPRReader,
    DPRReaderTokenizerFast,
)

from ..config import Config
from ..corpus.corpus import Corpus
from ..corpus.corpus import Dataset

from .reader import Reader

class DPR_reader(Reader):
    def __init__(self):
        self.r_encoder = DPRReader.from_pretrained("facebook/dpr-reader-single-nq-base").to(Config.device)
        self.r_tokenizer = DPRReaderTokenizerFast.from_pretrained("facebook/dpr-reader-single-nq-base")
