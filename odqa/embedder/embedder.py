from abc import ABC, abstractclassmethod

from ..corpus.corpus import Corpus
from ..config import Config

class Embedder(ABC):
    """
    Abstract class for all the embedders
    """