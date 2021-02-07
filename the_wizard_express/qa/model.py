from abc import ABC

from ..reader.reader import Reader
from ..retriever.retriever import Retriever


class Model(ABC):
    """
    Abstract class for all the final model
    """

    def __init__(self, retriever: Retriever, reader: Reader) -> None:
        self.retriever = retriever
        self.reader = reader
