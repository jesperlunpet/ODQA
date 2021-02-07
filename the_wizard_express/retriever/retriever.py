from abc import ABC, abstractclassmethod

from ..corpus.corpus import Corpus
from ..config import Config

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

import torch
import pandas as pd

class Retriever(ABC):
    """
    Abstract class for all the retrievers
    """

    __slots__ = ["corpus"]

    def __init__(self, corpus: Corpus) -> None:
        self.corpus = corpus

    @abstractclassmethod
    def retrieve_docs(self, question: str) -> str:
        pass


class TFIDFRetriever(Retriever):
    def __init__(self):
        documentA = 'the man went out for a walk'
        documentB = 'the children sat around the fire'

        vectorizer = TfidfVectorizer()
        
        vectors = vectorizer.fit_transform([documentA, documentB])
        feature_names = vectorizer.get_feature_names()

        print(vectors)

        dense = vectors.todense()
        denselist = dense.tolist()

        df = pd.DataFrame(denselist, columns=feature_names)

    def retrieve_docs(self, question: str):
        pass


class Faiss(Retriever):
    def __init__(self, corpus):
        super().__init__(corpus)
        
    def __call__(self):
        return self.corpus.add_faiss_index(column='embeddings', device=torch.cuda.current_device() if Config.device else None)

    def retrieve_docs(self, question: str, k=100):
        return self.corpus.get_nearest_examples('embeddings', question, k=k)