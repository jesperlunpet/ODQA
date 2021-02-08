from transformers import (
    DPRContextEncoderTokenizerFast,
    DPRContextEncoder,
    DPRQuestionEncoder, 
    DPRQuestionEncoderTokenizerFast
)

from ..config import Config
from ..corpus.corpus import Corpus
from ..corpus.corpus import Dataset

from .embedder import Embedder

class DPR_embedder(Embedder):
    def __init__(self):
        self.ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-multiset-base").to(Config.device)
        self.q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-multiset-base").to(Config.device)
        self.ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")
        self.q_tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained("facebook/dpr-question_encoder-multiset-base")

    def embed_context(self, corpus: Corpus) -> Corpus:
        return corpus.map(lambda example: {'embeddings': self.ctx_encoder(**self.q_tokenizer(example["context"], return_tensors="pt", padding=True, truncation=True).to(Config.device))[0].cpu().detach().numpy()}, batched=True, batch_size=3)

    def embed_questions(self, dataset: Dataset) -> Dataset:
        return dataset.map(lambda example: {'embeddings': self.q_encoder(**self.q_tokenizer(example["question"], return_tensors="pt", padding=True, truncation=True).to(Config.device))[0][0].cpu().detach().numpy()}, batched=True, batch_size=40)

    def embed_question(self, question: str, context: str):
        return self.q_encoder(**self.q_tokenizer(question, context, return_tensors="pt").to(Config.device))[0][0].cpu().detach().numpy()