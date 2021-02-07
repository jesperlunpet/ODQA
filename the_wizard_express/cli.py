"""Console script for the_wizard_express."""
import sys
from multiprocessing import cpu_count
from datasets import load_from_disk
from transformers import BertConfig
import tensorflow_hub as hub
from official.nlp import bert
import official.nlp.bert.tokenization
import tensorflow.compat.v1 as tf
from tqdm import tqdm

import click
import torch
import os
import numpy as np

from .config import Config
from .datasets.triviaqa import TriviaQA
from .datasets.nq import NQ
from .datasets.squad import Squad
from .datasets.msmarco import MSMarco
from .datasets.wq import WQ
from .corpus.wikipedia import Wikipedia
from .corpus.squad import Squad2
from .reader.realm import REALMReader
from .embedder.realm import REALMEmbedder
from .embedder.dpr import DPR_embedder
from .reader.dpr import DPR_reader
from .retriever.retriever import Faiss, TFIDFRetriever
from .validation.accuracy import Accuracy
from .trainer.re_trainer import QATrainer
from .tester.tester import test_sparse_retriever, test_dense_retriever
from .tester.testrealm import test_realm_tokenizer, test_realm_similarity_reader, test_realm_similarity_retriever

from datasets import load_dataset
from elasticsearch import Elasticsearch

@click.group()
@click.option("--debug/--no-debug", default=False)
@click.option(
    "--max-proc", default=min(cpu_count() - 1, 8), show_default=True, type=int
)
@click.option(
    "--embedder", default="gs://realm-data/cc_news_pretrained/embedder", show_default=True, type=str
)
@click.option(
    "--reader", default="gs://realm-data/cc_news_pretrained/bert", show_default=True, type=str
)
@click.option(
    "--device", default=torch.device("cuda" if torch.cuda.is_available() else "cpu"), type=str
    # "--device", default=torch.device("cpu"), type=str
)

def main(debug, max_proc, embedder, reader, device):
    Config.debug = debug
    Config.proc_to_use = max_proc
    Config.embedder = embedder
    Config.reader = reader
    Config.device = device

    # Set environment variable such that correct Huggingface cache folder is correct
    # If home drive is used or sufficiently large, this is redundant
    os.environ["HF_MODULES_PATH"] = Config.cache_dir

# ========================================
# PREP of corpus for DPR, REALM and TF-IDF
# ========================================
@main.command()
def prepwikipediadpr():
    """Prepares Wikipedia with the DPR embedder"""
    dpr_embedder = DPR_embedder()

    corpus = Squad2(dpr_embedder.ctx_tokenizer).get_corpus()

    print("\nEmbedding corpus as dense context vector representations.")
    corpus_with_embeddings = dpr_embedder.embed_context(corpus)
    corpus_with_embeddings.save_to_disk(os.path.join(Config.cache_dir, "dprwiki/"))
    
    print("\nAdding Faiss index for efficient similarity search and clustering of dense vectors.")
    corpus_with_embeddings.add_faiss_index(column="embeddings")

    # Save index
    print(f"\nSaving the index to {os.path.join(Config.cache_dir, 'wikipedia_dpr.faiss')}")
    corpus_with_embeddings.save_faiss_index("embeddings", os.path.join(Config.cache_dir, "wikipedia_dpr.faiss"))

    return 0

@main.command()
def prepwikipediarealm():
    """Prepares Wikipedia with the REALM embedder"""
    realm = REALMEmbedder.load(BertConfig(intermediate_size=3072), Config.embedder).to(Config.device)

    corpus = Squad2(realm.tokenizer).get_corpus()

    print("\nEmbedding corpus as dense context vector representations.")
    corpus_with_embeddings = realm.embed_context(corpus)
    corpus_with_embeddings.save_to_disk(os.path.join(Config.cache_dir, "realmwiki/"))
    
    print("\nAdding Faiss index for efficient similarity search and clustering of dense vectors.")
    corpus_with_embeddings.add_faiss_index(column="embeddings")

    # Save index
    print(f"\nSaving the index to {os.path.join(Config.cache_dir, 'wikipedia_realm.faiss')}")
    corpus_with_embeddings.save_faiss_index("embeddings", os.path.join(Config.cache_dir, "wikipedia_realm.faiss"))

    return 0

@main.command()
def prepwikipediatfidf():
    """Prepares Wikipedia with the TF-IDF"""
    # Remember to start ElasticSearch
    # $ sudo systemctl start elasticsearch.service

    dpr_embedder = DPR_embedder() # Only used for tokenization into chunks

    corpus = Squad2(dpr_embedder.ctx_tokenizer).get_corpus()

    print("\nCombining corpus into sets of sentence and title.")
    corpus = corpus.map(lambda example: {'ctx': example["context"]})

    print("\nAdding TF-IDF BM25 index for efficient similarity search and clustering of sparse vectors.")
    corpus.add_elasticsearch_index("ctx", host="localhost", port="9200", es_index_name="wikipediasquad")

    return 0

# ========================================
#           PREP of datasets
# ========================================
@main.command()
def prepdatasets():
    """Downloads, filters and prepares the datasets into {question, answers} cuncks"""

    Squad().get_train_data()
    NQ().get_train_data()
    TriviaQA().get_train_data()

    return 0

# ========================================
#             Train the model
# ========================================
@main.command()
def train():

    dpr_reader = DPR_reader()
    dpr_embedder = DPR_embedder()

    # tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    # model = BertForQuestionAnswering.from_pretrained('bert-base-uncased').to(Config.device)

    trainer = QATrainer(dpr_reader.r_encoder,
        dpr_embedder.q_encoder,
        dpr_reader.r_tokenizer,
        dpr_embedder.q_tokenizer,
        Squad(dpr_reader.r_tokenizer).get_train_data(),
        Squad(dpr_reader.r_tokenizer).get_validation_data(),
        Wikipedia(dpr_embedder.ctx_tokenizer).get_corpus()
    )

    reader, retriever = trainer.train()

    torch.save(reader.state_dict(), os.path.join(Config.cache_dir, "trainedreader.pth.tar"))
    torch.save(retriever.state_dict(), os.path.join(Config.cache_dir, "trainedretriever.pth.tar"))

    return 0

# ========================================
#      Similarity test of tokenizers
# ========================================
@main.command()
def testtokenizer():
    vocabfile = "../Bert/assets/vocab.txt"
    test_realm_tokenizer(vocabfile)

# ========================================
#       Similarity test of models
# ========================================
@main.command()
def testreader():
    test_realm_similarity_reader()

    return 0

@main.command()
def testretriever():
    test_realm_similarity_retriever()

    return 0

# ========================================
#           Retriever tests
# ========================================
@main.command()
def testretrieverrealm():
    model = REALMEmbedder.load(BertConfig(intermediate_size=3072), Config.embedder).to(Config.device)

    dataset = Squad().get_validation_data()
    corpus = Squad2(model.tokenizer).get_corpus()

    print("\nEmbedding corpus as dense context vector representations.")
    # corpus_with_embeddings = corpus.map(lambda example: {'embeddings': model(**model.tokenizer(example["context"]["title"], example["context"]["sentence"], return_tensors="pt").to(Config.device))[0][0].cpu().detach().numpy()})

    print("\nAdding Faiss index for efficient similarity search and clustering of dense vectors.")
    # corpus_with_embeddings.add_faiss_index(column="embeddings")
    corpus.load_faiss_index("embeddings", os.path.join(Config.cache_dir, "wikipedia_realm.faiss"))

    test_dense_retriever(model, model.tokenizer, dataset, corpus)

    return 0

@main.command()
def testretrieverdpr():
    dpr_embedder = DPR_embedder()

    dataset = Squad().get_validation_data()
    corpus = Squad2(dpr_embedder.ctx_tokenizer).get_corpus()

    print("\nEmbedding corpus as dense context vector representations.")
    # corpus_with_embeddings = dpr_embedder.embed_context(corpus)
    
    print("\nAdding Faiss index for efficient similarity search and clustering of dense vectors.")
    corpus.load_faiss_index("embeddings", os.path.join(Config.cache_dir, "wikipedia_dpr.faiss"))
    # corpus_with_embeddings.add_faiss_index(column="embeddings")

    test_dense_retriever(dpr_embedder.q_encoder, dpr_embedder.q_tokenizer, dataset, corpus)

    return 0

@main.command()
def testtfidf():
    dpr_embedder = DPR_embedder()
    dataset = Squad().get_validation_data()
    corpus = Squad2(dpr_embedder.ctx_tokenizer).get_corpus()

    corpus.load_elasticsearch_index("ctx", host="localhost", port="9200", es_index_name="wikipediasquad")

    test_sparse_retriever(dataset, corpus)

    return 0

@main.command()
def testembed():
    dpr_embedder = DPR_embedder()

    corpus = Wikipedia(dpr_embedder.ctx_tokenizer).get_corpus()

    print("\nEmbedding corpus as dense context vector representations.")
    dpr_embedder.embed_context(corpus)

    return 0

@main.command()
def testlength():
    dataset = MSMarco().get_train_data()


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
