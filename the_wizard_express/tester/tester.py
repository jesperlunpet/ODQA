import torch
import re

from unidecode import unidecode
from tqdm import tqdm

from the_wizard_express.config import Config

def test_dense_retriever(q_encoder, q_tokenizer, dataset, corpus, k=100):
    def normalize_string(text):
        return re.sub('\W|\d', '', unidecode(text.lower()))
    q_encoder.eval()
    # Tracking variables
    total_index_accuracy = [0] * k

    print("\nTesting...")
    for batch in tqdm(dataset):
        with torch.no_grad():
            question = q_encoder(**q_tokenizer(batch["question"], return_tensors="pt").to(Config.device))[0][0].cpu().detach().numpy()

        scores, retrieved_examples = corpus.get_nearest_examples('embeddings', question, k=k)

        print(batch["question"])
        print(retrieved_examples["context"][0])

        # Answer position accuracy
        correct = False
        for i in range(k):
            context = normalize_string(retrieved_examples["context"][i][1])
            for answer in batch["answers"]:
                answer = normalize_string(answer)
                if not correct:
                    correct = answer in context
                # print(str(i) + " - " + retrieved_examples["context"][i]["title"] + " - " + data["title"])
            if correct:
                total_index_accuracy[i] += 1

    for i in range(k):
        total_index_accuracy[i] = total_index_accuracy[i]/len(dataset)

    print("\nTraining complete!")
    print("  Index Accuracy for 1 answer: {0:.2f}".format(total_index_accuracy[0]))
    print("  Index Accuracy for 5 answers: {0:.2f}".format(total_index_accuracy[4]))
    print("  Index Accuracy for 10 answers: {0:.2f}".format(total_index_accuracy[9]))
    print("  Index Accuracy for 100 answers: {0:.2f}".format(total_index_accuracy[99]))

def test_sparse_retriever(dataset, corpus, k=100):
    def normalize_string(text):
        return re.sub('\W|\d', '', unidecode(text.lower()))

    # Tracking variables
    total_index_accuracy = [0] * k

    print("\nTesting...")
    for batch in tqdm(dataset):
        question = batch["question"]
        scores, retrieved_examples = corpus.get_nearest_examples('ctx', question, k=k)

        print(batch["question"])
        print(retrieved_examples["context"][0])

        # Answer position accuracy
        correct = False
        for i in range(k):
            context = normalize_string(retrieved_examples["context"][i][1])
            for answer in batch["answers"]:
                answer = normalize_string(answer)
                if not correct:
                    correct = answer in context
                # print(str(i) + " - " + retrieved_examples["context"][i]["title"] + " - " + data["title"])
            if correct:
                total_index_accuracy[i] += 1

    for i in range(k):
        total_index_accuracy[i] = total_index_accuracy[i]/len(dataset)

    print("\nTraining complete!")
    print("  Index Accuracy for 1 answer: {0:.2f}".format(total_index_accuracy[0]))
    print("  Index Accuracy for 5 answers: {0:.2f}".format(total_index_accuracy[4]))
    print("  Index Accuracy for 10 answers: {0:.2f}".format(total_index_accuracy[9]))
    print("  Index Accuracy for 100 answers: {0:.2f}".format(total_index_accuracy[99]))