from abc import ABC
import torch

class Validation(ABC):
    def __init__(self):
        self.metric = None

    def score_batch(self, model_predictions, gold_references):
        self.metric.add_batch(predictions=model_predictions, references=gold_references)
        return self.metric.compute()

    def score(self, model_prediction, gold_reference):
        self.metric.add(prediction=model_prediction, reference=gold_reference)
        return self.metric.compute()