from abc import ABC
import torch

class Evaluation(ABC):
    def score_batch(self, model_predictions, gold_references):
        self.metric.add_batch(predictions=model_predictions, references=gold_references)
        return self.metric.compute()

    def score(self, model_prediction, gold_reference):
        return self.metric.compute(predictions=model_prediction, references=gold_reference)