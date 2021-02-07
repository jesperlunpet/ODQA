from abc import ABC

from ..config import Config

class TrainTestDataset(ABC):
    def __init__(self):
        self.dataset = None

    def select_part(self, data, percent):
        return data.select(range(round(len(data) * percent)))

    def get_train_data(self):
        return self.dataset["train"]

    def get_test_data(self):
        return self.dataset["test"]

    def get_validation_data(self):
        return self.dataset["validation"]

    def check_answers(self, answers):
        short_answers = []
        for answer in answers:
            if len(answer.split()) < 6:
                short_answers.append(answer)
        return short_answers
        
    def filter(self, dataset):
        return dataset.filter(
            lambda example: len(example["answers"]) > 0,
            num_proc=Config.max_proc_to_use
        )