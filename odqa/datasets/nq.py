from functools import reduce
from datasets import load_dataset

from ..config import Config
from .dataset import TrainTestDataset

class NQ(TrainTestDataset):
    def __init__(self, percent_of_data_to_keep=1):
        super().__init__()
        dataset = load_dataset(
            "natural_questions",
            cache_dir=Config.cache_dir
        )

        dataset["train"] = self.select_part(
            dataset["train"], percent_of_data_to_keep
        )

        dataset["validation"] = self.select_part(
            dataset["validation"], percent_of_data_to_keep
        )

        dataset = dataset.map(
            lambda data: {
                "question": data["question"]["text"],
                "answers": reduce(lambda x,y: x+y,[answers["text"] for answers in data["annotations"]["short_answers"]])
            },
            remove_columns=dataset.column_names["train"],
            num_proc=Config.max_proc_to_use
        )

        print(len(dataset["train"]))

        dataset = self.filter(dataset)

        print(len(dataset["train"]))

        dataset = dataset.map(
            lambda data: {
                "question": data["question"],
                "answers": self.check_answers(data["answers"])
            },
            remove_columns=dataset.column_names["train"],
            num_proc=Config.max_proc_to_use
        )

        self.dataset = self.filter(dataset)

        print(len(self.dataset["train"]))