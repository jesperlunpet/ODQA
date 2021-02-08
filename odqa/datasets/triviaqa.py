from datasets import load_dataset

from ..config import Config
from .dataset import TrainTestDataset

class TriviaQA(TrainTestDataset):
    def __init__(self, percent_of_data_to_keep=1):
        super().__init__()
        dataset = load_dataset(
            "trivia_qa",
            "unfiltered",
            writer_batch_size=200,
            cache_dir=Config.cache_dir,
        )

        dataset["train"] = self.select_part(
            dataset["train"], percent_of_data_to_keep
        )
        dataset["test"] = self.select_part(
            dataset["test"], percent_of_data_to_keep
        )
        dataset["validation"] = self.select_part(
            dataset["validation"], percent_of_data_to_keep
        )

        dataset = dataset.map(
            lambda data: {
                "question": data["question"],
                "answers": data["answer"]["aliases"]
            },
            remove_columns=dataset.column_names["train"],
            num_proc=Config.max_proc_to_use,
        )

        print(len(dataset["validation"]))

        dataset = self.filter(dataset)

        print(len(dataset["validation"]))

        dataset = dataset.map(
            lambda data: {
                "question": data["question"],
                "answers": self.check_answers(data["answers"])
            },
            remove_columns=dataset.column_names["train"],
            num_proc=Config.max_proc_to_use,
        )

        dataset = dataset.filter(
            lambda example: len(example["answers"]) > 0,
            num_proc=Config.max_proc_to_use
        )

        self.dataset = self.filter(dataset)

        print(len(self.dataset["validation"]))