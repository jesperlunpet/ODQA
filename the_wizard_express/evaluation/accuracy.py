from datasets import load_metric

from ..config import Config
from .evaluation import Evaluation

class Accuracy(Evaluation):
    def __init__(self):
        super().__init__()
        self.metric = load_metric('accuracy', cache_dir=Config.cache_dir)