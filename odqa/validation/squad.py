from datasets import load_metric

from ..config import Config
from .validation import Validation

class Squad(Validation):
    def __init__(self):
        super().__init__()
        self.metric = load_metric('squad', cache_dir=Config.cache_dir)