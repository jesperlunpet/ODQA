from os import getenv, path


class _config:
    """
    A class with all the configuration used accross the projects
    """

    def __init__(self):
        self._debug = False
        self._proc = None
        self._realmembedder = None
        self._realmreader = None
        self._device = None

    cache_dir = getenv("TRANSFORMERS_CACHE", path.join(path.realpath("."), ".cache"))
    """Config.cache_dir The cache directory to store models and other temporary files"""

    @property
    def debug(self) -> bool:
        return self._debug

    @debug.setter
    def debug(self, value: bool) -> None:
        self._debug = value

    @property
    def max_proc_to_use(self) -> int:
        return self._proc

    @max_proc_to_use.setter
    def max_proc_to_use(self, value: int) -> None:
        self._proc = value

    @property
    def embedder(self) -> int:
        return self._realmembedder
        
    @embedder.setter
    def embedder(self, value: str) -> None:
        self._realmembedder = value

    @property
    def reader(self) -> int:
        return self._realmreader
        
    @reader.setter
    def reader(self, value: str) -> None:
        self._realmreader = value

    @property
    def device(self) -> int:
        return self._device
        
    @device.setter
    def device(self, value: str) -> None:
        self._device = value

Config = _config()
