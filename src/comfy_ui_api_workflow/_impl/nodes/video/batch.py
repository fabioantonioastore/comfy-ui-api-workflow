from ..base import BaseNode
from ..validation import validate_range


class ShuffleVideoDataset(BaseNode):
    __slots__ = ("videos", "seed")
    CATEGORY = "video/batch"
    OUTPUTS = {"videos": 0}
    RETURN_TYPES = ("VIDEO",)
    INPUT_SOCKET_TYPES = {"videos": "VIDEO", "seed": "INT"}

    def __init__(
        self, node_id: str, videos: tuple[str, int], seed: int | tuple[str, int] = 0
    ) -> None:
        super().__init__(node_id=node_id)
        self.videos = videos
        self.seed = seed
        self.__validate()

    def __validate(self) -> None:
        validate_range("seed", self.seed, 0, 18_446_744_073_709_551_615)


class ShuffleVideoTextDataset(BaseNode):
    __slots__ = ("videos", "texts", "seed")
    CATEGORY = "video/batch"
    OUTPUTS = {"videos": 0, "texts": 1}
    RETURN_TYPES = ("VIDEO", "STRING")
    INPUT_SOCKET_TYPES = {"videos": "VIDEO", "texts": "STRING", "seed": "INT"}

    def __init__(
        self,
        node_id: str,
        videos: tuple[str, int],
        texts: str | tuple[str, int],
        seed: int | tuple[str, int] = 0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.videos = videos
        self.texts = texts
        self.seed = seed
        self.__validate()

    def __validate(self) -> None:
        validate_range("seed", self.seed, 0, 18_446_744_073_709_551_615)
