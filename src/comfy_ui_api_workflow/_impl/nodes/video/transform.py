from ..base import BaseNode
from ..validation import validate_range


class VideoTemporalCrop(BaseNode):
    __slots__ = ("video", "start_frame", "length")
    CATEGORY = "video/transform"
    OUTPUTS = {"video": 0}
    RETURN_TYPES = ("VIDEO",)
    INPUT_SOCKET_TYPES = {"video": "VIDEO", "start_frame": "INT", "length": "INT"}

    def __init__(
        self,
        node_id: str,
        video: tuple[str, int],
        start_frame: int | tuple[str, int] = 0,
        length: int | tuple[str, int] = 16,
    ) -> None:
        super().__init__(node_id=node_id)
        self.video = video
        self.start_frame = start_frame
        self.length = length
        self.__validate()

    def __validate(self) -> None:
        validate_range("start_frame", self.start_frame, 0, 99_999)
        validate_range("length", self.length, 1, 99_999)


class VideoRandomTemporalCrop(BaseNode):
    __slots__ = ("video", "length", "seed")
    CATEGORY = "video/transform"
    OUTPUTS = {"video": 0}
    RETURN_TYPES = ("VIDEO",)
    INPUT_SOCKET_TYPES = {"video": "VIDEO", "length": "INT", "seed": "INT"}

    def __init__(
        self,
        node_id: str,
        video: tuple[str, int],
        length: int | tuple[str, int] = 16,
        seed: int | tuple[str, int] = 0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.video = video
        self.length = length
        self.seed = seed
        self.__validate()

    def __validate(self) -> None:
        validate_range("length", self.length, 1, 99_999)
        validate_range("seed", self.seed, 0, 18_446_744_073_709_551_615)
