from ..base import BaseNode
from ..validation import validate_range


class WanDancerPadKeyframes(BaseNode):
    __slots__ = ("images", "audio", "segment_length", "segment_index")
    CATEGORY = "image/video"
    OUTPUTS = {"keyframes_sequence": 0, "keyframes_mask": 1, "audio_segment": 2}
    RETURN_TYPES = ("IMAGE", "MASK", "AUDIO")
    INPUT_SOCKET_TYPES = {
        "images": "IMAGE",
        "segment_length": "INT",
        "segment_index": "INT",
        "audio": "AUDIO",
    }

    def __init__(
        self,
        node_id: str,
        images: tuple[str, int],
        audio: tuple[str, int],
        segment_length: int | tuple[str, int] = 149,
        segment_index: int | tuple[str, int] = 0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.images = images
        self.audio = audio
        self.segment_length = segment_length
        self.segment_index = segment_index
        self.__validate()

    def __validate(self) -> None:
        validate_range("segment_length", self.segment_length, 1, 10_000)
        validate_range("segment_index", self.segment_index, 0, 100)


class WanDancerPadKeyframesList(BaseNode):
    __slots__ = ("images", "audio", "segment_length", "num_segments")
    CATEGORY = "image/video"
    OUTPUTS = {"keyframes_sequence": 0, "keyframes_mask": 1, "audio_segment": 2}
    RETURN_TYPES = ("IMAGE", "MASK", "AUDIO")
    INPUT_SOCKET_TYPES = {
        "images": "IMAGE",
        "segment_length": "INT",
        "num_segments": "INT",
        "audio": "AUDIO",
    }

    def __init__(
        self,
        node_id: str,
        images: tuple[str, int],
        audio: tuple[str, int],
        segment_length: int | tuple[str, int] = 149,
        num_segments: int | tuple[str, int] = 1,
    ) -> None:
        super().__init__(node_id=node_id)
        self.images = images
        self.audio = audio
        self.segment_length = segment_length
        self.num_segments = num_segments
        self.__validate()

    def __validate(self) -> None:
        validate_range("segment_length", self.segment_length, 1, 10_000)
        validate_range("num_segments", self.num_segments, 1, 100)
