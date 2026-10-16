from ...base import BaseNode
from ...validation import validate_range


class VOIDWarpedNoise(BaseNode):
    __slots__ = ("optical_flow", "video", "width", "height", "length", "batch_size")
    CATEGORY = "model/latent/void"
    OUTPUTS = {"warped_noise": 0}
    RETURN_TYPES = ("LATENT",)
    INPUT_SOCKET_TYPES = {
        "optical_flow": "OPTICAL_FLOW",
        "video": "IMAGE",
        "width": "INT",
        "height": "INT",
        "length": "INT",
        "batch_size": "INT",
    }

    def __init__(
        self,
        node_id: str,
        optical_flow: tuple[str, int],
        video: tuple[str, int],
        width: int | tuple[str, int] = 672,
        height: int | tuple[str, int] = 384,
        length: int | tuple[str, int] = 45,
        batch_size: int | tuple[str, int] = 1,
    ) -> None:
        super().__init__(node_id=node_id)
        self.optical_flow = optical_flow
        self.video = video
        self.width = width
        self.height = height
        self.length = length
        self.batch_size = batch_size
        self.__validate()

    def __validate(self) -> None:
        validate_range("width", self.width, 16, 16_384)
        validate_range("height", self.height, 16, 16_384)
        validate_range("length", self.length, 1, 16_384)
        validate_range("batch_size", self.batch_size, 1, 64)


class VOIDWarpedNoiseSource(BaseNode):
    __slots__ = ("warped_noise",)
    CATEGORY = "model/latent/void"
    OUTPUTS = {"NOISE": 0}
    RETURN_TYPES = ("NOISE",)
    INPUT_SOCKET_TYPES = {"warped_noise": "LATENT"}

    def __init__(self, node_id: str, warped_noise: tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.warped_noise = warped_noise
