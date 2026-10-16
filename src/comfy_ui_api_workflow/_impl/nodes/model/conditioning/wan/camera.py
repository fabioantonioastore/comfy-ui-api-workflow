from typing import Literal, get_args

from ....base import BaseNode
from ....validation import validate_choice, validate_range


class WanCameraImageToVideo(BaseNode):
    __slots__ = (
        "positive",
        "negative",
        "vae",
        "width",
        "height",
        "length",
        "batch_size",
        "clip_vision_output",
        "start_image",
        "camera_conditions",
    )
    CATEGORY = "model/conditioning/wan/camera"
    OUTPUTS = {"positive": 0, "negative": 1, "latent": 2}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    INPUT_SOCKET_TYPES = {
        "positive": "CONDITIONING",
        "negative": "CONDITIONING",
        "vae": "VAE",
        "width": "INT",
        "height": "INT",
        "length": "INT",
        "batch_size": "INT",
        "clip_vision_output": "CLIP_VISION_OUTPUT",
        "start_image": "IMAGE",
        "camera_conditions": "WAN_CAMERA_EMBEDDING",
    }

    def __init__(
        self,
        node_id: str,
        positive: tuple[str, int],
        negative: tuple[str, int],
        vae: tuple[str, int],
        width: int | tuple[str, int] = 832,
        height: int | tuple[str, int] = 480,
        length: int | tuple[str, int] = 81,
        batch_size: int | tuple[str, int] = 1,
        clip_vision_output: tuple[str, int] | None = None,
        start_image: tuple[str, int] | None = None,
        camera_conditions: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.positive = positive
        self.negative = negative
        self.vae = vae
        self.width = width
        self.height = height
        self.length = length
        self.batch_size = batch_size
        if clip_vision_output is not None:
            self.clip_vision_output = clip_vision_output
        if start_image is not None:
            self.start_image = start_image
        if camera_conditions is not None:
            self.camera_conditions = camera_conditions
        self.__validate()

    def __validate(self) -> None:
        validate_range("width", self.width, 16, 16_384)
        validate_range("height", self.height, 16, 16_384)
        validate_range("length", self.length, 1, 16_384)
        validate_range("batch_size", self.batch_size, 1, 4096)


WanCameraEmbedding_Camera_Poses = Literal[
    "Static",
    "Pan Up",
    "Pan Down",
    "Pan Left",
    "Pan Right",
    "Zoom In",
    "Zoom Out",
    "Anti Clockwise (ACW)",
    "ClockWise (CW)",
]
WANCAMERAEMBEDDING_CAMERA_POSES = frozenset(get_args(WanCameraEmbedding_Camera_Poses))


class WanCameraEmbedding(BaseNode):
    __slots__ = (
        "camera_pose",
        "width",
        "height",
        "length",
        "speed",
        "fx",
        "fy",
        "cx",
        "cy",
    )
    CATEGORY = "model/conditioning/wan/camera"
    OUTPUTS = {"camera_embedding": 0, "width": 1, "height": 2, "length": 3}
    RETURN_TYPES = ("WAN_CAMERA_EMBEDDING", "INT", "INT", "INT")
    INPUT_SOCKET_TYPES = {
        "camera_pose": "COMBO",
        "width": "INT",
        "height": "INT",
        "length": "INT",
        "speed": "FLOAT",
        "fx": "FLOAT",
        "fy": "FLOAT",
        "cx": "FLOAT",
        "cy": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        camera_pose: WanCameraEmbedding_Camera_Poses | tuple[str, int] = "Static",
        width: int | tuple[str, int] = 832,
        height: int | tuple[str, int] = 480,
        length: int | tuple[str, int] = 81,
        speed: float | tuple[str, int] = 1.0,
        fx: float | tuple[str, int] = 0.5,
        fy: float | tuple[str, int] = 0.5,
        cx: float | tuple[str, int] = 0.5,
        cy: float | tuple[str, int] = 0.5,
    ) -> None:
        super().__init__(node_id=node_id)
        self.camera_pose = camera_pose
        self.width = width
        self.height = height
        self.length = length
        self.speed = speed
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.__validate()

    def __validate(self) -> None:
        validate_choice(
            "camera_pose", self.camera_pose, WANCAMERAEMBEDDING_CAMERA_POSES
        )
        validate_range("width", self.width, 16, 16_384)
        validate_range("height", self.height, 16, 16_384)
        validate_range("length", self.length, 1, 16_384)
        validate_range("speed", self.speed, 0, 10.0)
        validate_range("fx", self.fx, 0, 1)
        validate_range("fy", self.fy, 0, 1)
        validate_range("cx", self.cx, 0, 1)
        validate_range("cy", self.cy, 0, 1)
