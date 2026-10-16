from ...base import BaseNode
from ...validation import validate_range


class Trellis2Conditioning(BaseNode):
    __slots__ = ("clip_vision_model", "image")
    CATEGORY = "model/conditioning/trellis"
    OUTPUTS = {"positive": 0, "negative": 1}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    INPUT_SOCKET_TYPES = {"clip_vision_model": "CLIP_VISION", "image": "IMAGE"}

    def __init__(
        self, node_id: str, clip_vision_model: tuple[str, int], image: tuple[str, int]
    ) -> None:
        super().__init__(node_id=node_id)
        self.clip_vision_model = clip_vision_model
        self.image = image


class Pixal3DConditioning(BaseNode):
    __slots__ = ("clip_vision_model", "image", "camera_angle_x")
    CATEGORY = "model/conditioning/trellis"
    OUTPUTS = {"positive": 0, "negative": 1}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    INPUT_SOCKET_TYPES = {
        "clip_vision_model": "CLIP_VISION",
        "image": "IMAGE",
        "camera_angle_x": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        clip_vision_model: tuple[str, int],
        image: tuple[str, int],
        camera_angle_x: float | tuple[str, int] = 49.13,
    ) -> None:
        super().__init__(node_id=node_id)
        self.clip_vision_model = clip_vision_model
        self.image = image
        self.camera_angle_x = camera_angle_x
        self.__validate()

    def __validate(self) -> None:
        validate_range("camera_angle_x", self.camera_angle_x, 1.0, 170.0)


class Pixal3DMultiViewConditioning(BaseNode):
    __slots__ = ("clip_vision_model", "fov", "front", "left", "back", "right")
    CATEGORY = "model/conditioning/trellis"
    OUTPUTS = {"positive": 0, "negative": 1}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    INPUT_SOCKET_TYPES = {
        "clip_vision_model": "CLIP_VISION",
        "fov": "FLOAT",
        "front": "IMAGE",
        "left": "IMAGE",
        "back": "IMAGE",
        "right": "IMAGE",
    }

    def __init__(
        self,
        node_id: str,
        clip_vision_model: tuple[str, int],
        fov: float | tuple[str, int] = 20.0,
        front: tuple[str, int] | None = None,
        left: tuple[str, int] | None = None,
        back: tuple[str, int] | None = None,
        right: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.clip_vision_model = clip_vision_model
        self.fov = fov
        if front is not None:
            self.front = front
        if left is not None:
            self.left = left
        if back is not None:
            self.back = back
        if right is not None:
            self.right = right
        self.__validate()

    def __validate(self) -> None:
        validate_range("fov", self.fov, 1.0, 170.0)


class Trellis2ShapeStage(BaseNode):
    __slots__ = ("positive", "negative", "voxel")
    CATEGORY = "model/conditioning/trellis"
    OUTPUTS = {"positive": 0, "negative": 1, "LATENT": 2}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    INPUT_SOCKET_TYPES = {
        "positive": "CONDITIONING",
        "negative": "CONDITIONING",
        "voxel": "VOXEL",
    }

    def __init__(
        self,
        node_id: str,
        positive: tuple[str, int],
        negative: tuple[str, int],
        voxel: tuple[str, int],
    ) -> None:
        super().__init__(node_id=node_id)
        self.positive = positive
        self.negative = negative
        self.voxel = voxel


class Trellis2TextureStage(BaseNode):
    __slots__ = ("positive", "negative", "shape_latent")
    CATEGORY = "model/conditioning/trellis"
    OUTPUTS = {"positive": 0, "negative": 1, "LATENT": 2}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    INPUT_SOCKET_TYPES = {
        "positive": "CONDITIONING",
        "negative": "CONDITIONING",
        "shape_latent": "LATENT",
    }

    def __init__(
        self,
        node_id: str,
        positive: tuple[str, int],
        negative: tuple[str, int],
        shape_latent: tuple[str, int],
    ) -> None:
        super().__init__(node_id=node_id)
        self.positive = positive
        self.negative = negative
        self.shape_latent = shape_latent


class Trellis2UpsampleStage(BaseNode):
    __slots__ = ("positive", "negative", "shape_latent", "vae", "target_resolution")
    CATEGORY = "model/conditioning/trellis"
    OUTPUTS = {"positive": 0, "negative": 1, "LATENT": 2}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    INPUT_SOCKET_TYPES = {
        "positive": "CONDITIONING",
        "negative": "CONDITIONING",
        "shape_latent": "LATENT",
        "vae": "VAE",
        "target_resolution": "INT",
    }

    def __init__(
        self,
        node_id: str,
        positive: tuple[str, int],
        negative: tuple[str, int],
        shape_latent: tuple[str, int],
        vae: tuple[str, int],
        target_resolution: int | tuple[str, int] = 1024,
    ) -> None:
        super().__init__(node_id=node_id)
        self.positive = positive
        self.negative = negative
        self.shape_latent = shape_latent
        self.vae = vae
        self.target_resolution = target_resolution
        self.__validate()

    def __validate(self) -> None:
        validate_range("target_resolution", self.target_resolution, 1024, 2048)
