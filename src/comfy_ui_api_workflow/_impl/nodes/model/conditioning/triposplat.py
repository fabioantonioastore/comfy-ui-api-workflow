from ...base import BaseNode
from ...validation import validate_range


class TripoSplatPreprocessImage(BaseNode):
    __slots__ = ("image", "mask", "erode_radius", "size")
    CATEGORY = "model/conditioning/triposplat"
    OUTPUTS = {"image": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_SOCKET_TYPES = {
        "image": "IMAGE",
        "mask": "MASK",
        "erode_radius": "INT",
        "size": "INT",
    }

    def __init__(
        self,
        node_id: str,
        image: tuple[str, int],
        mask: tuple[str, int],
        erode_radius: int | tuple[str, int] = 1,
        size: int | tuple[str, int] = 1024,
    ) -> None:
        super().__init__(node_id=node_id)
        self.image = image
        self.mask = mask
        self.erode_radius = erode_radius
        self.size = size
        self.__validate()

    def __validate(self) -> None:
        validate_range("erode_radius", self.erode_radius, 0, 16)
        validate_range("size", self.size, 256, 4096)


class TripoSplatConditioning(BaseNode):
    __slots__ = ("clip_vision", "vae", "image")
    CATEGORY = "model/conditioning/triposplat"
    OUTPUTS = {"positive": 0, "negative": 1, "latent": 2}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    INPUT_SOCKET_TYPES = {"clip_vision": "CLIP_VISION", "vae": "VAE", "image": "IMAGE"}

    def __init__(
        self,
        node_id: str,
        clip_vision: tuple[str, int],
        vae: tuple[str, int],
        image: tuple[str, int],
    ) -> None:
        super().__init__(node_id=node_id)
        self.clip_vision = clip_vision
        self.vae = vae
        self.image = image
