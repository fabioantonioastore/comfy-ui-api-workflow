from ...base import BaseNode
from ...validation import validate_bool


class TextEncodeZImageOmni(BaseNode):
    __slots__ = (
        "clip",
        "prompt",
        "auto_resize_images",
        "image_encoder",
        "vae",
        "image1",
        "image2",
        "image3",
    )
    CATEGORY = "model/conditioning/z-image"
    OUTPUTS = {"CONDITIONING": 0}
    RETURN_TYPES = ("CONDITIONING",)
    INPUT_SOCKET_TYPES = {
        "clip": "CLIP",
        "prompt": "STRING",
        "auto_resize_images": "BOOLEAN",
        "image_encoder": "CLIP_VISION",
        "vae": "VAE",
        "image1": "IMAGE",
        "image2": "IMAGE",
        "image3": "IMAGE",
    }

    def __init__(
        self,
        node_id: str,
        clip: tuple[str, int],
        prompt: str | tuple[str, int],
        auto_resize_images: bool | tuple[str, int] = True,
        image_encoder: tuple[str, int] | None = None,
        vae: tuple[str, int] | None = None,
        image1: tuple[str, int] | None = None,
        image2: tuple[str, int] | None = None,
        image3: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.clip = clip
        self.prompt = prompt
        self.auto_resize_images = auto_resize_images
        if image_encoder is not None:
            self.image_encoder = image_encoder
        if vae is not None:
            self.vae = vae
        if image1 is not None:
            self.image1 = image1
        if image2 is not None:
            self.image2 = image2
        if image3 is not None:
            self.image3 = image3
        self.__validate()

    def __validate(self) -> None:
        validate_bool("auto_resize_images", self.auto_resize_images)
