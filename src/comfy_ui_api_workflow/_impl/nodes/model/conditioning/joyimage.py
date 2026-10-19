from ...base import BaseNode


class TextEncodeJoyImageEdit(BaseNode):
    __slots__ = (
        "clip",
        "prompt",
        "vae",
        "images_image0",
        "images_image1",
        "images_image2",
        "images_image3",
        "images_image4",
        "images_image5",
    )
    CATEGORY = "model/conditioning/joyimage"
    OUTPUTS = {"CONDITIONING": 0}
    RETURN_TYPES = ("CONDITIONING",)
    INPUT_NAMES = {
        "images_image0": "images.image0",
        "images_image1": "images.image1",
        "images_image2": "images.image2",
        "images_image3": "images.image3",
        "images_image4": "images.image4",
        "images_image5": "images.image5",
    }
    INPUT_SOCKET_TYPES = {
        "clip": "CLIP",
        "prompt": "STRING",
        "vae": "VAE",
        "images_image0": "IMAGE",
        "images_image1": "IMAGE",
        "images_image2": "IMAGE",
        "images_image3": "IMAGE",
        "images_image4": "IMAGE",
        "images_image5": "IMAGE",
    }

    def __init__(
        self,
        node_id: str,
        clip: tuple[str, int],
        prompt: str | tuple[str, int],
        vae: tuple[str, int] | None = None,
        images_image0: tuple[str, int] | None = None,
        images_image1: tuple[str, int] | None = None,
        images_image2: tuple[str, int] | None = None,
        images_image3: tuple[str, int] | None = None,
        images_image4: tuple[str, int] | None = None,
        images_image5: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.clip = clip
        self.prompt = prompt
        if vae is not None:
            self.vae = vae
        if images_image0 is not None:
            self.images_image0 = images_image0
        if images_image1 is not None:
            self.images_image1 = images_image1
        if images_image2 is not None:
            self.images_image2 = images_image2
        if images_image3 is not None:
            self.images_image3 = images_image3
        if images_image4 is not None:
            self.images_image4 = images_image4
        if images_image5 is not None:
            self.images_image5 = images_image5
