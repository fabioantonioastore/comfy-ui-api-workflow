from ...base import BaseNode


class TextEncodeMingImageEdit(BaseNode):
    __slots__ = (
        "clip",
        "prompt",
        "images_image_1",
        "images_image_2",
        "images_image_3",
        "images_image_4",
        "images_image_5",
        "images_image_6",
        "images_image_7",
        "images_image_8",
        "vae",
    )
    CATEGORY = "model/conditioning/ming image"
    OUTPUTS = {"CONDITIONING": 0}
    RETURN_TYPES = ("CONDITIONING",)
    INPUT_NAMES = {
        "images_image_1": "images.image_1",
        "images_image_2": "images.image_2",
        "images_image_3": "images.image_3",
        "images_image_4": "images.image_4",
        "images_image_5": "images.image_5",
        "images_image_6": "images.image_6",
        "images_image_7": "images.image_7",
        "images_image_8": "images.image_8",
    }
    INPUT_SOCKET_TYPES = {
        "clip": "CLIP",
        "prompt": "STRING",
        "images_image_1": "IMAGE",
        "images_image_2": "IMAGE",
        "images_image_3": "IMAGE",
        "images_image_4": "IMAGE",
        "images_image_5": "IMAGE",
        "images_image_6": "IMAGE",
        "images_image_7": "IMAGE",
        "images_image_8": "IMAGE",
        "vae": "VAE",
    }

    def __init__(
        self,
        node_id: str,
        clip: tuple[str, int],
        prompt: str | tuple[str, int],
        images_image_1: tuple[str, int] | None = None,
        images_image_2: tuple[str, int] | None = None,
        images_image_3: tuple[str, int] | None = None,
        images_image_4: tuple[str, int] | None = None,
        images_image_5: tuple[str, int] | None = None,
        images_image_6: tuple[str, int] | None = None,
        images_image_7: tuple[str, int] | None = None,
        images_image_8: tuple[str, int] | None = None,
        vae: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.clip = clip
        self.prompt = prompt
        if images_image_1 is not None:
            self.images_image_1 = images_image_1
        if images_image_2 is not None:
            self.images_image_2 = images_image_2
        if images_image_3 is not None:
            self.images_image_3 = images_image_3
        if images_image_4 is not None:
            self.images_image_4 = images_image_4
        if images_image_5 is not None:
            self.images_image_5 = images_image_5
        if images_image_6 is not None:
            self.images_image_6 = images_image_6
        if images_image_7 is not None:
            self.images_image_7 = images_image_7
        if images_image_8 is not None:
            self.images_image_8 = images_image_8
        if vae is not None:
            self.vae = vae
