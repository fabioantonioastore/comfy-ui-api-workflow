from ...base import BaseNode


class TextEncodeBooguEdit(BaseNode):
    __slots__ = (
        "clip",
        "prompt",
        "negative_prompt",
        "vae",
        "images_image_1",
        "images_image_2",
        "images_image_3",
        "images_image_4",
        "images_image_5",
        "images_image_6",
        "images_image_7",
        "images_image_8",
        "images_image_9",
        "images_image_10",
        "images_image_11",
        "images_image_12",
        "images_image_13",
        "images_image_14",
        "images_image_15",
        "images_image_16",
    )
    CATEGORY = "model/conditioning/boogu"
    OUTPUTS = {"positive": 0, "negative": 1}
    RETURN_TYPES = (
        "CONDITIONING",
        "CONDITIONING",
    )
    INPUT_NAMES = {
        "images_image_1": "images.image_1",
        "images_image_2": "images.image_2",
        "images_image_3": "images.image_3",
        "images_image_4": "images.image_4",
        "images_image_5": "images.image_5",
        "images_image_6": "images.image_6",
        "images_image_7": "images.image_7",
        "images_image_8": "images.image_8",
        "images_image_9": "images.image_9",
        "images_image_10": "images.image_10",
        "images_image_11": "images.image_11",
        "images_image_12": "images.image_12",
        "images_image_13": "images.image_13",
        "images_image_14": "images.image_14",
        "images_image_15": "images.image_15",
        "images_image_16": "images.image_16",
    }
    INPUT_SOCKET_TYPES = {
        "clip": "CLIP",
        "prompt": "STRING",
        "negative_prompt": "STRING",
        "vae": "VAE",
        "images_image_1": "IMAGE",
        "images_image_2": "IMAGE",
        "images_image_3": "IMAGE",
        "images_image_4": "IMAGE",
        "images_image_5": "IMAGE",
        "images_image_6": "IMAGE",
        "images_image_7": "IMAGE",
        "images_image_8": "IMAGE",
        "images_image_9": "IMAGE",
        "images_image_10": "IMAGE",
        "images_image_11": "IMAGE",
        "images_image_12": "IMAGE",
        "images_image_13": "IMAGE",
        "images_image_14": "IMAGE",
        "images_image_15": "IMAGE",
        "images_image_16": "IMAGE",
    }

    def __init__(
        self,
        node_id: str,
        clip: tuple[str, int],
        prompt: str | tuple[str, int],
        negative_prompt: str | tuple[str, int],
        vae: tuple[str, int],
        images_image_1: tuple[str, int] | None = None,
        images_image_2: tuple[str, int] | None = None,
        images_image_3: tuple[str, int] | None = None,
        images_image_4: tuple[str, int] | None = None,
        images_image_5: tuple[str, int] | None = None,
        images_image_6: tuple[str, int] | None = None,
        images_image_7: tuple[str, int] | None = None,
        images_image_8: tuple[str, int] | None = None,
        images_image_9: tuple[str, int] | None = None,
        images_image_10: tuple[str, int] | None = None,
        images_image_11: tuple[str, int] | None = None,
        images_image_12: tuple[str, int] | None = None,
        images_image_13: tuple[str, int] | None = None,
        images_image_14: tuple[str, int] | None = None,
        images_image_15: tuple[str, int] | None = None,
        images_image_16: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.clip = clip
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.vae = vae
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
        if images_image_9 is not None:
            self.images_image_9 = images_image_9
        if images_image_10 is not None:
            self.images_image_10 = images_image_10
        if images_image_11 is not None:
            self.images_image_11 = images_image_11
        if images_image_12 is not None:
            self.images_image_12 = images_image_12
        if images_image_13 is not None:
            self.images_image_13 = images_image_13
        if images_image_14 is not None:
            self.images_image_14 = images_image_14
        if images_image_15 is not None:
            self.images_image_15 = images_image_15
        if images_image_16 is not None:
            self.images_image_16 = images_image_16
