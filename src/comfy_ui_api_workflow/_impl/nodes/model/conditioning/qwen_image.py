from typing import Literal, get_args

from ...base import BaseNode
from ...validation import (
    validate_choice,
    validate_range,
)


class TextEncodeQwenImageEdit(BaseNode):
    __slots__ = ("clip", "prompt", "vae", "image")
    CATEGORY = "model/conditioning/qwen image"
    OUTPUTS = {"CONDITIONING": 0}
    RETURN_TYPES = ("CONDITIONING",)
    INPUT_SOCKET_TYPES = {
        "clip": "CLIP",
        "prompt": "STRING",
        "vae": "VAE",
        "image": "IMAGE",
    }

    def __init__(
        self,
        node_id: str,
        clip: tuple[str, int],
        prompt: str | tuple[str, int],
        vae: tuple[str, int] | None = None,
        image: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.clip = clip
        self.prompt = prompt
        if vae is not None:
            self.vae = vae
        if image is not None:
            self.image = image


class TextEncodeQwenImageEditPlus(BaseNode):
    __slots__ = ("clip", "prompt", "vae", "image1", "image2", "image3")
    CATEGORY = "model/conditioning/qwen image"
    OUTPUTS = {"CONDITIONING": 0}
    RETURN_TYPES = ("CONDITIONING",)
    INPUT_SOCKET_TYPES = {
        "clip": "CLIP",
        "prompt": "STRING",
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
        vae: tuple[str, int] | None = None,
        image1: tuple[str, int] | None = None,
        image2: tuple[str, int] | None = None,
        image3: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.clip = clip
        self.prompt = prompt
        if vae is not None:
            self.vae = vae
        if image1 is not None:
            self.image1 = image1
        if image2 is not None:
            self.image2 = image2
        if image3 is not None:
            self.image3 = image3


QwenImage21Cache_Devices = Literal["auto", "gpu", "cpu", "off"]
QwenImage21Cache_Dtypes = Literal["default", "int8", "int4"]
QWENIMAGE21CACHE_DEVICES = frozenset(get_args(QwenImage21Cache_Devices))
QWENIMAGE21CACHE_DTYPES = frozenset(get_args(QwenImage21Cache_Dtypes))


class QwenImage21Cache(BaseNode):
    __slots__ = ("model", "device", "dtype")
    CATEGORY = "model/conditioning/qwen image"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {"model": "MODEL", "device": "COMBO", "dtype": "COMBO"}

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        device: QwenImage21Cache_Devices | tuple[str, int] = "auto",
        dtype: QwenImage21Cache_Dtypes | tuple[str, int] = "default",
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.device = device
        self.dtype = dtype
        self.__validate()

    def __validate(self) -> None:
        validate_choice("device", self.device, QWENIMAGE21CACHE_DEVICES)
        validate_choice("dtype", self.dtype, QWENIMAGE21CACHE_DTYPES)


class TextEncodeQwenImage21(BaseNode):
    __slots__ = (
        "clip",
        "prompt",
        "negative_prompt",
        "resolution",
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
        "vae",
    )
    CATEGORY = "model/conditioning/qwen image"
    OUTPUTS = {"positive": 0, "negative": 1, "latent": 2}
    RETURN_TYPES = (
        "CONDITIONING",
        "CONDITIONING",
        "LATENT",
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
        "resolution": "INT",
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
        "vae": "VAE",
    }

    def __init__(
        self,
        node_id: str,
        clip: tuple[str, int],
        prompt: str | tuple[str, int],
        negative_prompt: str | tuple[str, int],
        resolution: int | tuple[str, int] = 1024,
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
        vae: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.clip = clip
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.resolution = resolution
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
        if vae is not None:
            self.vae = vae
        self.__validate()

    def __validate(self) -> None:
        validate_range("resolution", self.resolution, 0, 4096)
