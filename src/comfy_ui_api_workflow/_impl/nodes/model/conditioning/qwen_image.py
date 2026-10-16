from typing import Literal, get_args

from ...base import BaseNode
from ...validation import validate_choice


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
