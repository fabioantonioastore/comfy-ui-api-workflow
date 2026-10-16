from typing import Literal, get_args

from ...base import BaseNode
from ...validation import validate_choice, validate_range


class CLIPTextEncodeSDXLRefiner(BaseNode):
    __slots__ = ("text", "clip", "ascore", "width", "height")
    CATEGORY = "model/conditioning/stable diffusion"
    OUTPUTS = {"CONDITIONING": 0}
    RETURN_TYPES = ("CONDITIONING",)
    INPUT_SOCKET_TYPES = {
        "ascore": "FLOAT",
        "width": "INT",
        "height": "INT",
        "text": "STRING",
        "clip": "CLIP",
    }

    def __init__(
        self,
        node_id: str,
        text: str | tuple[str, int],
        clip: tuple[str, int],
        ascore: float | tuple[str, int] = 6.0,
        width: int | tuple[str, int] = 1024,
        height: int | tuple[str, int] = 1024,
    ) -> None:
        super().__init__(node_id=node_id)
        self.text = text
        self.clip = clip
        self.ascore = ascore
        self.width = width
        self.height = height
        self.__validate()

    def __validate(self) -> None:
        validate_range("ascore", self.ascore, 0.0, 1000.0)
        validate_range("width", self.width, 0, 16_384)
        validate_range("height", self.height, 0, 16_384)


class CLIPTextEncodeSDXL(BaseNode):
    __slots__ = (
        "clip",
        "text_g",
        "text_l",
        "width",
        "height",
        "crop_w",
        "crop_h",
        "target_width",
        "target_height",
    )
    CATEGORY = "model/conditioning/stable diffusion"
    OUTPUTS = {"CONDITIONING": 0}
    RETURN_TYPES = ("CONDITIONING",)
    INPUT_SOCKET_TYPES = {
        "clip": "CLIP",
        "width": "INT",
        "height": "INT",
        "crop_w": "INT",
        "crop_h": "INT",
        "target_width": "INT",
        "target_height": "INT",
        "text_g": "STRING",
        "text_l": "STRING",
    }

    def __init__(
        self,
        node_id: str,
        clip: tuple[str, int],
        text_g: str | tuple[str, int],
        text_l: str | tuple[str, int],
        width: int | tuple[str, int] = 1024,
        height: int | tuple[str, int] = 1024,
        crop_w: int | tuple[str, int] = 0,
        crop_h: int | tuple[str, int] = 0,
        target_width: int | tuple[str, int] = 1024,
        target_height: int | tuple[str, int] = 1024,
    ) -> None:
        super().__init__(node_id=node_id)
        self.clip = clip
        self.text_g = text_g
        self.text_l = text_l
        self.width = width
        self.height = height
        self.crop_w = crop_w
        self.crop_h = crop_h
        self.target_width = target_width
        self.target_height = target_height
        self.__validate()

    def __validate(self) -> None:
        validate_range("width", self.width, 0, 16_384)
        validate_range("height", self.height, 0, 16_384)
        validate_range("crop_w", self.crop_w, 0, 16_384)
        validate_range("crop_h", self.crop_h, 0, 16_384)
        validate_range("target_width", self.target_width, 0, 16_384)
        validate_range("target_height", self.target_height, 0, 16_384)


CLIPTextEncodeSD3_Empty_Paddings = Literal["none", "empty_prompt"]
CLIPTEXTENCODESD3_EMPTY_PADDINGS = frozenset(get_args(CLIPTextEncodeSD3_Empty_Paddings))


class CLIPTextEncodeSD3(BaseNode):
    __slots__ = ("clip", "clip_l", "clip_g", "t5xxl", "empty_padding")
    CATEGORY = "model/conditioning/stable diffusion"
    OUTPUTS = {"CONDITIONING": 0}
    RETURN_TYPES = ("CONDITIONING",)
    INPUT_SOCKET_TYPES = {
        "clip": "CLIP",
        "clip_l": "STRING",
        "clip_g": "STRING",
        "t5xxl": "STRING",
        "empty_padding": "COMBO",
    }

    def __init__(
        self,
        node_id: str,
        clip: tuple[str, int],
        clip_l: str | tuple[str, int],
        clip_g: str | tuple[str, int],
        t5xxl: str | tuple[str, int],
        empty_padding: CLIPTextEncodeSD3_Empty_Paddings | tuple[str, int],
    ) -> None:
        super().__init__(node_id=node_id)
        self.clip = clip
        self.clip_l = clip_l
        self.clip_g = clip_g
        self.t5xxl = t5xxl
        self.empty_padding = empty_padding
        self.__validate()

    def __validate(self) -> None:
        validate_choice(
            "empty_padding", self.empty_padding, CLIPTEXTENCODESD3_EMPTY_PADDINGS
        )
