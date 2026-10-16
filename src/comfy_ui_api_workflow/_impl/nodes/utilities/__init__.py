from typing import Any, Literal, get_args

from ..base import BaseNode
from ..validation import validate_choice, validate_not_empty, validate_range
from ...lazy import submodules_of


class CreateBoundingBoxes(BaseNode):
    __slots__ = (
        "width",
        "height",
        "editor_state",
        "background",
        "bboxes",
        "last_incoming",
    )
    CATEGORY = "utilities"
    OUTPUTS = {"preview": 0, "bboxes": 1, "elements": 2}
    RETURN_TYPES = ("IMAGE", "BOUNDING_BOX", "ARRAY")
    INPUT_SOCKET_TYPES = {
        "width": "INT",
        "height": "INT",
        "editor_state": "BOUNDING_BOXES",
        "background": "IMAGE",
        "bboxes": "BOUNDING_BOX,ARRAY,STRING",
        "last_incoming": "BOUNDING_BOXES",
    }

    def __init__(
        self,
        node_id: str,
        width: int | tuple[str, int] = 1024,
        height: int | tuple[str, int] = 1024,
        editor_state: Any = None,
        background: tuple[str, int] | None = None,
        bboxes: tuple[str, int] | None = None,
        last_incoming: Any = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.width = width
        self.height = height
        self.editor_state = [] if editor_state is None else editor_state
        if background is not None:
            self.background = background
        if bboxes is not None:
            self.bboxes = bboxes
        self.last_incoming = [] if last_incoming is None else last_incoming
        self.__validate()

    def __validate(self) -> None:
        validate_range("width", self.width, 64, 16_384)
        validate_range("height", self.height, 64, 16_384)


class PreviewAny(BaseNode):
    __slots__ = ("source",)
    CATEGORY = "utilities"
    OUTPUTS = {"STRING": 0}
    RETURN_TYPES = ("STRING",)
    INPUT_SOCKET_TYPES = {"source": "*"}

    def __init__(self, node_id: str, source: tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.source = source


class CustomCombo(BaseNode):
    __slots__ = ("choice",)
    CATEGORY = "utilities"
    OUTPUTS = {"STRING": 0, "INDEX": 1}
    RETURN_TYPES = ("STRING", "INT")
    INPUT_SOCKET_TYPES = {"choice": "COMBO"}

    def __init__(self, node_id: str, choice: str | tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.choice = choice
        self.__validate()

    def __validate(self) -> None:
        validate_not_empty("choice", self.choice)


ResolutionSelector_Aspect_Ratios = Literal[
    "1:1 (Square)",
    "2:3 (Portrait Photo)",
    "3:2 (Photo)",
    "3:4 (Portrait Standard)",
    "4:3 (Standard)",
    "9:16 (Portrait Widescreen)",
    "16:9 (Widescreen)",
    "21:9 (Ultrawide)",
]
RESOLUTIONSELECTOR_ASPECT_RATIOS = frozenset(get_args(ResolutionSelector_Aspect_Ratios))


class ResolutionSelector(BaseNode):
    __slots__ = ("aspect_ratio", "megapixels", "multiple", "preview")
    CATEGORY = "utilities"
    OUTPUTS = {"width": 0, "height": 1}
    RETURN_TYPES = ("INT", "INT")
    INPUT_SOCKET_TYPES = {
        "aspect_ratio": "COMBO",
        "megapixels": "FLOAT",
        "multiple": "INT",
        "preview": "RESOLUTION_PREVIEW",
    }

    def __init__(
        self,
        node_id: str,
        aspect_ratio: (
            ResolutionSelector_Aspect_Ratios | tuple[str, int]
        ) = "1:1 (Square)",
        megapixels: float | tuple[str, int] = 1.0,
        multiple: int | tuple[str, int] = 8,
        preview: Any = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.aspect_ratio = aspect_ratio
        self.megapixels = megapixels
        self.multiple = multiple
        if preview is not None:
            self.preview = preview
        self.__validate()

    def __validate(self) -> None:
        validate_choice(
            "aspect_ratio", self.aspect_ratio, RESOLUTIONSELECTOR_ASPECT_RATIOS
        )
        validate_range("megapixels", self.megapixels, 0.1, 16.0)
        validate_range("multiple", self.multiple, 8, 128)


class ColorToRGBInt(BaseNode):
    __slots__ = ("color",)
    CATEGORY = "utilities"
    OUTPUTS = {"rgb_int": 0, "hex": 1, "alpha": 2}
    RETURN_TYPES = ("INT", "COLOR", "FLOAT")
    INPUT_SOCKET_TYPES = {"color": "COLOR"}

    def __init__(self, node_id: str, color: Any = "#ffffff") -> None:
        super().__init__(node_id=node_id)
        self.color = color


class GetItemFromList(BaseNode):
    __slots__ = ("list", "index")
    CATEGORY = "utilities"
    OUTPUTS = {"*": 0}
    RETURN_TYPES = ("*",)
    INPUT_SOCKET_TYPES = {"list": "*", "index": "INT"}

    def __init__(
        self, node_id: str, list: tuple[str, int], index: int | tuple[str, int] = 0
    ) -> None:
        super().__init__(node_id=node_id)
        self.list = list
        self.index = index
        self.__validate()

    def __validate(self) -> None:
        validate_range("index", self.index)


class ComfyNumberConvert(BaseNode):
    __slots__ = ("value",)
    CATEGORY = "utilities"
    OUTPUTS = {"FLOAT": 0, "INT": 1}
    RETURN_TYPES = ("FLOAT", "INT")
    INPUT_SOCKET_TYPES = {"value": "INT,FLOAT,STRING,BOOLEAN"}

    def __init__(self, node_id: str, value: tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.value = value


class CurveEditor(BaseNode):
    __slots__ = ("curve", "histogram")
    CATEGORY = "utilities"
    OUTPUTS = {"curve": 0}
    RETURN_TYPES = ("CURVE",)
    INPUT_SOCKET_TYPES = {"curve": "CURVE", "histogram": "HISTOGRAM"}

    def __init__(
        self, node_id: str, curve: Any = None, histogram: tuple[str, int] | None = None
    ) -> None:
        super().__init__(node_id=node_id)
        self.curve = (
            {"points": [[0.0, 0.0], [1.0, 1.0]], "interpolation": "monotone_cubic"}
            if curve is None
            else curve
        )
        if histogram is not None:
            self.histogram = histogram


class ImageHistogram(BaseNode):
    __slots__ = ("image",)
    CATEGORY = "utilities"
    OUTPUTS = {"rgb": 0, "luminance": 1, "red": 2, "green": 3, "blue": 4}
    RETURN_TYPES = ("HISTOGRAM", "HISTOGRAM", "HISTOGRAM", "HISTOGRAM", "HISTOGRAM")
    INPUT_SOCKET_TYPES = {"image": "IMAGE"}

    def __init__(self, node_id: str, image: tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.image = image


class SeedNode(BaseNode):
    __slots__ = ("seed",)
    CATEGORY = "utilities"
    OUTPUTS = {"seed": 0}
    RETURN_TYPES = ("INT",)
    INPUT_SOCKET_TYPES = {"seed": "INT"}

    def __init__(self, node_id: str, seed: int | tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.seed = seed
        self.__validate()

    def __validate(self) -> None:
        validate_range("seed", self.seed, 0, 9_223_372_036_854_775_807)


# ComfyUI grows the autogrow input "values" from a to z. Each letter is an
# attribute sent as "values.<letter>", and values_a is the one required.
class ComfyMathExpression(BaseNode):
    __slots__ = (
        "expression",
        "values_a",
        "values_b",
        "values_c",
        "values_d",
        "values_e",
        "values_f",
        "values_g",
        "values_h",
        "values_i",
        "values_j",
        "values_k",
        "values_l",
        "values_m",
        "values_n",
        "values_o",
        "values_p",
        "values_q",
        "values_r",
        "values_s",
        "values_t",
        "values_u",
        "values_v",
        "values_w",
        "values_x",
        "values_y",
        "values_z",
    )
    CATEGORY = "utilities"
    OUTPUTS = {"FLOAT": 0, "INT": 1, "BOOL": 2}
    RETURN_TYPES = ("FLOAT", "INT", "BOOLEAN")
    INPUT_NAMES = {
        "values_a": "values.a",
        "values_b": "values.b",
        "values_c": "values.c",
        "values_d": "values.d",
        "values_e": "values.e",
        "values_f": "values.f",
        "values_g": "values.g",
        "values_h": "values.h",
        "values_i": "values.i",
        "values_j": "values.j",
        "values_k": "values.k",
        "values_l": "values.l",
        "values_m": "values.m",
        "values_n": "values.n",
        "values_o": "values.o",
        "values_p": "values.p",
        "values_q": "values.q",
        "values_r": "values.r",
        "values_s": "values.s",
        "values_t": "values.t",
        "values_u": "values.u",
        "values_v": "values.v",
        "values_w": "values.w",
        "values_x": "values.x",
        "values_y": "values.y",
        "values_z": "values.z",
    }
    INPUT_SOCKET_TYPES = {
        "expression": "STRING",
        "values_a": "FLOAT,INT,BOOLEAN",
        "values_b": "FLOAT,INT,BOOLEAN",
        "values_c": "FLOAT,INT,BOOLEAN",
        "values_d": "FLOAT,INT,BOOLEAN",
        "values_e": "FLOAT,INT,BOOLEAN",
        "values_f": "FLOAT,INT,BOOLEAN",
        "values_g": "FLOAT,INT,BOOLEAN",
        "values_h": "FLOAT,INT,BOOLEAN",
        "values_i": "FLOAT,INT,BOOLEAN",
        "values_j": "FLOAT,INT,BOOLEAN",
        "values_k": "FLOAT,INT,BOOLEAN",
        "values_l": "FLOAT,INT,BOOLEAN",
        "values_m": "FLOAT,INT,BOOLEAN",
        "values_n": "FLOAT,INT,BOOLEAN",
        "values_o": "FLOAT,INT,BOOLEAN",
        "values_p": "FLOAT,INT,BOOLEAN",
        "values_q": "FLOAT,INT,BOOLEAN",
        "values_r": "FLOAT,INT,BOOLEAN",
        "values_s": "FLOAT,INT,BOOLEAN",
        "values_t": "FLOAT,INT,BOOLEAN",
        "values_u": "FLOAT,INT,BOOLEAN",
        "values_v": "FLOAT,INT,BOOLEAN",
        "values_w": "FLOAT,INT,BOOLEAN",
        "values_x": "FLOAT,INT,BOOLEAN",
        "values_y": "FLOAT,INT,BOOLEAN",
        "values_z": "FLOAT,INT,BOOLEAN",
    }

    def __init__(
        self,
        node_id: str,
        values_a: tuple[str, int],
        expression: str | tuple[str, int] = "a + b",
        values_b: tuple[str, int] | None = None,
        values_c: tuple[str, int] | None = None,
        values_d: tuple[str, int] | None = None,
        values_e: tuple[str, int] | None = None,
        values_f: tuple[str, int] | None = None,
        values_g: tuple[str, int] | None = None,
        values_h: tuple[str, int] | None = None,
        values_i: tuple[str, int] | None = None,
        values_j: tuple[str, int] | None = None,
        values_k: tuple[str, int] | None = None,
        values_l: tuple[str, int] | None = None,
        values_m: tuple[str, int] | None = None,
        values_n: tuple[str, int] | None = None,
        values_o: tuple[str, int] | None = None,
        values_p: tuple[str, int] | None = None,
        values_q: tuple[str, int] | None = None,
        values_r: tuple[str, int] | None = None,
        values_s: tuple[str, int] | None = None,
        values_t: tuple[str, int] | None = None,
        values_u: tuple[str, int] | None = None,
        values_v: tuple[str, int] | None = None,
        values_w: tuple[str, int] | None = None,
        values_x: tuple[str, int] | None = None,
        values_y: tuple[str, int] | None = None,
        values_z: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.expression = expression
        self.values_a = values_a
        if values_b is not None:
            self.values_b = values_b
        if values_c is not None:
            self.values_c = values_c
        if values_d is not None:
            self.values_d = values_d
        if values_e is not None:
            self.values_e = values_e
        if values_f is not None:
            self.values_f = values_f
        if values_g is not None:
            self.values_g = values_g
        if values_h is not None:
            self.values_h = values_h
        if values_i is not None:
            self.values_i = values_i
        if values_j is not None:
            self.values_j = values_j
        if values_k is not None:
            self.values_k = values_k
        if values_l is not None:
            self.values_l = values_l
        if values_m is not None:
            self.values_m = values_m
        if values_n is not None:
            self.values_n = values_n
        if values_o is not None:
            self.values_o = values_o
        if values_p is not None:
            self.values_p = values_p
        if values_q is not None:
            self.values_q = values_q
        if values_r is not None:
            self.values_r = values_r
        if values_s is not None:
            self.values_s = values_s
        if values_t is not None:
            self.values_t = values_t
        if values_u is not None:
            self.values_u = values_u
        if values_v is not None:
            self.values_v = values_v
        if values_w is not None:
            self.values_w = values_w
        if values_x is not None:
            self.values_x = values_x
        if values_y is not None:
            self.values_y = values_y
        if values_z is not None:
            self.values_z = values_z


__getattr__ = submodules_of(__name__)
