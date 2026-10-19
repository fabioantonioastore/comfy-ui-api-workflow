from typing import Any, Literal, get_args

from .base import BaseNode
from .validation import validate_bool, validate_choice, validate_range
from .dynamic import Combo, DynamicInputs, Number, Text, validate_dynamic


class TruncateText(BaseNode):
    __slots__ = ("texts", "max_length")
    CATEGORY = "text"
    OUTPUTS = {"texts": 0}
    RETURN_TYPES = ("STRING",)
    INPUT_SOCKET_TYPES = {"texts": "STRING", "max_length": "INT"}

    def __init__(
        self,
        node_id: str,
        texts: str | tuple[str, int],
        max_length: int | tuple[str, int] = 77,
    ) -> None:
        super().__init__(node_id=node_id)
        self.texts = texts
        self.max_length = max_length
        self.__validate()

    def __validate(self) -> None:
        validate_range("max_length", self.max_length, 1, 10_000)


class StringConcatenate(BaseNode):
    __slots__ = ("string_a", "string_b", "delimiter")
    CATEGORY = "text"
    OUTPUTS = {"STRING": 0}
    RETURN_TYPES = ("STRING",)
    INPUT_SOCKET_TYPES = {
        "string_a": "STRING",
        "string_b": "STRING",
        "delimiter": "STRING",
    }

    def __init__(
        self,
        node_id: str,
        string_a: str | tuple[str, int],
        string_b: str | tuple[str, int],
        delimiter: str | tuple[str, int] = "",
    ) -> None:
        super().__init__(node_id=node_id)
        self.string_a = string_a
        self.string_b = string_b
        self.delimiter = delimiter


class StringSubstring(BaseNode):
    __slots__ = ("string", "start", "end")
    CATEGORY = "text"
    OUTPUTS = {"STRING": 0}
    RETURN_TYPES = ("STRING",)
    INPUT_SOCKET_TYPES = {"string": "STRING", "start": "INT", "end": "INT"}

    def __init__(
        self,
        node_id: str,
        string: str | tuple[str, int],
        start: int | tuple[str, int],
        end: int | tuple[str, int],
    ) -> None:
        super().__init__(node_id=node_id)
        self.string = string
        self.start = start
        self.end = end
        self.__validate()

    def __validate(self) -> None:
        validate_range("start", self.start)
        validate_range("end", self.end)


class StringLength(BaseNode):
    __slots__ = ("string",)
    CATEGORY = "text"
    OUTPUTS = {"length": 0}
    RETURN_TYPES = ("INT",)
    INPUT_SOCKET_TYPES = {"string": "STRING"}

    def __init__(self, node_id: str, string: str | tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.string = string


CaseConverter_Modes = Literal["UPPERCASE", "lowercase", "Capitalize", "Title Case"]
CASECONVERTER_MODES = frozenset(get_args(CaseConverter_Modes))


class CaseConverter(BaseNode):
    __slots__ = ("string", "mode")
    CATEGORY = "text"
    OUTPUTS = {"STRING": 0}
    RETURN_TYPES = ("STRING",)
    INPUT_SOCKET_TYPES = {"string": "STRING", "mode": "COMBO"}

    def __init__(
        self,
        node_id: str,
        string: str | tuple[str, int],
        mode: CaseConverter_Modes | tuple[str, int],
    ) -> None:
        super().__init__(node_id=node_id)
        self.string = string
        self.mode = mode
        self.__validate()

    def __validate(self) -> None:
        validate_choice("mode", self.mode, CASECONVERTER_MODES)


StringTrim_Modes = Literal["Both", "Left", "Right"]
STRINGTRIM_MODES = frozenset(get_args(StringTrim_Modes))


class StringTrim(BaseNode):
    __slots__ = ("string", "mode")
    CATEGORY = "text"
    OUTPUTS = {"STRING": 0}
    RETURN_TYPES = ("STRING",)
    INPUT_SOCKET_TYPES = {"string": "STRING", "mode": "COMBO"}

    def __init__(
        self,
        node_id: str,
        string: str | tuple[str, int],
        mode: StringTrim_Modes | tuple[str, int],
    ) -> None:
        super().__init__(node_id=node_id)
        self.string = string
        self.mode = mode
        self.__validate()

    def __validate(self) -> None:
        validate_choice("mode", self.mode, STRINGTRIM_MODES)


class StringReplace(BaseNode):
    __slots__ = ("string", "find", "replace")
    CATEGORY = "text"
    OUTPUTS = {"STRING": 0}
    RETURN_TYPES = ("STRING",)
    INPUT_SOCKET_TYPES = {"string": "STRING", "find": "STRING", "replace": "STRING"}

    def __init__(
        self,
        node_id: str,
        string: str | tuple[str, int],
        find: str | tuple[str, int],
        replace: str | tuple[str, int],
    ) -> None:
        super().__init__(node_id=node_id)
        self.string = string
        self.find = find
        self.replace = replace


class StringContains(BaseNode):
    __slots__ = ("string", "substring", "case_sensitive")
    CATEGORY = "text"
    OUTPUTS = {"contains": 0}
    RETURN_TYPES = ("BOOLEAN",)
    INPUT_SOCKET_TYPES = {
        "string": "STRING",
        "substring": "STRING",
        "case_sensitive": "BOOLEAN",
    }

    def __init__(
        self,
        node_id: str,
        string: str | tuple[str, int],
        substring: str | tuple[str, int],
        case_sensitive: bool | tuple[str, int] = True,
    ) -> None:
        super().__init__(node_id=node_id)
        self.string = string
        self.substring = substring
        self.case_sensitive = case_sensitive
        self.__validate()

    def __validate(self) -> None:
        validate_bool("case_sensitive", self.case_sensitive)


StringCompare_Modes = Literal["Starts With", "Ends With", "Equal"]
STRINGCOMPARE_MODES = frozenset(get_args(StringCompare_Modes))


class StringCompare(BaseNode):
    __slots__ = ("string_a", "string_b", "mode", "case_sensitive")
    CATEGORY = "text"
    OUTPUTS = {"BOOLEAN": 0}
    RETURN_TYPES = ("BOOLEAN",)
    INPUT_SOCKET_TYPES = {
        "string_a": "STRING",
        "string_b": "STRING",
        "mode": "COMBO",
        "case_sensitive": "BOOLEAN",
    }

    def __init__(
        self,
        node_id: str,
        string_a: str | tuple[str, int],
        string_b: str | tuple[str, int],
        mode: StringCompare_Modes | tuple[str, int],
        case_sensitive: bool | tuple[str, int] = True,
    ) -> None:
        super().__init__(node_id=node_id)
        self.string_a = string_a
        self.string_b = string_b
        self.mode = mode
        self.case_sensitive = case_sensitive
        self.__validate()

    def __validate(self) -> None:
        validate_choice("mode", self.mode, STRINGCOMPARE_MODES)
        validate_bool("case_sensitive", self.case_sensitive)


class RegexMatch(BaseNode):
    __slots__ = ("string", "regex_pattern", "case_insensitive", "multiline", "dotall")
    CATEGORY = "text"
    OUTPUTS = {"matches": 0}
    RETURN_TYPES = ("BOOLEAN",)
    INPUT_SOCKET_TYPES = {
        "string": "STRING",
        "regex_pattern": "STRING",
        "case_insensitive": "BOOLEAN",
        "multiline": "BOOLEAN",
        "dotall": "BOOLEAN",
    }

    def __init__(
        self,
        node_id: str,
        string: str | tuple[str, int],
        regex_pattern: str | tuple[str, int],
        case_insensitive: bool | tuple[str, int] = True,
        multiline: bool | tuple[str, int] = False,
        dotall: bool | tuple[str, int] = False,
    ) -> None:
        super().__init__(node_id=node_id)
        self.string = string
        self.regex_pattern = regex_pattern
        self.case_insensitive = case_insensitive
        self.multiline = multiline
        self.dotall = dotall
        self.__validate()

    def __validate(self) -> None:
        validate_bool("case_insensitive", self.case_insensitive)
        validate_bool("multiline", self.multiline)
        validate_bool("dotall", self.dotall)


RegexExtract_Modes = Literal["First Match", "All Matches", "First Group", "All Groups"]
REGEXEXTRACT_MODES = frozenset(get_args(RegexExtract_Modes))


class RegexExtract(BaseNode):
    __slots__ = (
        "string",
        "regex_pattern",
        "mode",
        "case_insensitive",
        "multiline",
        "dotall",
        "group_index",
    )
    CATEGORY = "text"
    OUTPUTS = {"STRING": 0}
    RETURN_TYPES = ("STRING",)
    INPUT_SOCKET_TYPES = {
        "string": "STRING",
        "regex_pattern": "STRING",
        "mode": "COMBO",
        "case_insensitive": "BOOLEAN",
        "multiline": "BOOLEAN",
        "dotall": "BOOLEAN",
        "group_index": "INT",
    }

    def __init__(
        self,
        node_id: str,
        string: str | tuple[str, int],
        regex_pattern: str | tuple[str, int],
        mode: RegexExtract_Modes | tuple[str, int],
        case_insensitive: bool | tuple[str, int] = True,
        multiline: bool | tuple[str, int] = False,
        dotall: bool | tuple[str, int] = False,
        group_index: int | tuple[str, int] = 1,
    ) -> None:
        super().__init__(node_id=node_id)
        self.string = string
        self.regex_pattern = regex_pattern
        self.mode = mode
        self.case_insensitive = case_insensitive
        self.multiline = multiline
        self.dotall = dotall
        self.group_index = group_index
        self.__validate()

    def __validate(self) -> None:
        validate_choice("mode", self.mode, REGEXEXTRACT_MODES)
        validate_bool("case_insensitive", self.case_insensitive)
        validate_bool("multiline", self.multiline)
        validate_bool("dotall", self.dotall)
        validate_range("group_index", self.group_index, 0, 100)


class RegexReplace(BaseNode):
    __slots__ = (
        "string",
        "regex_pattern",
        "replace",
        "case_insensitive",
        "multiline",
        "dotall",
        "count",
    )
    CATEGORY = "text"
    OUTPUTS = {"STRING": 0}
    RETURN_TYPES = ("STRING",)
    INPUT_SOCKET_TYPES = {
        "string": "STRING",
        "regex_pattern": "STRING",
        "replace": "STRING",
        "case_insensitive": "BOOLEAN",
        "multiline": "BOOLEAN",
        "dotall": "BOOLEAN",
        "count": "INT",
    }

    def __init__(
        self,
        node_id: str,
        string: str | tuple[str, int],
        regex_pattern: str | tuple[str, int],
        replace: str | tuple[str, int],
        case_insensitive: bool | tuple[str, int] = True,
        multiline: bool | tuple[str, int] = False,
        dotall: bool | tuple[str, int] = False,
        count: int | tuple[str, int] = 0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.string = string
        self.regex_pattern = regex_pattern
        self.replace = replace
        self.case_insensitive = case_insensitive
        self.multiline = multiline
        self.dotall = dotall
        self.count = count
        self.__validate()

    def __validate(self) -> None:
        validate_bool("case_insensitive", self.case_insensitive)
        validate_bool("multiline", self.multiline)
        validate_bool("dotall", self.dotall)
        validate_range("count", self.count, 0, 100)


class JsonExtractString(BaseNode):
    __slots__ = ("json_string", "key")
    CATEGORY = "text"
    OUTPUTS = {"STRING": 0}
    RETURN_TYPES = ("STRING",)
    INPUT_SOCKET_TYPES = {"json_string": "STRING", "key": "STRING"}

    def __init__(
        self,
        node_id: str,
        json_string: str | tuple[str, int],
        key: str | tuple[str, int],
    ) -> None:
        super().__init__(node_id=node_id)
        self.json_string = json_string
        self.key = key


class ConvertDictionaryToString(BaseNode):
    __slots__ = ("dictionary", "indent")
    CATEGORY = "text"
    OUTPUTS = {"STRING": 0}
    RETURN_TYPES = ("STRING",)
    INPUT_SOCKET_TYPES = {"dictionary": "DICT", "indent": "INT"}

    def __init__(
        self,
        node_id: str,
        dictionary: tuple[str, int],
        indent: int | tuple[str, int] = 2,
    ) -> None:
        super().__init__(node_id=node_id)
        self.dictionary = dictionary
        self.indent = indent
        self.__validate()

    def __validate(self) -> None:
        validate_range("indent", self.indent, 0, 8)


class ConvertArrayToString(BaseNode):
    __slots__ = ("array", "indent")
    CATEGORY = "text"
    OUTPUTS = {"STRING": 0}
    RETURN_TYPES = ("STRING",)
    INPUT_SOCKET_TYPES = {"array": "ARRAY", "indent": "INT"}

    def __init__(
        self, node_id: str, array: tuple[str, int], indent: int | tuple[str, int] = 2
    ) -> None:
        super().__init__(node_id=node_id)
        self.array = array
        self.indent = indent
        self.__validate()

    def __validate(self) -> None:
        validate_range("indent", self.indent, 0, 8)


TextOverlay_Positions = Literal["top", "bottom"]
TextOverlay_Aligns = Literal["left", "center", "right"]
TEXTOVERLAY_POSITIONS = frozenset(get_args(TextOverlay_Positions))
TEXTOVERLAY_ALIGNS = frozenset(get_args(TextOverlay_Aligns))


class TextOverlay(BaseNode):
    __slots__ = ("images", "text", "font_size", "color", "position", "align", "outline")
    CATEGORY = "text"
    OUTPUTS = {"images": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_SOCKET_TYPES = {
        "images": "IMAGE",
        "text": "STRING",
        "font_size": "FLOAT",
        "color": "COLOR",
        "position": "COMBO",
        "align": "COMBO",
        "outline": "BOOLEAN",
    }

    def __init__(
        self,
        node_id: str,
        images: tuple[str, int],
        text: str | tuple[str, int] = "",
        font_size: float | tuple[str, int] = 5.0,
        color: Any = "#ffffff",
        position: TextOverlay_Positions | tuple[str, int] = "top",
        align: TextOverlay_Aligns | tuple[str, int] = "left",
        outline: bool | tuple[str, int] = True,
    ) -> None:
        super().__init__(node_id=node_id)
        self.images = images
        self.text = text
        self.font_size = font_size
        self.color = color
        self.position = position
        self.align = align
        self.outline = outline
        self.__validate()

    def __validate(self) -> None:
        validate_range("font_size", self.font_size, 0.5, 50.0)
        validate_choice("position", self.position, TEXTOVERLAY_POSITIONS)
        validate_choice("align", self.align, TEXTOVERLAY_ALIGNS)
        validate_bool("outline", self.outline)


SaveText_Formats = Literal["txt", "csv", "md", "json"]
SAVETEXT_FORMATS = frozenset(get_args(SaveText_Formats))


class SaveText(BaseNode):
    __slots__ = ("text", "filename_prefix", "format")
    CATEGORY = "text"
    OUTPUTS = {"text": 0}
    RETURN_TYPES = ("STRING",)
    INPUT_SOCKET_TYPES = {
        "text": "STRING",
        "filename_prefix": "STRING",
        "format": "COMBO",
    }

    def __init__(
        self,
        node_id: str,
        text: tuple[str, int],
        filename_prefix: str | tuple[str, int] = "ComfyUI",
        format: SaveText_Formats | tuple[str, int] = "txt",
    ) -> None:
        super().__init__(node_id=node_id)
        self.text = text
        self.filename_prefix = filename_prefix
        self.format = format
        self.__validate()

    def __validate(self) -> None:
        validate_choice("format", self.format, SAVETEXT_FORMATS)


BuildJsonPromptIdeogram_Styles = Literal["none", "photo", "art_style"]


class BuildJsonPromptIdeogram(DynamicInputs, BaseNode):
    __slots__ = (
        "element",
        "high_level_description",
        "background",
        "style",
        "style_photo",
        "style_art_style",
        "aesthetics",
        "lighting",
        "medium",
        "color_palette",
    )
    CATEGORY = "text"
    OUTPUTS = {"prompt": 0}
    RETURN_TYPES = ("DICT",)
    INPUT_NAMES = {"style_photo": "style.photo", "style_art_style": "style.art_style"}
    INPUT_SOCKET_TYPES = {
        "element": "ARRAY",
        "high_level_description": "STRING",
        "background": "STRING",
        "style": "COMFY_DYNAMICCOMBO_V3",
        "style_photo": "STRING",
        "style_art_style": "STRING",
        "aesthetics": "STRING",
        "lighting": "STRING",
        "medium": "STRING",
        "color_palette": "COLORS",
    }
    DYNAMIC_INPUTS = {
        "style": Combo(
            {
                "none": {},
                "photo": {"style_photo": Text(default="")},
                "art_style": {"style_art_style": Text(default="")},
            }
        )
    }

    def __init__(
        self,
        node_id: str,
        element: tuple[str, int],
        high_level_description: str | tuple[str, int] = "",
        background: str | tuple[str, int] = "",
        style: BuildJsonPromptIdeogram_Styles = "none",
        style_photo: str | tuple[str, int] | None = None,
        style_art_style: str | tuple[str, int] | None = None,
        aesthetics: str | tuple[str, int] = "",
        lighting: str | tuple[str, int] = "",
        medium: str | tuple[str, int] = "",
        color_palette: Any = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.element = element
        self.high_level_description = high_level_description
        self.background = background
        self.style = style
        if style_photo is not None:
            self.style_photo = style_photo
        if style_art_style is not None:
            self.style_art_style = style_art_style
        self.aesthetics = aesthetics
        self.lighting = lighting
        self.medium = medium
        self.color_palette = [] if color_palette is None else color_palette
        self.__validate()

    def __validate(self) -> None:
        validate_dynamic(self)


class StringFormat(BaseNode):
    __slots__ = (
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
        "f_string",
    )
    CATEGORY = "text"
    OUTPUTS = {"STRING": 0}
    RETURN_TYPES = ("STRING",)
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
        "values_a": "*",
        "values_b": "*",
        "values_c": "*",
        "values_d": "*",
        "values_e": "*",
        "values_f": "*",
        "values_g": "*",
        "values_h": "*",
        "values_i": "*",
        "values_j": "*",
        "values_k": "*",
        "values_l": "*",
        "values_m": "*",
        "values_n": "*",
        "values_o": "*",
        "values_p": "*",
        "values_q": "*",
        "values_r": "*",
        "values_s": "*",
        "values_t": "*",
        "values_u": "*",
        "values_v": "*",
        "values_w": "*",
        "values_x": "*",
        "values_y": "*",
        "values_z": "*",
        "f_string": "STRING",
    }

    def __init__(
        self,
        node_id: str,
        values_a: tuple[str, int] | None = None,
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
        f_string: str | tuple[str, int] = "{a}",
    ) -> None:
        super().__init__(node_id=node_id)
        if values_a is not None:
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
        self.f_string = f_string


TextGenerate_Sampling_Modes = Literal["on", "off"]


TextGenerate_Mtps = Literal["auto", "off", "2", "3", "4", "5"]


TEXTGENERATE_MTPS = frozenset(get_args(TextGenerate_Mtps))


class TextGenerate(DynamicInputs, BaseNode):
    __slots__ = (
        "clip",
        "prompt",
        "max_length",
        "sampling_mode",
        "sampling_mode_temperature",
        "sampling_mode_top_k",
        "sampling_mode_top_p",
        "sampling_mode_min_p",
        "sampling_mode_repetition_penalty",
        "sampling_mode_seed",
        "sampling_mode_presence_penalty",
        "image",
        "video",
        "audio",
        "thinking",
        "use_default_template",
        "mtp",
        "system_prompt",
    )
    CATEGORY = "text"
    OUTPUTS = {"generated_text": 0, "thinking": 1}
    RETURN_TYPES = (
        "STRING",
        "STRING",
    )
    INPUT_NAMES = {
        "sampling_mode_temperature": "sampling_mode.temperature",
        "sampling_mode_top_k": "sampling_mode.top_k",
        "sampling_mode_top_p": "sampling_mode.top_p",
        "sampling_mode_min_p": "sampling_mode.min_p",
        "sampling_mode_repetition_penalty": "sampling_mode.repetition_penalty",
        "sampling_mode_seed": "sampling_mode.seed",
        "sampling_mode_presence_penalty": "sampling_mode.presence_penalty",
    }
    INPUT_SOCKET_TYPES = {
        "clip": "CLIP",
        "prompt": "STRING",
        "max_length": "INT",
        "sampling_mode": "COMFY_DYNAMICCOMBO_V3",
        "sampling_mode_temperature": "FLOAT",
        "sampling_mode_top_k": "INT",
        "sampling_mode_top_p": "FLOAT",
        "sampling_mode_min_p": "FLOAT",
        "sampling_mode_repetition_penalty": "FLOAT",
        "sampling_mode_seed": "INT",
        "sampling_mode_presence_penalty": "FLOAT",
        "image": "IMAGE",
        "video": "IMAGE",
        "audio": "AUDIO",
        "thinking": "BOOLEAN",
        "use_default_template": "BOOLEAN",
        "mtp": "COMBO",
        "system_prompt": "STRING",
    }
    DYNAMIC_INPUTS = {
        "sampling_mode": Combo(
            {
                "on": {
                    "sampling_mode_temperature": Number(
                        default=0.7, low=0.01, high=2.0
                    ),
                    "sampling_mode_top_k": Number(default=64, low=0, high=1000),
                    "sampling_mode_top_p": Number(default=0.95, low=0.0, high=1.0),
                    "sampling_mode_min_p": Number(default=0.05, low=0.0, high=1.0),
                    "sampling_mode_repetition_penalty": Number(
                        default=1.05, low=0.0, high=5.0
                    ),
                    "sampling_mode_seed": Number(
                        default=0, low=0, high=18446744073709551615
                    ),
                    "sampling_mode_presence_penalty": Number(
                        default=0.0, low=0.0, high=5.0, required=False
                    ),
                },
                "off": {},
            }
        )
    }

    def __init__(
        self,
        node_id: str,
        clip: tuple[str, int],
        prompt: str | tuple[str, int] = "",
        max_length: int | tuple[str, int] = 512,
        sampling_mode: TextGenerate_Sampling_Modes = "on",
        sampling_mode_temperature: float | tuple[str, int] | None = None,
        sampling_mode_top_k: int | tuple[str, int] | None = None,
        sampling_mode_top_p: float | tuple[str, int] | None = None,
        sampling_mode_min_p: float | tuple[str, int] | None = None,
        sampling_mode_repetition_penalty: float | tuple[str, int] | None = None,
        sampling_mode_seed: int | tuple[str, int] | None = None,
        sampling_mode_presence_penalty: float | tuple[str, int] | None = None,
        image: tuple[str, int] | None = None,
        video: tuple[str, int] | None = None,
        audio: tuple[str, int] | None = None,
        thinking: bool | tuple[str, int] = False,
        use_default_template: bool | tuple[str, int] = True,
        mtp: TextGenerate_Mtps | tuple[str, int] = "auto",
        system_prompt: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.clip = clip
        self.prompt = prompt
        self.max_length = max_length
        self.sampling_mode = sampling_mode
        if sampling_mode_temperature is not None:
            self.sampling_mode_temperature = sampling_mode_temperature
        if sampling_mode_top_k is not None:
            self.sampling_mode_top_k = sampling_mode_top_k
        if sampling_mode_top_p is not None:
            self.sampling_mode_top_p = sampling_mode_top_p
        if sampling_mode_min_p is not None:
            self.sampling_mode_min_p = sampling_mode_min_p
        if sampling_mode_repetition_penalty is not None:
            self.sampling_mode_repetition_penalty = sampling_mode_repetition_penalty
        if sampling_mode_seed is not None:
            self.sampling_mode_seed = sampling_mode_seed
        if sampling_mode_presence_penalty is not None:
            self.sampling_mode_presence_penalty = sampling_mode_presence_penalty
        if image is not None:
            self.image = image
        if video is not None:
            self.video = video
        if audio is not None:
            self.audio = audio
        self.thinking = thinking
        self.use_default_template = use_default_template
        self.mtp = mtp
        if system_prompt is not None:
            self.system_prompt = system_prompt
        self.__validate()

    def __validate(self) -> None:
        validate_range("max_length", self.max_length, 1, 32768)
        validate_bool("thinking", self.thinking)
        validate_bool("use_default_template", self.use_default_template)
        validate_choice("mtp", self.mtp, TEXTGENERATE_MTPS)
        validate_dynamic(self)


TextGenerateLTX2Prompt_Sampling_Modes = Literal["on", "off"]


TextGenerateLTX2Prompt_Mtps = Literal["auto", "off", "2", "3", "4", "5"]


TEXTGENERATELTX2PROMPT_MTPS = frozenset(get_args(TextGenerateLTX2Prompt_Mtps))


class TextGenerateLTX2Prompt(DynamicInputs, BaseNode):
    __slots__ = (
        "clip",
        "prompt",
        "max_length",
        "sampling_mode",
        "sampling_mode_temperature",
        "sampling_mode_top_k",
        "sampling_mode_top_p",
        "sampling_mode_min_p",
        "sampling_mode_repetition_penalty",
        "sampling_mode_seed",
        "sampling_mode_presence_penalty",
        "image",
        "video",
        "audio",
        "thinking",
        "use_default_template",
        "mtp",
        "system_prompt",
    )
    CATEGORY = "text"
    OUTPUTS = {"generated_text": 0, "thinking": 1}
    RETURN_TYPES = (
        "STRING",
        "STRING",
    )
    INPUT_NAMES = {
        "sampling_mode_temperature": "sampling_mode.temperature",
        "sampling_mode_top_k": "sampling_mode.top_k",
        "sampling_mode_top_p": "sampling_mode.top_p",
        "sampling_mode_min_p": "sampling_mode.min_p",
        "sampling_mode_repetition_penalty": "sampling_mode.repetition_penalty",
        "sampling_mode_seed": "sampling_mode.seed",
        "sampling_mode_presence_penalty": "sampling_mode.presence_penalty",
    }
    INPUT_SOCKET_TYPES = {
        "clip": "CLIP",
        "prompt": "STRING",
        "max_length": "INT",
        "sampling_mode": "COMFY_DYNAMICCOMBO_V3",
        "sampling_mode_temperature": "FLOAT",
        "sampling_mode_top_k": "INT",
        "sampling_mode_top_p": "FLOAT",
        "sampling_mode_min_p": "FLOAT",
        "sampling_mode_repetition_penalty": "FLOAT",
        "sampling_mode_seed": "INT",
        "sampling_mode_presence_penalty": "FLOAT",
        "image": "IMAGE",
        "video": "IMAGE",
        "audio": "AUDIO",
        "thinking": "BOOLEAN",
        "use_default_template": "BOOLEAN",
        "mtp": "COMBO",
        "system_prompt": "STRING",
    }
    DYNAMIC_INPUTS = {
        "sampling_mode": Combo(
            {
                "on": {
                    "sampling_mode_temperature": Number(
                        default=0.7, low=0.01, high=2.0
                    ),
                    "sampling_mode_top_k": Number(default=64, low=0, high=1000),
                    "sampling_mode_top_p": Number(default=0.95, low=0.0, high=1.0),
                    "sampling_mode_min_p": Number(default=0.05, low=0.0, high=1.0),
                    "sampling_mode_repetition_penalty": Number(
                        default=1.05, low=0.0, high=5.0
                    ),
                    "sampling_mode_seed": Number(
                        default=0, low=0, high=18446744073709551615
                    ),
                    "sampling_mode_presence_penalty": Number(
                        default=0.0, low=0.0, high=5.0, required=False
                    ),
                },
                "off": {},
            }
        )
    }

    def __init__(
        self,
        node_id: str,
        clip: tuple[str, int],
        prompt: str | tuple[str, int] = "",
        max_length: int | tuple[str, int] = 512,
        sampling_mode: TextGenerateLTX2Prompt_Sampling_Modes = "on",
        sampling_mode_temperature: float | tuple[str, int] | None = None,
        sampling_mode_top_k: int | tuple[str, int] | None = None,
        sampling_mode_top_p: float | tuple[str, int] | None = None,
        sampling_mode_min_p: float | tuple[str, int] | None = None,
        sampling_mode_repetition_penalty: float | tuple[str, int] | None = None,
        sampling_mode_seed: int | tuple[str, int] | None = None,
        sampling_mode_presence_penalty: float | tuple[str, int] | None = None,
        image: tuple[str, int] | None = None,
        video: tuple[str, int] | None = None,
        audio: tuple[str, int] | None = None,
        thinking: bool | tuple[str, int] = False,
        use_default_template: bool | tuple[str, int] = True,
        mtp: TextGenerateLTX2Prompt_Mtps | tuple[str, int] = "auto",
        system_prompt: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.clip = clip
        self.prompt = prompt
        self.max_length = max_length
        self.sampling_mode = sampling_mode
        if sampling_mode_temperature is not None:
            self.sampling_mode_temperature = sampling_mode_temperature
        if sampling_mode_top_k is not None:
            self.sampling_mode_top_k = sampling_mode_top_k
        if sampling_mode_top_p is not None:
            self.sampling_mode_top_p = sampling_mode_top_p
        if sampling_mode_min_p is not None:
            self.sampling_mode_min_p = sampling_mode_min_p
        if sampling_mode_repetition_penalty is not None:
            self.sampling_mode_repetition_penalty = sampling_mode_repetition_penalty
        if sampling_mode_seed is not None:
            self.sampling_mode_seed = sampling_mode_seed
        if sampling_mode_presence_penalty is not None:
            self.sampling_mode_presence_penalty = sampling_mode_presence_penalty
        if image is not None:
            self.image = image
        if video is not None:
            self.video = video
        if audio is not None:
            self.audio = audio
        self.thinking = thinking
        self.use_default_template = use_default_template
        self.mtp = mtp
        if system_prompt is not None:
            self.system_prompt = system_prompt
        self.__validate()

    def __validate(self) -> None:
        validate_range("max_length", self.max_length, 1, 32768)
        validate_bool("thinking", self.thinking)
        validate_bool("use_default_template", self.use_default_template)
        validate_choice("mtp", self.mtp, TEXTGENERATELTX2PROMPT_MTPS)
        validate_dynamic(self)
