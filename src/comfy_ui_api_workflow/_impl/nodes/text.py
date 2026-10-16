from typing import Any, Literal, get_args

from .base import BaseNode
from .validation import validate_bool, validate_choice, validate_range


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
