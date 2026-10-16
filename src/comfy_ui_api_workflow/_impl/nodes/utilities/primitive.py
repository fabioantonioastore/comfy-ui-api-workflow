from ..base import BaseNode
from ..validation import validate_bool, validate_range


class PrimitiveBoundingBox(BaseNode):
    __slots__ = ("x", "y", "width", "height")
    CATEGORY = "utilities/primitive"
    OUTPUTS = {"BOUNDING_BOX": 0}
    RETURN_TYPES = ("BOUNDING_BOX",)
    INPUT_SOCKET_TYPES = {"x": "INT", "y": "INT", "width": "INT", "height": "INT"}

    def __init__(
        self,
        node_id: str,
        x: int | tuple[str, int] = 0,
        y: int | tuple[str, int] = 0,
        width: int | tuple[str, int] = 512,
        height: int | tuple[str, int] = 512,
    ) -> None:
        super().__init__(node_id=node_id)
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.__validate()

    def __validate(self) -> None:
        validate_range("x", self.x, 0, 16_384)
        validate_range("y", self.y, 0, 16_384)
        validate_range("width", self.width, 1, 16_384)
        validate_range("height", self.height, 1, 16_384)


class PrimitiveString(BaseNode):
    __slots__ = ("value",)
    CATEGORY = "utilities/primitive"
    OUTPUTS = {"STRING": 0}
    RETURN_TYPES = ("STRING",)
    INPUT_SOCKET_TYPES = {"value": "STRING"}

    def __init__(self, node_id: str, value: str | tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.value = value


class PrimitiveStringMultiline(BaseNode):
    __slots__ = ("value",)
    CATEGORY = "utilities/primitive"
    OUTPUTS = {"STRING": 0}
    RETURN_TYPES = ("STRING",)
    INPUT_SOCKET_TYPES = {"value": "STRING"}

    def __init__(self, node_id: str, value: str | tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.value = value


class PrimitiveInt(BaseNode):
    __slots__ = ("value",)
    CATEGORY = "utilities/primitive"
    OUTPUTS = {"INT": 0}
    RETURN_TYPES = ("INT",)
    INPUT_SOCKET_TYPES = {"value": "INT"}

    def __init__(self, node_id: str, value: int | tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.value = value
        self.__validate()

    def __validate(self) -> None:
        validate_range(
            "value", self.value, -9_223_372_036_854_775_807, 9_223_372_036_854_775_807
        )


class PrimitiveFloat(BaseNode):
    __slots__ = ("value",)
    CATEGORY = "utilities/primitive"
    OUTPUTS = {"FLOAT": 0}
    RETURN_TYPES = ("FLOAT",)
    INPUT_SOCKET_TYPES = {"value": "FLOAT"}

    def __init__(self, node_id: str, value: float | tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.value = value
        self.__validate()

    def __validate(self) -> None:
        validate_range(
            "value", self.value, -9_223_372_036_854_775_807, 9_223_372_036_854_775_807
        )


class PrimitiveBoolean(BaseNode):
    __slots__ = ("value",)
    CATEGORY = "utilities/primitive"
    OUTPUTS = {"BOOLEAN": 0}
    RETURN_TYPES = ("BOOLEAN",)
    INPUT_SOCKET_TYPES = {"value": "BOOLEAN"}

    def __init__(self, node_id: str, value: bool | tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.value = value
        self.__validate()

    def __validate(self) -> None:
        validate_bool("value", self.value)
