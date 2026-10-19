from ..base import BaseNode
from ..validation import validate_bool


class ComfySwitchNode(BaseNode):
    __slots__ = ("switch", "on_false", "on_true")
    CATEGORY = "utilities/logic"
    OUTPUTS = {"output": 0}
    RETURN_TYPES = ("COMFY_MATCHTYPE_V3",)
    INPUT_SOCKET_TYPES = {
        "switch": "BOOLEAN",
        "on_false": "COMFY_MATCHTYPE_V3",
        "on_true": "COMFY_MATCHTYPE_V3",
    }

    def __init__(
        self,
        node_id: str,
        switch: bool | tuple[str, int],
        on_false: tuple[str, int] | None = None,
        on_true: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.switch = switch
        if on_false is not None:
            self.on_false = on_false
        if on_true is not None:
            self.on_true = on_true
        self.__validate()

    def __validate(self) -> None:
        validate_bool("switch", self.switch)


class ComfyNotNode(BaseNode):
    __slots__ = ("value",)
    CATEGORY = "utilities/logic"
    OUTPUTS = {"BOOLEAN": 0}
    RETURN_TYPES = ("BOOLEAN",)
    INPUT_SOCKET_TYPES = {"value": "*"}

    def __init__(self, node_id: str, value: tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.value = value


class ComfyAndNode(BaseNode):
    __slots__ = (
        "values_value0",
        "values_value1",
        "values_value2",
        "values_value3",
        "values_value4",
        "values_value5",
        "values_value6",
        "values_value7",
        "values_value8",
        "values_value9",
    )
    CATEGORY = "utilities/logic"
    OUTPUTS = {"BOOLEAN": 0}
    RETURN_TYPES = ("BOOLEAN",)
    INPUT_NAMES = {
        "values_value0": "values.value0",
        "values_value1": "values.value1",
        "values_value2": "values.value2",
        "values_value3": "values.value3",
        "values_value4": "values.value4",
        "values_value5": "values.value5",
        "values_value6": "values.value6",
        "values_value7": "values.value7",
        "values_value8": "values.value8",
        "values_value9": "values.value9",
    }
    INPUT_SOCKET_TYPES = {
        "values_value0": "*",
        "values_value1": "*",
        "values_value2": "*",
        "values_value3": "*",
        "values_value4": "*",
        "values_value5": "*",
        "values_value6": "*",
        "values_value7": "*",
        "values_value8": "*",
        "values_value9": "*",
    }

    def __init__(
        self,
        node_id: str,
        values_value0: tuple[str, int],
        values_value1: tuple[str, int] | None = None,
        values_value2: tuple[str, int] | None = None,
        values_value3: tuple[str, int] | None = None,
        values_value4: tuple[str, int] | None = None,
        values_value5: tuple[str, int] | None = None,
        values_value6: tuple[str, int] | None = None,
        values_value7: tuple[str, int] | None = None,
        values_value8: tuple[str, int] | None = None,
        values_value9: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.values_value0 = values_value0
        if values_value1 is not None:
            self.values_value1 = values_value1
        if values_value2 is not None:
            self.values_value2 = values_value2
        if values_value3 is not None:
            self.values_value3 = values_value3
        if values_value4 is not None:
            self.values_value4 = values_value4
        if values_value5 is not None:
            self.values_value5 = values_value5
        if values_value6 is not None:
            self.values_value6 = values_value6
        if values_value7 is not None:
            self.values_value7 = values_value7
        if values_value8 is not None:
            self.values_value8 = values_value8
        if values_value9 is not None:
            self.values_value9 = values_value9


class ComfyOrNode(BaseNode):
    __slots__ = (
        "values_value0",
        "values_value1",
        "values_value2",
        "values_value3",
        "values_value4",
        "values_value5",
        "values_value6",
        "values_value7",
        "values_value8",
        "values_value9",
    )
    CATEGORY = "utilities/logic"
    OUTPUTS = {"BOOLEAN": 0}
    RETURN_TYPES = ("BOOLEAN",)
    INPUT_NAMES = {
        "values_value0": "values.value0",
        "values_value1": "values.value1",
        "values_value2": "values.value2",
        "values_value3": "values.value3",
        "values_value4": "values.value4",
        "values_value5": "values.value5",
        "values_value6": "values.value6",
        "values_value7": "values.value7",
        "values_value8": "values.value8",
        "values_value9": "values.value9",
    }
    INPUT_SOCKET_TYPES = {
        "values_value0": "*",
        "values_value1": "*",
        "values_value2": "*",
        "values_value3": "*",
        "values_value4": "*",
        "values_value5": "*",
        "values_value6": "*",
        "values_value7": "*",
        "values_value8": "*",
        "values_value9": "*",
    }

    def __init__(
        self,
        node_id: str,
        values_value0: tuple[str, int],
        values_value1: tuple[str, int] | None = None,
        values_value2: tuple[str, int] | None = None,
        values_value3: tuple[str, int] | None = None,
        values_value4: tuple[str, int] | None = None,
        values_value5: tuple[str, int] | None = None,
        values_value6: tuple[str, int] | None = None,
        values_value7: tuple[str, int] | None = None,
        values_value8: tuple[str, int] | None = None,
        values_value9: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.values_value0 = values_value0
        if values_value1 is not None:
            self.values_value1 = values_value1
        if values_value2 is not None:
            self.values_value2 = values_value2
        if values_value3 is not None:
            self.values_value3 = values_value3
        if values_value4 is not None:
            self.values_value4 = values_value4
        if values_value5 is not None:
            self.values_value5 = values_value5
        if values_value6 is not None:
            self.values_value6 = values_value6
        if values_value7 is not None:
            self.values_value7 = values_value7
        if values_value8 is not None:
            self.values_value8 = values_value8
        if values_value9 is not None:
            self.values_value9 = values_value9
