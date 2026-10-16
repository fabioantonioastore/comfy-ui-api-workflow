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
