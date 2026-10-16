from ...base import BaseNode
from ...validation import validate_not_empty, validate_range


class CreateHookLora(BaseNode):
    __slots__ = ("lora_name", "strength_model", "strength_clip", "prev_hooks")
    CATEGORY = "advanced/hooks/create"
    OUTPUTS = {"HOOKS": 0}
    RETURN_TYPES = ("HOOKS",)
    INPUT_SOCKET_TYPES = {
        "lora_name": "V1 COMBO",
        "strength_model": "FLOAT",
        "strength_clip": "FLOAT",
        "prev_hooks": "HOOKS",
    }

    def __init__(
        self,
        node_id: str,
        lora_name: str | tuple[str, int],
        strength_model: float | tuple[str, int] = 1.0,
        strength_clip: float | tuple[str, int] = 1.0,
        prev_hooks: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.lora_name = lora_name
        self.strength_model = strength_model
        self.strength_clip = strength_clip
        if prev_hooks is not None:
            self.prev_hooks = prev_hooks
        self.__validate()

    def __validate(self) -> None:
        validate_not_empty("lora_name", self.lora_name)
        validate_range("strength_model", self.strength_model, -20.0, 20.0)
        validate_range("strength_clip", self.strength_clip, -20.0, 20.0)


class CreateHookLoraModelOnly(BaseNode):
    __slots__ = ("lora_name", "strength_model", "prev_hooks")
    CATEGORY = "advanced/hooks/create"
    OUTPUTS = {"HOOKS": 0}
    RETURN_TYPES = ("HOOKS",)
    INPUT_SOCKET_TYPES = {
        "lora_name": "V1 COMBO",
        "strength_model": "FLOAT",
        "prev_hooks": "HOOKS",
    }

    def __init__(
        self,
        node_id: str,
        lora_name: str | tuple[str, int],
        strength_model: float | tuple[str, int] = 1.0,
        prev_hooks: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.lora_name = lora_name
        self.strength_model = strength_model
        if prev_hooks is not None:
            self.prev_hooks = prev_hooks
        self.__validate()

    def __validate(self) -> None:
        validate_not_empty("lora_name", self.lora_name)
        validate_range("strength_model", self.strength_model, -20.0, 20.0)


class CreateHookModelAsLora(BaseNode):
    __slots__ = ("ckpt_name", "strength_model", "strength_clip", "prev_hooks")
    CATEGORY = "advanced/hooks/create"
    OUTPUTS = {"HOOKS": 0}
    RETURN_TYPES = ("HOOKS",)
    INPUT_SOCKET_TYPES = {
        "ckpt_name": "V1 COMBO",
        "strength_model": "FLOAT",
        "strength_clip": "FLOAT",
        "prev_hooks": "HOOKS",
    }

    def __init__(
        self,
        node_id: str,
        ckpt_name: str | tuple[str, int],
        strength_model: float | tuple[str, int] = 1.0,
        strength_clip: float | tuple[str, int] = 1.0,
        prev_hooks: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.ckpt_name = ckpt_name
        self.strength_model = strength_model
        self.strength_clip = strength_clip
        if prev_hooks is not None:
            self.prev_hooks = prev_hooks
        self.__validate()

    def __validate(self) -> None:
        validate_not_empty("ckpt_name", self.ckpt_name)
        validate_range("strength_model", self.strength_model, -20.0, 20.0)
        validate_range("strength_clip", self.strength_clip, -20.0, 20.0)


class CreateHookModelAsLoraModelOnly(BaseNode):
    __slots__ = ("ckpt_name", "strength_model", "prev_hooks")
    CATEGORY = "advanced/hooks/create"
    OUTPUTS = {"HOOKS": 0}
    RETURN_TYPES = ("HOOKS",)
    INPUT_SOCKET_TYPES = {
        "ckpt_name": "V1 COMBO",
        "strength_model": "FLOAT",
        "prev_hooks": "HOOKS",
    }

    def __init__(
        self,
        node_id: str,
        ckpt_name: str | tuple[str, int],
        strength_model: float | tuple[str, int] = 1.0,
        prev_hooks: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.ckpt_name = ckpt_name
        self.strength_model = strength_model
        if prev_hooks is not None:
            self.prev_hooks = prev_hooks
        self.__validate()

    def __validate(self) -> None:
        validate_not_empty("ckpt_name", self.ckpt_name)
        validate_range("strength_model", self.strength_model, -20.0, 20.0)
