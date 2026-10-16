from typing import Literal, get_args

from ...base import BaseNode
from ...validation import validate_choice, validate_range


class ConditioningAverage(BaseNode):
    __slots__ = ("conditioning_to", "conditioning_from", "conditioning_to_strength")
    CATEGORY = "model/conditioning/transform"
    OUTPUTS = {"CONDITIONING": 0}
    RETURN_TYPES = ("CONDITIONING",)
    INPUT_SOCKET_TYPES = {
        "conditioning_to": "CONDITIONING",
        "conditioning_from": "CONDITIONING",
        "conditioning_to_strength": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        conditioning_to: tuple[str, int],
        conditioning_from: tuple[str, int],
        conditioning_to_strength: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.conditioning_to = conditioning_to
        self.conditioning_from = conditioning_from
        self.conditioning_to_strength = conditioning_to_strength
        self.__validate()

    def __validate(self) -> None:
        validate_range(
            "conditioning_to_strength", self.conditioning_to_strength, 0.0, 1.0
        )


class ConditioningCombine(BaseNode):
    __slots__ = ("conditioning_1", "conditioning_2")
    CATEGORY = "model/conditioning/transform"
    OUTPUTS = {"CONDITIONING": 0}
    RETURN_TYPES = ("CONDITIONING",)
    INPUT_SOCKET_TYPES = {
        "conditioning_1": "CONDITIONING",
        "conditioning_2": "CONDITIONING",
    }

    def __init__(
        self,
        node_id: str,
        conditioning_1: tuple[str, int],
        conditioning_2: tuple[str, int],
    ) -> None:
        super().__init__(node_id=node_id)
        self.conditioning_1 = conditioning_1
        self.conditioning_2 = conditioning_2


class ConditioningConcat(BaseNode):
    __slots__ = ("conditioning_to", "conditioning_from")
    CATEGORY = "model/conditioning/transform"
    OUTPUTS = {"CONDITIONING": 0}
    RETURN_TYPES = ("CONDITIONING",)
    INPUT_SOCKET_TYPES = {
        "conditioning_to": "CONDITIONING",
        "conditioning_from": "CONDITIONING",
    }

    def __init__(
        self,
        node_id: str,
        conditioning_to: tuple[str, int],
        conditioning_from: tuple[str, int],
    ) -> None:
        super().__init__(node_id=node_id)
        self.conditioning_to = conditioning_to
        self.conditioning_from = conditioning_from


class ConditioningMultiply(BaseNode):
    __slots__ = ("conditioning", "multiplier")
    CATEGORY = "model/conditioning/transform"
    OUTPUTS = {"CONDITIONING": 0}
    RETURN_TYPES = ("CONDITIONING",)
    INPUT_SOCKET_TYPES = {"conditioning": "CONDITIONING", "multiplier": "FLOAT"}

    def __init__(
        self,
        node_id: str,
        conditioning: tuple[str, int],
        multiplier: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.conditioning = conditioning
        self.multiplier = multiplier
        self.__validate()

    def __validate(self) -> None:
        validate_range("multiplier", self.multiplier, -100.0, 100.0)


class ConditioningSetArea(BaseNode):
    __slots__ = ("conditioning", "width", "height", "x", "y", "strength")
    CATEGORY = "model/conditioning/transform"
    OUTPUTS = {"CONDITIONING": 0}
    RETURN_TYPES = ("CONDITIONING",)
    INPUT_SOCKET_TYPES = {
        "conditioning": "CONDITIONING",
        "width": "INT",
        "height": "INT",
        "x": "INT",
        "y": "INT",
        "strength": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        conditioning: tuple[str, int],
        width: int | tuple[str, int] = 64,
        height: int | tuple[str, int] = 64,
        x: int | tuple[str, int] = 0,
        y: int | tuple[str, int] = 0,
        strength: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.conditioning = conditioning
        self.width = width
        self.height = height
        self.x = x
        self.y = y
        self.strength = strength
        self.__validate()

    def __validate(self) -> None:
        validate_range("width", self.width, 64, 16_384)
        validate_range("height", self.height, 64, 16_384)
        validate_range("x", self.x, 0, 16_384)
        validate_range("y", self.y, 0, 16_384)
        validate_range("strength", self.strength, 0.0, 10.0)


class ConditioningSetAreaPercentage(BaseNode):
    __slots__ = ("conditioning", "width", "height", "x", "y", "strength")
    CATEGORY = "model/conditioning/transform"
    OUTPUTS = {"CONDITIONING": 0}
    RETURN_TYPES = ("CONDITIONING",)
    INPUT_SOCKET_TYPES = {
        "conditioning": "CONDITIONING",
        "width": "FLOAT",
        "height": "FLOAT",
        "x": "FLOAT",
        "y": "FLOAT",
        "strength": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        conditioning: tuple[str, int],
        width: float | tuple[str, int] = 1.0,
        height: float | tuple[str, int] = 1.0,
        x: float | tuple[str, int] = 0.0,
        y: float | tuple[str, int] = 0.0,
        strength: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.conditioning = conditioning
        self.width = width
        self.height = height
        self.x = x
        self.y = y
        self.strength = strength
        self.__validate()

    def __validate(self) -> None:
        validate_range("width", self.width, 0, 1.0)
        validate_range("height", self.height, 0, 1.0)
        validate_range("x", self.x, 0, 1.0)
        validate_range("y", self.y, 0, 1.0)
        validate_range("strength", self.strength, 0.0, 10.0)


class ConditioningSetAreaStrength(BaseNode):
    __slots__ = ("conditioning", "strength")
    CATEGORY = "model/conditioning/transform"
    OUTPUTS = {"CONDITIONING": 0}
    RETURN_TYPES = ("CONDITIONING",)
    INPUT_SOCKET_TYPES = {"conditioning": "CONDITIONING", "strength": "FLOAT"}

    def __init__(
        self,
        node_id: str,
        conditioning: tuple[str, int],
        strength: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.conditioning = conditioning
        self.strength = strength
        self.__validate()

    def __validate(self) -> None:
        validate_range("strength", self.strength, 0.0, 10.0)


ConditioningSetMask_Set_Cond_Areas = Literal["default", "mask bounds"]
CONDITIONINGSETMASK_SET_COND_AREAS = frozenset(
    get_args(ConditioningSetMask_Set_Cond_Areas)
)


class ConditioningSetMask(BaseNode):
    __slots__ = ("conditioning", "mask", "set_cond_area", "strength")
    CATEGORY = "model/conditioning/transform"
    OUTPUTS = {"CONDITIONING": 0}
    RETURN_TYPES = ("CONDITIONING",)
    INPUT_SOCKET_TYPES = {
        "conditioning": "CONDITIONING",
        "mask": "MASK",
        "strength": "FLOAT",
        "set_cond_area": "V1 COMBO",
    }

    def __init__(
        self,
        node_id: str,
        conditioning: tuple[str, int],
        mask: tuple[str, int],
        set_cond_area: ConditioningSetMask_Set_Cond_Areas | tuple[str, int],
        strength: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.conditioning = conditioning
        self.mask = mask
        self.set_cond_area = set_cond_area
        self.strength = strength
        self.__validate()

    def __validate(self) -> None:
        validate_choice(
            "set_cond_area", self.set_cond_area, CONDITIONINGSETMASK_SET_COND_AREAS
        )
        validate_range("strength", self.strength, 0.0, 10.0)


class ConditioningZeroOut(BaseNode):
    __slots__ = ("conditioning",)
    CATEGORY = "model/conditioning/transform"
    OUTPUTS = {"CONDITIONING": 0}
    RETURN_TYPES = ("CONDITIONING",)
    INPUT_SOCKET_TYPES = {"conditioning": "CONDITIONING"}

    def __init__(self, node_id: str, conditioning: tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.conditioning = conditioning


class ConditioningSetTimestepRange(BaseNode):
    __slots__ = ("conditioning", "start", "end")
    CATEGORY = "model/conditioning/transform"
    OUTPUTS = {"CONDITIONING": 0}
    RETURN_TYPES = ("CONDITIONING",)
    INPUT_SOCKET_TYPES = {
        "conditioning": "CONDITIONING",
        "start": "FLOAT",
        "end": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        conditioning: tuple[str, int],
        start: float | tuple[str, int] = 0.0,
        end: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.conditioning = conditioning
        self.start = start
        self.end = end
        self.__validate()

    def __validate(self) -> None:
        validate_range("start", self.start, 0.0, 1.0)
        validate_range("end", self.end, 0.0, 1.0)


class ConditioningSetAreaPercentageVideo(BaseNode):
    __slots__ = (
        "conditioning",
        "width",
        "height",
        "temporal",
        "x",
        "y",
        "z",
        "strength",
    )
    CATEGORY = "model/conditioning/transform"
    OUTPUTS = {"CONDITIONING": 0}
    RETURN_TYPES = ("CONDITIONING",)
    INPUT_SOCKET_TYPES = {
        "conditioning": "CONDITIONING",
        "width": "FLOAT",
        "height": "FLOAT",
        "temporal": "FLOAT",
        "x": "FLOAT",
        "y": "FLOAT",
        "z": "FLOAT",
        "strength": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        conditioning: tuple[str, int],
        width: float | tuple[str, int] = 1.0,
        height: float | tuple[str, int] = 1.0,
        temporal: float | tuple[str, int] = 1.0,
        x: float | tuple[str, int] = 0.0,
        y: float | tuple[str, int] = 0.0,
        z: float | tuple[str, int] = 0.0,
        strength: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.conditioning = conditioning
        self.width = width
        self.height = height
        self.temporal = temporal
        self.x = x
        self.y = y
        self.z = z
        self.strength = strength
        self.__validate()

    def __validate(self) -> None:
        validate_range("width", self.width, 0, 1.0)
        validate_range("height", self.height, 0, 1.0)
        validate_range("temporal", self.temporal, 0, 1.0)
        validate_range("x", self.x, 0, 1.0)
        validate_range("y", self.y, 0, 1.0)
        validate_range("z", self.z, 0, 1.0)
        validate_range("strength", self.strength, 0.0, 10.0)
