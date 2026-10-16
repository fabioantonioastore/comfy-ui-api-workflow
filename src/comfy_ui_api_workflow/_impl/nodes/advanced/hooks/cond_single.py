from typing import Literal, get_args

from ...base import BaseNode
from ...validation import validate_choice, validate_range

ConditioningSetProperties_Set_Cond_Areas = Literal["default", "mask bounds"]
CONDITIONINGSETPROPERTIES_SET_COND_AREAS = frozenset(
    get_args(ConditioningSetProperties_Set_Cond_Areas)
)


class ConditioningSetProperties(BaseNode):
    __slots__ = ("cond_NEW", "set_cond_area", "strength", "mask", "hooks", "timesteps")
    CATEGORY = "advanced/hooks/cond single"
    OUTPUTS = {"CONDITIONING": 0}
    RETURN_TYPES = ("CONDITIONING",)
    INPUT_SOCKET_TYPES = {
        "cond_NEW": "CONDITIONING",
        "strength": "FLOAT",
        "set_cond_area": "V1 COMBO",
        "mask": "MASK",
        "hooks": "HOOKS",
        "timesteps": "TIMESTEPS_RANGE",
    }

    def __init__(
        self,
        node_id: str,
        cond_NEW: tuple[str, int],
        set_cond_area: ConditioningSetProperties_Set_Cond_Areas | tuple[str, int],
        strength: float | tuple[str, int] = 1.0,
        mask: tuple[str, int] | None = None,
        hooks: tuple[str, int] | None = None,
        timesteps: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.cond_NEW = cond_NEW
        self.set_cond_area = set_cond_area
        self.strength = strength
        if mask is not None:
            self.mask = mask
        if hooks is not None:
            self.hooks = hooks
        if timesteps is not None:
            self.timesteps = timesteps
        self.__validate()

    def __validate(self) -> None:
        validate_choice(
            "set_cond_area",
            self.set_cond_area,
            CONDITIONINGSETPROPERTIES_SET_COND_AREAS,
        )
        validate_range("strength", self.strength, 0.0, 10.0)


ConditioningSetPropertiesAndCombine_Set_Cond_Areas = Literal["default", "mask bounds"]
CONDITIONINGSETPROPERTIESANDCOMBINE_SET_COND_AREAS = frozenset(
    get_args(ConditioningSetPropertiesAndCombine_Set_Cond_Areas)
)


class ConditioningSetPropertiesAndCombine(BaseNode):
    __slots__ = (
        "cond",
        "cond_NEW",
        "set_cond_area",
        "strength",
        "mask",
        "hooks",
        "timesteps",
    )
    CATEGORY = "advanced/hooks/cond single"
    OUTPUTS = {"CONDITIONING": 0}
    RETURN_TYPES = ("CONDITIONING",)
    INPUT_SOCKET_TYPES = {
        "cond": "CONDITIONING",
        "cond_NEW": "CONDITIONING",
        "strength": "FLOAT",
        "set_cond_area": "V1 COMBO",
        "mask": "MASK",
        "hooks": "HOOKS",
        "timesteps": "TIMESTEPS_RANGE",
    }

    def __init__(
        self,
        node_id: str,
        cond: tuple[str, int],
        cond_NEW: tuple[str, int],
        set_cond_area: (
            ConditioningSetPropertiesAndCombine_Set_Cond_Areas | tuple[str, int]
        ),
        strength: float | tuple[str, int] = 1.0,
        mask: tuple[str, int] | None = None,
        hooks: tuple[str, int] | None = None,
        timesteps: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.cond = cond
        self.cond_NEW = cond_NEW
        self.set_cond_area = set_cond_area
        self.strength = strength
        if mask is not None:
            self.mask = mask
        if hooks is not None:
            self.hooks = hooks
        if timesteps is not None:
            self.timesteps = timesteps
        self.__validate()

    def __validate(self) -> None:
        validate_choice(
            "set_cond_area",
            self.set_cond_area,
            CONDITIONINGSETPROPERTIESANDCOMBINE_SET_COND_AREAS,
        )
        validate_range("strength", self.strength, 0.0, 10.0)


class ConditioningSetDefaultCombine(BaseNode):
    __slots__ = ("cond", "cond_DEFAULT", "hooks")
    CATEGORY = "advanced/hooks/cond single"
    OUTPUTS = {"CONDITIONING": 0}
    RETURN_TYPES = ("CONDITIONING",)
    INPUT_SOCKET_TYPES = {
        "cond": "CONDITIONING",
        "cond_DEFAULT": "CONDITIONING",
        "hooks": "HOOKS",
    }

    def __init__(
        self,
        node_id: str,
        cond: tuple[str, int],
        cond_DEFAULT: tuple[str, int],
        hooks: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.cond = cond
        self.cond_DEFAULT = cond_DEFAULT
        if hooks is not None:
            self.hooks = hooks
