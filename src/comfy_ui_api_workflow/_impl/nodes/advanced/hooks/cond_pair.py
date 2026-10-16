from typing import Literal, get_args

from ...base import BaseNode
from ...validation import validate_choice, validate_range

PairConditioningSetProperties_Set_Cond_Areas = Literal["default", "mask bounds"]
PAIRCONDITIONINGSETPROPERTIES_SET_COND_AREAS = frozenset(
    get_args(PairConditioningSetProperties_Set_Cond_Areas)
)


class PairConditioningSetProperties(BaseNode):
    __slots__ = (
        "positive_NEW",
        "negative_NEW",
        "set_cond_area",
        "strength",
        "mask",
        "hooks",
        "timesteps",
    )
    CATEGORY = "advanced/hooks/cond pair"
    OUTPUTS = {"positive": 0, "negative": 1}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    INPUT_SOCKET_TYPES = {
        "positive_NEW": "CONDITIONING",
        "negative_NEW": "CONDITIONING",
        "strength": "FLOAT",
        "set_cond_area": "V1 COMBO",
        "mask": "MASK",
        "hooks": "HOOKS",
        "timesteps": "TIMESTEPS_RANGE",
    }

    def __init__(
        self,
        node_id: str,
        positive_NEW: tuple[str, int],
        negative_NEW: tuple[str, int],
        set_cond_area: PairConditioningSetProperties_Set_Cond_Areas | tuple[str, int],
        strength: float | tuple[str, int] = 1.0,
        mask: tuple[str, int] | None = None,
        hooks: tuple[str, int] | None = None,
        timesteps: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.positive_NEW = positive_NEW
        self.negative_NEW = negative_NEW
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
            PAIRCONDITIONINGSETPROPERTIES_SET_COND_AREAS,
        )
        validate_range("strength", self.strength, 0.0, 10.0)


PairConditioningSetPropertiesAndCombine_Set_Cond_Areas = Literal[
    "default", "mask bounds"
]
PAIRCONDITIONINGSETPROPERTIESANDCOMBINE_SET_COND_AREAS = frozenset(
    get_args(PairConditioningSetPropertiesAndCombine_Set_Cond_Areas)
)


class PairConditioningSetPropertiesAndCombine(BaseNode):
    __slots__ = (
        "positive",
        "negative",
        "positive_NEW",
        "negative_NEW",
        "set_cond_area",
        "strength",
        "mask",
        "hooks",
        "timesteps",
    )
    CATEGORY = "advanced/hooks/cond pair"
    OUTPUTS = {"positive": 0, "negative": 1}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    INPUT_SOCKET_TYPES = {
        "positive": "CONDITIONING",
        "negative": "CONDITIONING",
        "positive_NEW": "CONDITIONING",
        "negative_NEW": "CONDITIONING",
        "strength": "FLOAT",
        "set_cond_area": "V1 COMBO",
        "mask": "MASK",
        "hooks": "HOOKS",
        "timesteps": "TIMESTEPS_RANGE",
    }

    def __init__(
        self,
        node_id: str,
        positive: tuple[str, int],
        negative: tuple[str, int],
        positive_NEW: tuple[str, int],
        negative_NEW: tuple[str, int],
        set_cond_area: (
            PairConditioningSetPropertiesAndCombine_Set_Cond_Areas | tuple[str, int]
        ),
        strength: float | tuple[str, int] = 1.0,
        mask: tuple[str, int] | None = None,
        hooks: tuple[str, int] | None = None,
        timesteps: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.positive = positive
        self.negative = negative
        self.positive_NEW = positive_NEW
        self.negative_NEW = negative_NEW
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
            PAIRCONDITIONINGSETPROPERTIESANDCOMBINE_SET_COND_AREAS,
        )
        validate_range("strength", self.strength, 0.0, 10.0)


class PairConditioningSetDefaultCombine(BaseNode):
    __slots__ = (
        "positive",
        "negative",
        "positive_DEFAULT",
        "negative_DEFAULT",
        "hooks",
    )
    CATEGORY = "advanced/hooks/cond pair"
    OUTPUTS = {"positive": 0, "negative": 1}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    INPUT_SOCKET_TYPES = {
        "positive": "CONDITIONING",
        "negative": "CONDITIONING",
        "positive_DEFAULT": "CONDITIONING",
        "negative_DEFAULT": "CONDITIONING",
        "hooks": "HOOKS",
    }

    def __init__(
        self,
        node_id: str,
        positive: tuple[str, int],
        negative: tuple[str, int],
        positive_DEFAULT: tuple[str, int],
        negative_DEFAULT: tuple[str, int],
        hooks: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.positive = positive
        self.negative = negative
        self.positive_DEFAULT = positive_DEFAULT
        self.negative_DEFAULT = negative_DEFAULT
        if hooks is not None:
            self.hooks = hooks


class PairConditioningCombine(BaseNode):
    __slots__ = ("positive_A", "negative_A", "positive_B", "negative_B")
    CATEGORY = "advanced/hooks/cond pair"
    OUTPUTS = {"positive": 0, "negative": 1}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    INPUT_SOCKET_TYPES = {
        "positive_A": "CONDITIONING",
        "negative_A": "CONDITIONING",
        "positive_B": "CONDITIONING",
        "negative_B": "CONDITIONING",
    }

    def __init__(
        self,
        node_id: str,
        positive_A: tuple[str, int],
        negative_A: tuple[str, int],
        positive_B: tuple[str, int],
        negative_B: tuple[str, int],
    ) -> None:
        super().__init__(node_id=node_id)
        self.positive_A = positive_A
        self.negative_A = negative_A
        self.positive_B = positive_B
        self.negative_B = negative_B
