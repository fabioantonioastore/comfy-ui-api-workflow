from typing import Literal, get_args

from ..base import BaseNode
from ..validation import validate_bool, validate_choice, validate_range

ModelComputeDtype_Dtypes = Literal["default", "fp32", "fp16", "bf16"]
MODELCOMPUTEDTYPE_DTYPES = frozenset(get_args(ModelComputeDtype_Dtypes))


class ModelComputeDtype(BaseNode):
    __slots__ = ("model", "dtype")
    CATEGORY = "advanced/debug"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {"model": "MODEL", "dtype": "V1 COMBO"}

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        dtype: ModelComputeDtype_Dtypes | tuple[str, int],
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.dtype = dtype
        self.__validate()

    def __validate(self) -> None:
        validate_choice("dtype", self.dtype, MODELCOMPUTEDTYPE_DTYPES)


class EasyCache(BaseNode):
    __slots__ = ("model", "reuse_threshold", "start_percent", "end_percent", "verbose")
    CATEGORY = "advanced/debug"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "reuse_threshold": "FLOAT",
        "start_percent": "FLOAT",
        "end_percent": "FLOAT",
        "verbose": "BOOLEAN",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        reuse_threshold: float | tuple[str, int] = 0.2,
        start_percent: float | tuple[str, int] = 0.15,
        end_percent: float | tuple[str, int] = 0.95,
        verbose: bool | tuple[str, int] = False,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.reuse_threshold = reuse_threshold
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.verbose = verbose
        self.__validate()

    def __validate(self) -> None:
        validate_range("reuse_threshold", self.reuse_threshold, 0.0, 3.0)
        validate_range("start_percent", self.start_percent, 0.0, 1.0)
        validate_range("end_percent", self.end_percent, 0.0, 1.0)
        validate_bool("verbose", self.verbose)


class LazyCache(BaseNode):
    __slots__ = ("model", "reuse_threshold", "start_percent", "end_percent", "verbose")
    CATEGORY = "advanced/debug"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "reuse_threshold": "FLOAT",
        "start_percent": "FLOAT",
        "end_percent": "FLOAT",
        "verbose": "BOOLEAN",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        reuse_threshold: float | tuple[str, int] = 0.2,
        start_percent: float | tuple[str, int] = 0.15,
        end_percent: float | tuple[str, int] = 0.95,
        verbose: bool | tuple[str, int] = False,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.reuse_threshold = reuse_threshold
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.verbose = verbose
        self.__validate()

    def __validate(self) -> None:
        validate_range("reuse_threshold", self.reuse_threshold, 0.0, 3.0)
        validate_range("start_percent", self.start_percent, 0.0, 1.0)
        validate_range("end_percent", self.end_percent, 0.0, 1.0)
        validate_bool("verbose", self.verbose)
