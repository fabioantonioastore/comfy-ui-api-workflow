from typing import Literal, get_args

from ....base import BaseNode
from ....validation import validate_choice, validate_range
from .....lazy import submodules_of


class LatentAdd(BaseNode):
    __slots__ = ("samples1", "samples2")
    CATEGORY = "model/latent/advanced"
    OUTPUTS = {"LATENT": 0}
    RETURN_TYPES = ("LATENT",)
    INPUT_SOCKET_TYPES = {"samples1": "LATENT", "samples2": "LATENT"}

    def __init__(
        self, node_id: str, samples1: tuple[str, int], samples2: tuple[str, int]
    ) -> None:
        super().__init__(node_id=node_id)
        self.samples1 = samples1
        self.samples2 = samples2


class LatentSubtract(BaseNode):
    __slots__ = ("samples1", "samples2")
    CATEGORY = "model/latent/advanced"
    OUTPUTS = {"LATENT": 0}
    RETURN_TYPES = ("LATENT",)
    INPUT_SOCKET_TYPES = {"samples1": "LATENT", "samples2": "LATENT"}

    def __init__(
        self, node_id: str, samples1: tuple[str, int], samples2: tuple[str, int]
    ) -> None:
        super().__init__(node_id=node_id)
        self.samples1 = samples1
        self.samples2 = samples2


class LatentMultiply(BaseNode):
    __slots__ = ("samples", "multiplier")
    CATEGORY = "model/latent/advanced"
    OUTPUTS = {"LATENT": 0}
    RETURN_TYPES = ("LATENT",)
    INPUT_SOCKET_TYPES = {"samples": "LATENT", "multiplier": "FLOAT"}

    def __init__(
        self,
        node_id: str,
        samples: tuple[str, int],
        multiplier: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.samples = samples
        self.multiplier = multiplier
        self.__validate()

    def __validate(self) -> None:
        validate_range("multiplier", self.multiplier, -10.0, 10.0)


class LatentInterpolate(BaseNode):
    __slots__ = ("samples1", "samples2", "ratio")
    CATEGORY = "model/latent/advanced"
    OUTPUTS = {"LATENT": 0}
    RETURN_TYPES = ("LATENT",)
    INPUT_SOCKET_TYPES = {"samples1": "LATENT", "samples2": "LATENT", "ratio": "FLOAT"}

    def __init__(
        self,
        node_id: str,
        samples1: tuple[str, int],
        samples2: tuple[str, int],
        ratio: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.samples1 = samples1
        self.samples2 = samples2
        self.ratio = ratio
        self.__validate()

    def __validate(self) -> None:
        validate_range("ratio", self.ratio, 0.0, 1.0)


LatentConcat_Dims = Literal["x", "-x", "y", "-y", "t", "-t"]
LATENTCONCAT_DIMS = frozenset(get_args(LatentConcat_Dims))


class LatentConcat(BaseNode):
    __slots__ = ("samples1", "samples2", "dim")
    CATEGORY = "model/latent/advanced"
    OUTPUTS = {"LATENT": 0}
    RETURN_TYPES = ("LATENT",)
    INPUT_SOCKET_TYPES = {"samples1": "LATENT", "samples2": "LATENT", "dim": "COMBO"}

    def __init__(
        self,
        node_id: str,
        samples1: tuple[str, int],
        samples2: tuple[str, int],
        dim: LatentConcat_Dims | tuple[str, int],
    ) -> None:
        super().__init__(node_id=node_id)
        self.samples1 = samples1
        self.samples2 = samples2
        self.dim = dim
        self.__validate()

    def __validate(self) -> None:
        validate_choice("dim", self.dim, LATENTCONCAT_DIMS)


LatentCut_Dims = Literal["x", "y", "t"]
LATENTCUT_DIMS = frozenset(get_args(LatentCut_Dims))


class LatentCut(BaseNode):
    __slots__ = ("samples", "dim", "index", "amount")
    CATEGORY = "model/latent/advanced"
    OUTPUTS = {"LATENT": 0}
    RETURN_TYPES = ("LATENT",)
    INPUT_SOCKET_TYPES = {
        "samples": "LATENT",
        "dim": "COMBO",
        "index": "INT",
        "amount": "INT",
    }

    def __init__(
        self,
        node_id: str,
        samples: tuple[str, int],
        dim: LatentCut_Dims | tuple[str, int],
        index: int | tuple[str, int] = 0,
        amount: int | tuple[str, int] = 1,
    ) -> None:
        super().__init__(node_id=node_id)
        self.samples = samples
        self.dim = dim
        self.index = index
        self.amount = amount
        self.__validate()

    def __validate(self) -> None:
        validate_choice("dim", self.dim, LATENTCUT_DIMS)
        validate_range("index", self.index, -16_384, 16_384)
        validate_range("amount", self.amount, 1, 16_384)


LatentCutToBatch_Dims = Literal["t", "x", "y"]
LATENTCUTTOBATCH_DIMS = frozenset(get_args(LatentCutToBatch_Dims))


class LatentCutToBatch(BaseNode):
    __slots__ = ("samples", "dim", "slice_size")
    CATEGORY = "model/latent/advanced"
    OUTPUTS = {"LATENT": 0}
    RETURN_TYPES = ("LATENT",)
    INPUT_SOCKET_TYPES = {"samples": "LATENT", "dim": "COMBO", "slice_size": "INT"}

    def __init__(
        self,
        node_id: str,
        samples: tuple[str, int],
        dim: LatentCutToBatch_Dims | tuple[str, int],
        slice_size: int | tuple[str, int] = 1,
    ) -> None:
        super().__init__(node_id=node_id)
        self.samples = samples
        self.dim = dim
        self.slice_size = slice_size
        self.__validate()

    def __validate(self) -> None:
        validate_choice("dim", self.dim, LATENTCUTTOBATCH_DIMS)
        validate_range("slice_size", self.slice_size, 1, 16_384)


LatentBatchSeedBehavior_Seed_Behaviors = Literal["random", "fixed"]
LATENTBATCHSEEDBEHAVIOR_SEED_BEHAVIORS = frozenset(
    get_args(LatentBatchSeedBehavior_Seed_Behaviors)
)


class LatentBatchSeedBehavior(BaseNode):
    __slots__ = ("samples", "seed_behavior")
    CATEGORY = "model/latent/advanced"
    OUTPUTS = {"LATENT": 0}
    RETURN_TYPES = ("LATENT",)
    INPUT_SOCKET_TYPES = {"samples": "LATENT", "seed_behavior": "COMBO"}

    def __init__(
        self,
        node_id: str,
        samples: tuple[str, int],
        seed_behavior: (
            LatentBatchSeedBehavior_Seed_Behaviors | tuple[str, int]
        ) = "fixed",
    ) -> None:
        super().__init__(node_id=node_id)
        self.samples = samples
        self.seed_behavior = seed_behavior
        self.__validate()

    def __validate(self) -> None:
        validate_choice(
            "seed_behavior", self.seed_behavior, LATENTBATCHSEEDBEHAVIOR_SEED_BEHAVIORS
        )


__getattr__ = submodules_of(__name__)
