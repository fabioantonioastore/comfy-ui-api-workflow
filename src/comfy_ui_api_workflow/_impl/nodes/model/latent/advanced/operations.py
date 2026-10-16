from ....base import BaseNode
from ....validation import validate_range


class LatentApplyOperation(BaseNode):
    __slots__ = ("samples", "operation")
    CATEGORY = "model/latent/advanced/operations"
    OUTPUTS = {"LATENT": 0}
    RETURN_TYPES = ("LATENT",)
    INPUT_SOCKET_TYPES = {"samples": "LATENT", "operation": "LATENT_OPERATION"}

    def __init__(
        self, node_id: str, samples: tuple[str, int], operation: tuple[str, int]
    ) -> None:
        super().__init__(node_id=node_id)
        self.samples = samples
        self.operation = operation


class LatentApplyOperationCFG(BaseNode):
    __slots__ = ("model", "operation", "start_percent", "end_percent")
    CATEGORY = "model/latent/advanced/operations"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "operation": "LATENT_OPERATION",
        "start_percent": "FLOAT",
        "end_percent": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        operation: tuple[str, int],
        start_percent: float | tuple[str, int] = 0.0,
        end_percent: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.operation = operation
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.__validate()

    def __validate(self) -> None:
        validate_range("start_percent", self.start_percent, 0.0, 1.0)
        validate_range("end_percent", self.end_percent, 0.0, 1.0)


class LatentOperationBlend(BaseNode):
    __slots__ = ("reference", "strength")
    CATEGORY = "model/latent/advanced/operations"
    OUTPUTS = {"LATENT_OPERATION": 0}
    RETURN_TYPES = ("LATENT_OPERATION",)
    INPUT_SOCKET_TYPES = {"reference": "LATENT", "strength": "FLOAT"}

    def __init__(
        self,
        node_id: str,
        reference: tuple[str, int],
        strength: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.reference = reference
        self.strength = strength
        self.__validate()

    def __validate(self) -> None:
        validate_range("strength", self.strength, 0.0, 1.0)


class LatentOperationTonemapReinhard(BaseNode):
    __slots__ = ("multiplier",)
    CATEGORY = "model/latent/advanced/operations"
    OUTPUTS = {"LATENT_OPERATION": 0}
    RETURN_TYPES = ("LATENT_OPERATION",)
    INPUT_SOCKET_TYPES = {"multiplier": "FLOAT"}

    def __init__(self, node_id: str, multiplier: float | tuple[str, int] = 1.0) -> None:
        super().__init__(node_id=node_id)
        self.multiplier = multiplier
        self.__validate()

    def __validate(self) -> None:
        validate_range("multiplier", self.multiplier, 0.0, 100.0)


class LatentOperationSharpen(BaseNode):
    __slots__ = ("sharpen_radius", "sigma", "alpha")
    CATEGORY = "model/latent/advanced/operations"
    OUTPUTS = {"LATENT_OPERATION": 0}
    RETURN_TYPES = ("LATENT_OPERATION",)
    INPUT_SOCKET_TYPES = {"sharpen_radius": "INT", "sigma": "FLOAT", "alpha": "FLOAT"}

    def __init__(
        self,
        node_id: str,
        sharpen_radius: int | tuple[str, int] = 9,
        sigma: float | tuple[str, int] = 1.0,
        alpha: float | tuple[str, int] = 0.1,
    ) -> None:
        super().__init__(node_id=node_id)
        self.sharpen_radius = sharpen_radius
        self.sigma = sigma
        self.alpha = alpha
        self.__validate()

    def __validate(self) -> None:
        validate_range("sharpen_radius", self.sharpen_radius, 1, 31)
        validate_range("sigma", self.sigma, 0.1, 10.0)
        validate_range("alpha", self.alpha, 0.0, 5.0)
