from typing import Literal, get_args

from ...base import BaseNode
from ...validation import validate_bool, validate_choice, validate_range


class SplitSigmas(BaseNode):
    __slots__ = ("sigmas", "step")
    CATEGORY = "model/sampling/sigmas"
    OUTPUTS = {"high_sigmas": 0, "low_sigmas": 1}
    RETURN_TYPES = ("SIGMAS", "SIGMAS")
    INPUT_SOCKET_TYPES = {"sigmas": "SIGMAS", "step": "INT"}

    def __init__(
        self, node_id: str, sigmas: tuple[str, int], step: int | tuple[str, int] = 0
    ) -> None:
        super().__init__(node_id=node_id)
        self.sigmas = sigmas
        self.step = step
        self.__validate()

    def __validate(self) -> None:
        validate_range("step", self.step, 0, 10_000)


class SplitSigmasDenoise(BaseNode):
    __slots__ = ("sigmas", "denoise")
    CATEGORY = "model/sampling/sigmas"
    OUTPUTS = {"high_sigmas": 0, "low_sigmas": 1}
    RETURN_TYPES = ("SIGMAS", "SIGMAS")
    INPUT_SOCKET_TYPES = {"sigmas": "SIGMAS", "denoise": "FLOAT"}

    def __init__(
        self,
        node_id: str,
        sigmas: tuple[str, int],
        denoise: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.sigmas = sigmas
        self.denoise = denoise
        self.__validate()

    def __validate(self) -> None:
        validate_range("denoise", self.denoise, 0.0, 1.0)


class FlipSigmas(BaseNode):
    __slots__ = ("sigmas",)
    CATEGORY = "model/sampling/sigmas"
    OUTPUTS = {"SIGMAS": 0}
    RETURN_TYPES = ("SIGMAS",)
    INPUT_SOCKET_TYPES = {"sigmas": "SIGMAS"}

    def __init__(self, node_id: str, sigmas: tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.sigmas = sigmas


class SetFirstSigma(BaseNode):
    __slots__ = ("sigmas", "sigma")
    CATEGORY = "model/sampling/sigmas"
    OUTPUTS = {"SIGMAS": 0}
    RETURN_TYPES = ("SIGMAS",)
    INPUT_SOCKET_TYPES = {"sigmas": "SIGMAS", "sigma": "FLOAT"}

    def __init__(
        self,
        node_id: str,
        sigmas: tuple[str, int],
        sigma: float | tuple[str, int] = 136.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.sigmas = sigmas
        self.sigma = sigma
        self.__validate()

    def __validate(self) -> None:
        validate_range("sigma", self.sigma, 0.0, 20000.0)


ExtendIntermediateSigmas_Spacings = Literal["linear", "cosine", "sine"]
EXTENDINTERMEDIATESIGMAS_SPACINGS = frozenset(
    get_args(ExtendIntermediateSigmas_Spacings)
)


class ExtendIntermediateSigmas(BaseNode):
    __slots__ = ("sigmas", "spacing", "steps", "start_at_sigma", "end_at_sigma")
    CATEGORY = "model/sampling/sigmas"
    OUTPUTS = {"SIGMAS": 0}
    RETURN_TYPES = ("SIGMAS",)
    INPUT_SOCKET_TYPES = {
        "sigmas": "SIGMAS",
        "steps": "INT",
        "start_at_sigma": "FLOAT",
        "end_at_sigma": "FLOAT",
        "spacing": "COMBO",
    }

    def __init__(
        self,
        node_id: str,
        sigmas: tuple[str, int],
        spacing: ExtendIntermediateSigmas_Spacings | tuple[str, int],
        steps: int | tuple[str, int] = 2,
        start_at_sigma: float | tuple[str, int] = -1.0,
        end_at_sigma: float | tuple[str, int] = 12.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.sigmas = sigmas
        self.spacing = spacing
        self.steps = steps
        self.start_at_sigma = start_at_sigma
        self.end_at_sigma = end_at_sigma
        self.__validate()

    def __validate(self) -> None:
        validate_choice("spacing", self.spacing, EXTENDINTERMEDIATESIGMAS_SPACINGS)
        validate_range("steps", self.steps, 1, 100)
        validate_range("start_at_sigma", self.start_at_sigma, -1.0, 20000.0)
        validate_range("end_at_sigma", self.end_at_sigma, 0.0, 20000.0)


class SamplingPercentToSigma(BaseNode):
    __slots__ = ("model", "sampling_percent", "return_actual_sigma")
    CATEGORY = "model/sampling/sigmas"
    OUTPUTS = {"sigma_value": 0}
    RETURN_TYPES = ("FLOAT",)
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "sampling_percent": "FLOAT",
        "return_actual_sigma": "BOOLEAN",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        sampling_percent: float | tuple[str, int] = 0.0,
        return_actual_sigma: bool | tuple[str, int] = False,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.sampling_percent = sampling_percent
        self.return_actual_sigma = return_actual_sigma
        self.__validate()

    def __validate(self) -> None:
        validate_range("sampling_percent", self.sampling_percent, 0.0, 1.0)
        validate_bool("return_actual_sigma", self.return_actual_sigma)


class ManualSigmas(BaseNode):
    __slots__ = ("sigmas",)
    CATEGORY = "model/sampling/sigmas"
    OUTPUTS = {"SIGMAS": 0}
    RETURN_TYPES = ("SIGMAS",)
    INPUT_SOCKET_TYPES = {"sigmas": "STRING"}

    def __init__(self, node_id: str, sigmas: str | tuple[str, int] = "1, 0.5") -> None:
        super().__init__(node_id=node_id)
        self.sigmas = sigmas
