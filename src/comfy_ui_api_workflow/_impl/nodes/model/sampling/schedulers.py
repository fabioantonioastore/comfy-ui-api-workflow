from typing import Literal, get_args

from ...base import BaseNode
from ...model.sampling import KSampler_Schedulers, KSAMPLER_SCHEDULERS
from ...validation import validate_bool, validate_choice, validate_range


class BasicScheduler(BaseNode):
    __slots__ = ("model", "scheduler", "steps", "denoise")
    CATEGORY = "model/sampling/schedulers"
    OUTPUTS = {"SIGMAS": 0}
    RETURN_TYPES = ("SIGMAS",)
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "scheduler": "COMBO",
        "steps": "INT",
        "denoise": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        scheduler: KSampler_Schedulers | tuple[str, int],
        steps: int | tuple[str, int] = 20,
        denoise: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.scheduler = scheduler
        self.steps = steps
        self.denoise = denoise
        self.__validate()

    def __validate(self) -> None:
        validate_choice("scheduler", self.scheduler, KSAMPLER_SCHEDULERS)
        validate_range("steps", self.steps, 1, 10_000)
        validate_range("denoise", self.denoise, 0.0, 1.0)


class KarrasScheduler(BaseNode):
    __slots__ = ("steps", "sigma_max", "sigma_min", "rho")
    CATEGORY = "model/sampling/schedulers"
    OUTPUTS = {"SIGMAS": 0}
    RETURN_TYPES = ("SIGMAS",)
    INPUT_SOCKET_TYPES = {
        "steps": "INT",
        "sigma_max": "FLOAT",
        "sigma_min": "FLOAT",
        "rho": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        steps: int | tuple[str, int] = 20,
        sigma_max: float | tuple[str, int] = 14.614642,
        sigma_min: float | tuple[str, int] = 0.0291675,
        rho: float | tuple[str, int] = 7.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.steps = steps
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.rho = rho
        self.__validate()

    def __validate(self) -> None:
        validate_range("steps", self.steps, 1, 10_000)
        validate_range("sigma_max", self.sigma_max, 0.0, 5000.0)
        validate_range("sigma_min", self.sigma_min, 0.0, 5000.0)
        validate_range("rho", self.rho, 0.0, 100.0)


class ExponentialScheduler(BaseNode):
    __slots__ = ("steps", "sigma_max", "sigma_min")
    CATEGORY = "model/sampling/schedulers"
    OUTPUTS = {"SIGMAS": 0}
    RETURN_TYPES = ("SIGMAS",)
    INPUT_SOCKET_TYPES = {"steps": "INT", "sigma_max": "FLOAT", "sigma_min": "FLOAT"}

    def __init__(
        self,
        node_id: str,
        steps: int | tuple[str, int] = 20,
        sigma_max: float | tuple[str, int] = 14.614642,
        sigma_min: float | tuple[str, int] = 0.0291675,
    ) -> None:
        super().__init__(node_id=node_id)
        self.steps = steps
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.__validate()

    def __validate(self) -> None:
        validate_range("steps", self.steps, 1, 10_000)
        validate_range("sigma_max", self.sigma_max, 0.0, 5000.0)
        validate_range("sigma_min", self.sigma_min, 0.0, 5000.0)


class PolyexponentialScheduler(BaseNode):
    __slots__ = ("steps", "sigma_max", "sigma_min", "rho")
    CATEGORY = "model/sampling/schedulers"
    OUTPUTS = {"SIGMAS": 0}
    RETURN_TYPES = ("SIGMAS",)
    INPUT_SOCKET_TYPES = {
        "steps": "INT",
        "sigma_max": "FLOAT",
        "sigma_min": "FLOAT",
        "rho": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        steps: int | tuple[str, int] = 20,
        sigma_max: float | tuple[str, int] = 14.614642,
        sigma_min: float | tuple[str, int] = 0.0291675,
        rho: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.steps = steps
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.rho = rho
        self.__validate()

    def __validate(self) -> None:
        validate_range("steps", self.steps, 1, 10_000)
        validate_range("sigma_max", self.sigma_max, 0.0, 5000.0)
        validate_range("sigma_min", self.sigma_min, 0.0, 5000.0)
        validate_range("rho", self.rho, 0.0, 100.0)


class LaplaceScheduler(BaseNode):
    __slots__ = ("steps", "sigma_max", "sigma_min", "mu", "beta")
    CATEGORY = "model/sampling/schedulers"
    OUTPUTS = {"SIGMAS": 0}
    RETURN_TYPES = ("SIGMAS",)
    INPUT_SOCKET_TYPES = {
        "steps": "INT",
        "sigma_max": "FLOAT",
        "sigma_min": "FLOAT",
        "mu": "FLOAT",
        "beta": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        steps: int | tuple[str, int] = 20,
        sigma_max: float | tuple[str, int] = 14.614642,
        sigma_min: float | tuple[str, int] = 0.0291675,
        mu: float | tuple[str, int] = 0.0,
        beta: float | tuple[str, int] = 0.5,
    ) -> None:
        super().__init__(node_id=node_id)
        self.steps = steps
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.mu = mu
        self.beta = beta
        self.__validate()

    def __validate(self) -> None:
        validate_range("steps", self.steps, 1, 10_000)
        validate_range("sigma_max", self.sigma_max, 0.0, 5000.0)
        validate_range("sigma_min", self.sigma_min, 0.0, 5000.0)
        validate_range("mu", self.mu, -10.0, 10.0)
        validate_range("beta", self.beta, 0.0, 10.0)


class VPScheduler(BaseNode):
    __slots__ = ("steps", "beta_d", "beta_min", "eps_s")
    CATEGORY = "model/sampling/schedulers"
    OUTPUTS = {"SIGMAS": 0}
    RETURN_TYPES = ("SIGMAS",)
    INPUT_SOCKET_TYPES = {
        "steps": "INT",
        "beta_d": "FLOAT",
        "beta_min": "FLOAT",
        "eps_s": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        steps: int | tuple[str, int] = 20,
        beta_d: float | tuple[str, int] = 19.9,
        beta_min: float | tuple[str, int] = 0.1,
        eps_s: float | tuple[str, int] = 0.001,
    ) -> None:
        super().__init__(node_id=node_id)
        self.steps = steps
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.eps_s = eps_s
        self.__validate()

    def __validate(self) -> None:
        validate_range("steps", self.steps, 1, 10_000)
        validate_range("beta_d", self.beta_d, 0.0, 5000.0)
        validate_range("beta_min", self.beta_min, 0.0, 5000.0)
        validate_range("eps_s", self.eps_s, 0.0, 1.0)


class BetaSamplingScheduler(BaseNode):
    __slots__ = ("model", "steps", "alpha", "beta")
    CATEGORY = "model/sampling/schedulers"
    OUTPUTS = {"SIGMAS": 0}
    RETURN_TYPES = ("SIGMAS",)
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "steps": "INT",
        "alpha": "FLOAT",
        "beta": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        steps: int | tuple[str, int] = 20,
        alpha: float | tuple[str, int] = 0.6,
        beta: float | tuple[str, int] = 0.6,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.steps = steps
        self.alpha = alpha
        self.beta = beta
        self.__validate()

    def __validate(self) -> None:
        validate_range("steps", self.steps, 1, 10_000)
        validate_range("alpha", self.alpha, 0.0, 50.0)
        validate_range("beta", self.beta, 0.0, 50.0)


class SDTurboScheduler(BaseNode):
    __slots__ = ("model", "steps", "denoise")
    CATEGORY = "model/sampling/schedulers"
    OUTPUTS = {"SIGMAS": 0}
    RETURN_TYPES = ("SIGMAS",)
    INPUT_SOCKET_TYPES = {"model": "MODEL", "steps": "INT", "denoise": "FLOAT"}

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        steps: int | tuple[str, int] = 1,
        denoise: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.steps = steps
        self.denoise = denoise
        self.__validate()

    def __validate(self) -> None:
        validate_range("steps", self.steps, 1, 10)
        validate_range("denoise", self.denoise, 0, 1.0)


class Ideogram4Scheduler(BaseNode):
    __slots__ = ("steps", "width", "height", "mu", "std")
    CATEGORY = "model/sampling/schedulers"
    OUTPUTS = {"SIGMAS": 0}
    RETURN_TYPES = ("SIGMAS",)
    INPUT_SOCKET_TYPES = {
        "steps": "INT",
        "width": "INT",
        "height": "INT",
        "mu": "FLOAT",
        "std": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        steps: int | tuple[str, int] = 20,
        width: int | tuple[str, int] = 1024,
        height: int | tuple[str, int] = 1024,
        mu: float | tuple[str, int] = 0.0,
        std: float | tuple[str, int] = 1.75,
    ) -> None:
        super().__init__(node_id=node_id)
        self.steps = steps
        self.width = width
        self.height = height
        self.mu = mu
        self.std = std
        self.__validate()

    def __validate(self) -> None:
        validate_range("steps", self.steps, 1, 200)
        validate_range("width", self.width, 256, 8192)
        validate_range("height", self.height, 256, 8192)
        validate_range("mu", self.mu, -10.0, 10.0)
        validate_range("std", self.std, 0.1, 5.0)


AlignYourStepsScheduler_Model_Types = Literal["SD1", "SDXL", "SVD"]
ALIGNYOURSTEPSSCHEDULER_MODEL_TYPES = frozenset(
    get_args(AlignYourStepsScheduler_Model_Types)
)


class AlignYourStepsScheduler(BaseNode):
    __slots__ = ("model_type", "steps", "denoise")
    CATEGORY = "model/sampling/schedulers"
    OUTPUTS = {"SIGMAS": 0}
    RETURN_TYPES = ("SIGMAS",)
    INPUT_SOCKET_TYPES = {"model_type": "COMBO", "steps": "INT", "denoise": "FLOAT"}

    def __init__(
        self,
        node_id: str,
        model_type: AlignYourStepsScheduler_Model_Types | tuple[str, int],
        steps: int | tuple[str, int] = 10,
        denoise: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model_type = model_type
        self.steps = steps
        self.denoise = denoise
        self.__validate()

    def __validate(self) -> None:
        validate_choice(
            "model_type", self.model_type, ALIGNYOURSTEPSSCHEDULER_MODEL_TYPES
        )
        validate_range("steps", self.steps, 1, 10_000)
        validate_range("denoise", self.denoise, 0.0, 1.0)


class GITSScheduler(BaseNode):
    __slots__ = ("coeff", "steps", "denoise")
    CATEGORY = "model/sampling/schedulers"
    OUTPUTS = {"SIGMAS": 0}
    RETURN_TYPES = ("SIGMAS",)
    INPUT_SOCKET_TYPES = {"coeff": "FLOAT", "steps": "INT", "denoise": "FLOAT"}

    def __init__(
        self,
        node_id: str,
        coeff: float | tuple[str, int] = 1.2,
        steps: int | tuple[str, int] = 10,
        denoise: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.coeff = coeff
        self.steps = steps
        self.denoise = denoise
        self.__validate()

    def __validate(self) -> None:
        validate_range("coeff", self.coeff, 0.8, 1.5)
        validate_range("steps", self.steps, 2, 1000)
        validate_range("denoise", self.denoise, 0.0, 1.0)


class Flux2Scheduler(BaseNode):
    __slots__ = ("steps", "width", "height")
    CATEGORY = "model/sampling/schedulers"
    OUTPUTS = {"SIGMAS": 0}
    RETURN_TYPES = ("SIGMAS",)
    INPUT_SOCKET_TYPES = {"steps": "INT", "width": "INT", "height": "INT"}

    def __init__(
        self,
        node_id: str,
        steps: int | tuple[str, int] = 20,
        width: int | tuple[str, int] = 1024,
        height: int | tuple[str, int] = 1024,
    ) -> None:
        super().__init__(node_id=node_id)
        self.steps = steps
        self.width = width
        self.height = height
        self.__validate()

    def __validate(self) -> None:
        validate_range("steps", self.steps, 1, 4096)
        validate_range("width", self.width, 16, 16_384)
        validate_range("height", self.height, 16, 16_384)


class LTXVScheduler(BaseNode):
    __slots__ = ("steps", "max_shift", "base_shift", "stretch", "terminal", "latent")
    CATEGORY = "model/sampling/schedulers"
    OUTPUTS = {"SIGMAS": 0}
    RETURN_TYPES = ("SIGMAS",)
    INPUT_SOCKET_TYPES = {
        "steps": "INT",
        "max_shift": "FLOAT",
        "base_shift": "FLOAT",
        "stretch": "BOOLEAN",
        "terminal": "FLOAT",
        "latent": "LATENT",
    }

    def __init__(
        self,
        node_id: str,
        steps: int | tuple[str, int] = 20,
        max_shift: float | tuple[str, int] = 2.05,
        base_shift: float | tuple[str, int] = 0.95,
        stretch: bool | tuple[str, int] = True,
        terminal: float | tuple[str, int] = 0.1,
        latent: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.steps = steps
        self.max_shift = max_shift
        self.base_shift = base_shift
        self.stretch = stretch
        self.terminal = terminal
        if latent is not None:
            self.latent = latent
        self.__validate()

    def __validate(self) -> None:
        validate_range("steps", self.steps, 1, 10_000)
        validate_range("max_shift", self.max_shift, 0.0, 100.0)
        validate_range("base_shift", self.base_shift, 0.0, 100.0)
        validate_bool("stretch", self.stretch)
        validate_range("terminal", self.terminal, 0.0, 0.99)


OptimalStepsScheduler_Model_Types = Literal["FLUX", "Wan", "Chroma"]
OPTIMALSTEPSSCHEDULER_MODEL_TYPES = frozenset(
    get_args(OptimalStepsScheduler_Model_Types)
)


class OptimalStepsScheduler(BaseNode):
    __slots__ = ("model_type", "steps", "denoise")
    CATEGORY = "model/sampling/schedulers"
    OUTPUTS = {"SIGMAS": 0}
    RETURN_TYPES = ("SIGMAS",)
    INPUT_SOCKET_TYPES = {"model_type": "COMBO", "steps": "INT", "denoise": "FLOAT"}

    def __init__(
        self,
        node_id: str,
        model_type: OptimalStepsScheduler_Model_Types | tuple[str, int],
        steps: int | tuple[str, int] = 20,
        denoise: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model_type = model_type
        self.steps = steps
        self.denoise = denoise
        self.__validate()

    def __validate(self) -> None:
        validate_choice(
            "model_type", self.model_type, OPTIMALSTEPSSCHEDULER_MODEL_TYPES
        )
        validate_range("steps", self.steps, 3, 1000)
        validate_range("denoise", self.denoise, 0.0, 1.0)
