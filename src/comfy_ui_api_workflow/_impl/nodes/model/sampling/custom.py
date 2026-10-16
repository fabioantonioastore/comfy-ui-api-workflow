from ...base import BaseNode
from ...validation import validate_bool, validate_range


class SamplerCustom(BaseNode):
    __slots__ = (
        "model",
        "positive",
        "negative",
        "sampler",
        "sigmas",
        "latent_image",
        "add_noise",
        "noise_seed",
        "cfg",
    )
    CATEGORY = "model/sampling/custom"
    OUTPUTS = {"output": 0, "denoised_output": 1}
    RETURN_TYPES = ("LATENT", "LATENT")
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "add_noise": "BOOLEAN",
        "noise_seed": "INT",
        "cfg": "FLOAT",
        "positive": "CONDITIONING",
        "negative": "CONDITIONING",
        "sampler": "SAMPLER",
        "sigmas": "SIGMAS",
        "latent_image": "LATENT",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        positive: tuple[str, int],
        negative: tuple[str, int],
        sampler: tuple[str, int],
        sigmas: tuple[str, int],
        latent_image: tuple[str, int],
        add_noise: bool | tuple[str, int] = True,
        noise_seed: int | tuple[str, int] = 0,
        cfg: float | tuple[str, int] = 8.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.positive = positive
        self.negative = negative
        self.sampler = sampler
        self.sigmas = sigmas
        self.latent_image = latent_image
        self.add_noise = add_noise
        self.noise_seed = noise_seed
        self.cfg = cfg
        self.__validate()

    def __validate(self) -> None:
        validate_bool("add_noise", self.add_noise)
        validate_range("noise_seed", self.noise_seed, 0, 18_446_744_073_709_551_615)
        validate_range("cfg", self.cfg, 0.0, 100.0)


class SamplerCustomAdvanced(BaseNode):
    __slots__ = ("noise", "guider", "sampler", "sigmas", "latent_image")
    CATEGORY = "model/sampling/custom"
    OUTPUTS = {"output": 0, "denoised_output": 1}
    RETURN_TYPES = ("LATENT", "LATENT")
    INPUT_SOCKET_TYPES = {
        "noise": "NOISE",
        "guider": "GUIDER",
        "sampler": "SAMPLER",
        "sigmas": "SIGMAS",
        "latent_image": "LATENT",
    }

    def __init__(
        self,
        node_id: str,
        noise: tuple[str, int],
        guider: tuple[str, int],
        sampler: tuple[str, int],
        sigmas: tuple[str, int],
        latent_image: tuple[str, int],
    ) -> None:
        super().__init__(node_id=node_id)
        self.noise = noise
        self.guider = guider
        self.sampler = sampler
        self.sigmas = sigmas
        self.latent_image = latent_image


class APG(BaseNode):
    __slots__ = ("model", "eta", "norm_threshold", "momentum")
    CATEGORY = "model/sampling/custom"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "eta": "FLOAT",
        "norm_threshold": "FLOAT",
        "momentum": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        eta: float | tuple[str, int] = 1.0,
        norm_threshold: float | tuple[str, int] = 5.0,
        momentum: float | tuple[str, int] = 0.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.eta = eta
        self.norm_threshold = norm_threshold
        self.momentum = momentum
        self.__validate()

    def __validate(self) -> None:
        validate_range("eta", self.eta, -10.0, 10.0)
        validate_range("norm_threshold", self.norm_threshold, 0.0, 50.0)
        validate_range("momentum", self.momentum, -5.0, 1.0)
