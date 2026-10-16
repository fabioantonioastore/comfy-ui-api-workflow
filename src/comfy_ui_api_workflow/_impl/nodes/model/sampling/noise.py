from ...base import BaseNode
from ...validation import validate_range


class RandomNoise(BaseNode):
    __slots__ = ("noise_seed",)
    CATEGORY = "model/sampling/noise"
    OUTPUTS = {"NOISE": 0}
    RETURN_TYPES = ("NOISE",)
    INPUT_SOCKET_TYPES = {"noise_seed": "INT"}

    def __init__(self, node_id: str, noise_seed: int | tuple[str, int] = 0) -> None:
        super().__init__(node_id=node_id)
        self.noise_seed = noise_seed
        self.__validate()

    def __validate(self) -> None:
        validate_range("noise_seed", self.noise_seed, 0, 18_446_744_073_709_551_615)


class DisableNoise(BaseNode):
    __slots__ = ()
    CATEGORY = "model/sampling/noise"
    OUTPUTS = {"NOISE": 0}
    RETURN_TYPES = ("NOISE",)


class AddNoise(BaseNode):
    __slots__ = ("model", "noise", "sigmas", "latent_image")
    CATEGORY = "model/sampling/noise"
    OUTPUTS = {"LATENT": 0}
    RETURN_TYPES = ("LATENT",)
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "noise": "NOISE",
        "sigmas": "SIGMAS",
        "latent_image": "LATENT",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        noise: tuple[str, int],
        sigmas: tuple[str, int],
        latent_image: tuple[str, int],
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.noise = noise
        self.sigmas = sigmas
        self.latent_image = latent_image
