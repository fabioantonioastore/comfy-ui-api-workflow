from ...base import BaseNode
from ...validation import validate_range


class VAEDecodeTripoSplat(BaseNode):
    __slots__ = ("samples", "vae", "num_gaussians", "seed")
    CATEGORY = "model/latent/triposplat"
    OUTPUTS = {"splat": 0}
    RETURN_TYPES = ("SPLAT",)
    INPUT_SOCKET_TYPES = {
        "samples": "LATENT",
        "vae": "VAE",
        "num_gaussians": "INT",
        "seed": "INT",
    }

    def __init__(
        self,
        node_id: str,
        samples: tuple[str, int],
        vae: tuple[str, int],
        num_gaussians: int | tuple[str, int] = 262_144,
        seed: int | tuple[str, int] = 0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.samples = samples
        self.vae = vae
        self.num_gaussians = num_gaussians
        self.seed = seed
        self.__validate()

    def __validate(self) -> None:
        validate_range("num_gaussians", self.num_gaussians, 32_768, 1_048_576)
        validate_range("seed", self.seed, 0, 18_446_744_073_709_551_615)


class TripoSplatSamplingPreview(BaseNode):
    __slots__ = (
        "model",
        "vae",
        "octree_level",
        "num_gaussians",
        "yaw",
        "pitch",
        "point_size",
    )
    CATEGORY = "model/latent/triposplat"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "vae": "VAE",
        "octree_level": "INT",
        "num_gaussians": "INT",
        "yaw": "FLOAT",
        "pitch": "FLOAT",
        "point_size": "INT",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        vae: tuple[str, int],
        octree_level: int | tuple[str, int] = 5,
        num_gaussians: int | tuple[str, int] = 16_384,
        yaw: float | tuple[str, int] = 90.0,
        pitch: float | tuple[str, int] = 15.0,
        point_size: int | tuple[str, int] = 3,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.vae = vae
        self.octree_level = octree_level
        self.num_gaussians = num_gaussians
        self.yaw = yaw
        self.pitch = pitch
        self.point_size = point_size
        self.__validate()

    def __validate(self) -> None:
        validate_range("octree_level", self.octree_level, 2, 8)
        validate_range("num_gaussians", self.num_gaussians, 1024, 262_144)
        validate_range("yaw", self.yaw, -360.0, 360.0)
        validate_range("pitch", self.pitch, -89.0, 89.0)
        validate_range("point_size", self.point_size, 1, 16)
