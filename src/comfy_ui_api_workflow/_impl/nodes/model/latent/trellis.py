from typing import Literal, get_args

from ...base import BaseNode
from ...validation import validate_choice, validate_range


class EmptyTrellis2LatentStructure(BaseNode):
    __slots__ = ("batch_size",)
    CATEGORY = "model/latent/trellis"
    OUTPUTS = {"LATENT": 0}
    RETURN_TYPES = ("LATENT",)
    INPUT_SOCKET_TYPES = {"batch_size": "INT"}

    def __init__(self, node_id: str, batch_size: int | tuple[str, int] = 1) -> None:
        super().__init__(node_id=node_id)
        self.batch_size = batch_size
        self.__validate()

    def __validate(self) -> None:
        validate_range("batch_size", self.batch_size, 1, 4096)


class VaeDecodeTextureTrellis(BaseNode):
    __slots__ = ("samples", "vae", "shape_subdivides")
    CATEGORY = "model/latent/trellis"
    OUTPUTS = {"voxel_colors": 0}
    RETURN_TYPES = ("VOXEL",)
    INPUT_SOCKET_TYPES = {
        "samples": "LATENT",
        "vae": "VAE",
        "shape_subdivides": "SHAPE_SUBDIVIDES",
    }

    def __init__(
        self,
        node_id: str,
        samples: tuple[str, int],
        vae: tuple[str, int],
        shape_subdivides: tuple[str, int],
    ) -> None:
        super().__init__(node_id=node_id)
        self.samples = samples
        self.vae = vae
        self.shape_subdivides = shape_subdivides


class VaeDecodeShapeTrellis(BaseNode):
    __slots__ = ("samples", "vae")
    CATEGORY = "model/latent/trellis"
    OUTPUTS = {"mesh": 0, "shape_subdivides": 1}
    RETURN_TYPES = ("MESH", "SHAPE_SUBDIVIDES")
    INPUT_SOCKET_TYPES = {"samples": "LATENT", "vae": "VAE"}

    def __init__(
        self, node_id: str, samples: tuple[str, int], vae: tuple[str, int]
    ) -> None:
        super().__init__(node_id=node_id)
        self.samples = samples
        self.vae = vae


VaeDecodeStructureTrellis2_Resolutions = Literal["32", "64"]
VAEDECODESTRUCTURETRELLIS2_RESOLUTIONS = frozenset(
    get_args(VaeDecodeStructureTrellis2_Resolutions)
)


class VaeDecodeStructureTrellis2(BaseNode):
    __slots__ = ("samples", "vae", "resolution")
    CATEGORY = "model/latent/trellis"
    OUTPUTS = {"voxel": 0}
    RETURN_TYPES = ("VOXEL",)
    INPUT_SOCKET_TYPES = {"samples": "LATENT", "vae": "VAE", "resolution": "COMBO"}

    def __init__(
        self,
        node_id: str,
        samples: tuple[str, int],
        vae: tuple[str, int],
        resolution: VaeDecodeStructureTrellis2_Resolutions | tuple[str, int] = "32",
    ) -> None:
        super().__init__(node_id=node_id)
        self.samples = samples
        self.vae = vae
        self.resolution = resolution
        self.__validate()

    def __validate(self) -> None:
        validate_choice(
            "resolution", self.resolution, VAEDECODESTRUCTURETRELLIS2_RESOLUTIONS
        )
