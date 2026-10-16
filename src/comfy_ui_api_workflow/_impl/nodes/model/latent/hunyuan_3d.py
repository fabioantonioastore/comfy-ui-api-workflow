from ...base import BaseNode
from ...validation import validate_range


class EmptyLatentHunyuan3Dv2(BaseNode):
    __slots__ = ("resolution", "batch_size")
    CATEGORY = "model/latent/hunyuan 3d"
    OUTPUTS = {"LATENT": 0}
    RETURN_TYPES = ("LATENT",)
    INPUT_SOCKET_TYPES = {"resolution": "INT", "batch_size": "INT"}

    def __init__(
        self,
        node_id: str,
        resolution: int | tuple[str, int] = 3072,
        batch_size: int | tuple[str, int] = 1,
    ) -> None:
        super().__init__(node_id=node_id)
        self.resolution = resolution
        self.batch_size = batch_size
        self.__validate()

    def __validate(self) -> None:
        validate_range("resolution", self.resolution, 1, 8192)
        validate_range("batch_size", self.batch_size, 1, 4096)


class VAEDecodeHunyuan3D(BaseNode):
    __slots__ = ("samples", "vae", "num_chunks", "octree_resolution")
    CATEGORY = "model/latent/hunyuan 3d"
    OUTPUTS = {"VOXEL": 0}
    RETURN_TYPES = ("VOXEL",)
    INPUT_SOCKET_TYPES = {
        "samples": "LATENT",
        "vae": "VAE",
        "num_chunks": "INT",
        "octree_resolution": "INT",
    }

    def __init__(
        self,
        node_id: str,
        samples: tuple[str, int],
        vae: tuple[str, int],
        num_chunks: int | tuple[str, int] = 8000,
        octree_resolution: int | tuple[str, int] = 256,
    ) -> None:
        super().__init__(node_id=node_id)
        self.samples = samples
        self.vae = vae
        self.num_chunks = num_chunks
        self.octree_resolution = octree_resolution
        self.__validate()

    def __validate(self) -> None:
        validate_range("num_chunks", self.num_chunks, 1000, 500_000)
        validate_range("octree_resolution", self.octree_resolution, 16, 512)
