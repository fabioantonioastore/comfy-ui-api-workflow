from typing import Literal, get_args

from ...base import BaseNode
from ...validation import (
    validate_bool,
    validate_choice,
    validate_not_empty,
    validate_range,
)
from ....lazy import submodules_of


class VAEDecode(BaseNode):
    __slots__ = ("samples", "vae")
    CATEGORY = "model/latent"
    OUTPUTS = {"IMAGE": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_SOCKET_TYPES = {"samples": "LATENT", "vae": "VAE"}

    def __init__(
        self, node_id: str, samples: tuple[str, int], vae: tuple[str, int]
    ) -> None:
        super().__init__(node_id=node_id)
        self.samples = samples
        self.vae = vae


class VAEEncode(BaseNode):
    __slots__ = ("pixels", "vae")
    CATEGORY = "model/latent"
    OUTPUTS = {"LATENT": 0}
    RETURN_TYPES = ("LATENT",)
    INPUT_SOCKET_TYPES = {"pixels": "IMAGE", "vae": "VAE"}

    def __init__(
        self, node_id: str, pixels: tuple[str, int], vae: tuple[str, int]
    ) -> None:
        super().__init__(node_id=node_id)
        self.pixels = pixels
        self.vae = vae


class VAEEncodeForInpaint(BaseNode):
    __slots__ = ("pixels", "vae", "mask", "grow_mask_by")
    CATEGORY = "model/latent"
    OUTPUTS = {"LATENT": 0}
    RETURN_TYPES = ("LATENT",)
    INPUT_SOCKET_TYPES = {
        "pixels": "IMAGE",
        "vae": "VAE",
        "mask": "MASK",
        "grow_mask_by": "INT",
    }

    def __init__(
        self,
        node_id: str,
        pixels: tuple[str, int],
        vae: tuple[str, int],
        mask: tuple[str, int],
        grow_mask_by: int | tuple[str, int] = 6,
    ) -> None:
        super().__init__(node_id=node_id)
        self.pixels = pixels
        self.vae = vae
        self.mask = mask
        self.grow_mask_by = grow_mask_by
        self.__validate()

    def __validate(self) -> None:
        validate_range("grow_mask_by", self.grow_mask_by, 0, 64)


class EmptyLatentImage(BaseNode):
    __slots__ = ("width", "height", "batch_size")
    CATEGORY = "model/latent"
    OUTPUTS = {"LATENT": 0}
    RETURN_TYPES = ("LATENT",)
    INPUT_SOCKET_TYPES = {"width": "INT", "height": "INT", "batch_size": "INT"}

    def __init__(
        self,
        node_id: str,
        width: int | tuple[str, int] = 1024,
        height: int | tuple[str, int] = 1024,
        batch_size: int | tuple[str, int] = 1,
    ) -> None:
        super().__init__(node_id=node_id)
        self.width = width
        self.height = height
        self.batch_size = batch_size
        self.__validate()

    def __validate(self) -> None:
        validate_range("width", self.width, 16, 16_384)
        validate_range("height", self.height, 16, 16_384)
        validate_range("batch_size", self.batch_size, 1, 4096)


LatentUpscale_Upscale_Methods = Literal[
    "nearest-exact", "bilinear", "area", "bicubic", "bislerp"
]
LatentUpscale_Crops = Literal["disabled", "center"]
LATENTUPSCALE_UPSCALE_METHODS = frozenset(get_args(LatentUpscale_Upscale_Methods))
LATENTUPSCALE_CROPS = frozenset(get_args(LatentUpscale_Crops))


class LatentUpscale(BaseNode):
    __slots__ = ("samples", "upscale_method", "crop", "width", "height")
    CATEGORY = "model/latent"
    OUTPUTS = {"LATENT": 0}
    RETURN_TYPES = ("LATENT",)
    INPUT_SOCKET_TYPES = {
        "samples": "LATENT",
        "upscale_method": "V1 COMBO",
        "width": "INT",
        "height": "INT",
        "crop": "V1 COMBO",
    }

    def __init__(
        self,
        node_id: str,
        samples: tuple[str, int],
        upscale_method: LatentUpscale_Upscale_Methods | tuple[str, int],
        crop: LatentUpscale_Crops | tuple[str, int],
        width: int | tuple[str, int] = 512,
        height: int | tuple[str, int] = 512,
    ) -> None:
        super().__init__(node_id=node_id)
        self.samples = samples
        self.upscale_method = upscale_method
        self.crop = crop
        self.width = width
        self.height = height
        self.__validate()

    def __validate(self) -> None:
        validate_choice(
            "upscale_method", self.upscale_method, LATENTUPSCALE_UPSCALE_METHODS
        )
        validate_choice("crop", self.crop, LATENTUPSCALE_CROPS)
        validate_range("width", self.width, 0, 16_384)
        validate_range("height", self.height, 0, 16_384)


LatentUpscaleBy_Upscale_Methods = Literal[
    "nearest-exact", "bilinear", "area", "bicubic", "bislerp"
]
LATENTUPSCALEBY_UPSCALE_METHODS = frozenset(get_args(LatentUpscaleBy_Upscale_Methods))


class LatentUpscaleBy(BaseNode):
    __slots__ = ("samples", "upscale_method", "scale_by")
    CATEGORY = "model/latent"
    OUTPUTS = {"LATENT": 0}
    RETURN_TYPES = ("LATENT",)
    INPUT_SOCKET_TYPES = {
        "samples": "LATENT",
        "upscale_method": "V1 COMBO",
        "scale_by": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        samples: tuple[str, int],
        upscale_method: LatentUpscaleBy_Upscale_Methods | tuple[str, int],
        scale_by: float | tuple[str, int] = 1.5,
    ) -> None:
        super().__init__(node_id=node_id)
        self.samples = samples
        self.upscale_method = upscale_method
        self.scale_by = scale_by
        self.__validate()

    def __validate(self) -> None:
        validate_choice(
            "upscale_method", self.upscale_method, LATENTUPSCALEBY_UPSCALE_METHODS
        )
        validate_range("scale_by", self.scale_by, 0.01, 8.0)


class SetLatentNoiseMask(BaseNode):
    __slots__ = ("samples", "mask")
    CATEGORY = "model/latent"
    OUTPUTS = {"LATENT": 0}
    RETURN_TYPES = ("LATENT",)
    INPUT_SOCKET_TYPES = {"samples": "LATENT", "mask": "MASK"}

    def __init__(
        self, node_id: str, samples: tuple[str, int], mask: tuple[str, int]
    ) -> None:
        super().__init__(node_id=node_id)
        self.samples = samples
        self.mask = mask


class LatentComposite(BaseNode):
    __slots__ = ("samples_to", "samples_from", "x", "y", "feather")
    CATEGORY = "model/latent"
    OUTPUTS = {"LATENT": 0}
    RETURN_TYPES = ("LATENT",)
    INPUT_SOCKET_TYPES = {
        "samples_to": "LATENT",
        "samples_from": "LATENT",
        "x": "INT",
        "y": "INT",
        "feather": "INT",
    }

    def __init__(
        self,
        node_id: str,
        samples_to: tuple[str, int],
        samples_from: tuple[str, int],
        x: int | tuple[str, int] = 0,
        y: int | tuple[str, int] = 0,
        feather: int | tuple[str, int] = 0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.samples_to = samples_to
        self.samples_from = samples_from
        self.x = x
        self.y = y
        self.feather = feather
        self.__validate()

    def __validate(self) -> None:
        validate_range("x", self.x, 0, 16_384)
        validate_range("y", self.y, 0, 16_384)
        validate_range("feather", self.feather, 0, 16_384)


class VAEDecodeTiled(BaseNode):
    __slots__ = (
        "samples",
        "vae",
        "tile_size",
        "overlap",
        "temporal_size",
        "temporal_overlap",
    )
    CATEGORY = "model/latent"
    OUTPUTS = {"IMAGE": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_SOCKET_TYPES = {
        "samples": "LATENT",
        "vae": "VAE",
        "tile_size": "INT",
        "overlap": "INT",
        "temporal_size": "INT",
        "temporal_overlap": "INT",
    }

    def __init__(
        self,
        node_id: str,
        samples: tuple[str, int],
        vae: tuple[str, int],
        tile_size: int | tuple[str, int] = 512,
        overlap: int | tuple[str, int] = 64,
        temporal_size: int | tuple[str, int] = 64,
        temporal_overlap: int | tuple[str, int] = 8,
    ) -> None:
        super().__init__(node_id=node_id)
        self.samples = samples
        self.vae = vae
        self.tile_size = tile_size
        self.overlap = overlap
        self.temporal_size = temporal_size
        self.temporal_overlap = temporal_overlap
        self.__validate()

    def __validate(self) -> None:
        validate_range("tile_size", self.tile_size, 64, 4096)
        validate_range("overlap", self.overlap, 0, 4096)
        validate_range("temporal_size", self.temporal_size, 8, 4096)
        validate_range("temporal_overlap", self.temporal_overlap, 4, 4096)


class VAEEncodeTiled(BaseNode):
    __slots__ = (
        "pixels",
        "vae",
        "tile_size",
        "overlap",
        "temporal_size",
        "temporal_overlap",
    )
    CATEGORY = "model/latent"
    OUTPUTS = {"LATENT": 0}
    RETURN_TYPES = ("LATENT",)
    INPUT_SOCKET_TYPES = {
        "pixels": "IMAGE",
        "vae": "VAE",
        "tile_size": "INT",
        "overlap": "INT",
        "temporal_size": "INT",
        "temporal_overlap": "INT",
    }

    def __init__(
        self,
        node_id: str,
        pixels: tuple[str, int],
        vae: tuple[str, int],
        tile_size: int | tuple[str, int] = 512,
        overlap: int | tuple[str, int] = 64,
        temporal_size: int | tuple[str, int] = 64,
        temporal_overlap: int | tuple[str, int] = 8,
    ) -> None:
        super().__init__(node_id=node_id)
        self.pixels = pixels
        self.vae = vae
        self.tile_size = tile_size
        self.overlap = overlap
        self.temporal_size = temporal_size
        self.temporal_overlap = temporal_overlap
        self.__validate()

    def __validate(self) -> None:
        validate_range("tile_size", self.tile_size, 64, 4096)
        validate_range("overlap", self.overlap, 0, 4096)
        validate_range("temporal_size", self.temporal_size, 8, 4096)
        validate_range("temporal_overlap", self.temporal_overlap, 4, 4096)


class LoadLatent(BaseNode):
    __slots__ = ("latent",)
    CATEGORY = "model/latent"
    OUTPUTS = {"LATENT": 0}
    RETURN_TYPES = ("LATENT",)
    INPUT_SOCKET_TYPES = {"latent": "V1 COMBO"}

    def __init__(self, node_id: str, latent: str | tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.latent = latent
        self.__validate()

    def __validate(self) -> None:
        validate_not_empty("latent", self.latent)


class SaveLatent(BaseNode):
    __slots__ = ("samples", "filename_prefix")
    CATEGORY = "model/latent"
    OUTPUTS = {"samples": 0}
    RETURN_TYPES = ("LATENT",)
    INPUT_SOCKET_TYPES = {"samples": "LATENT", "filename_prefix": "STRING"}

    def __init__(
        self,
        node_id: str,
        samples: tuple[str, int],
        filename_prefix: str | tuple[str, int] = "latents/ComfyUI",
    ) -> None:
        super().__init__(node_id=node_id)
        self.samples = samples
        self.filename_prefix = filename_prefix


class LatentCompositeMasked(BaseNode):
    __slots__ = ("destination", "source", "x", "y", "resize_source", "mask")
    CATEGORY = "model/latent"
    OUTPUTS = {"LATENT": 0}
    RETURN_TYPES = ("LATENT",)
    INPUT_SOCKET_TYPES = {
        "destination": "LATENT",
        "source": "LATENT",
        "x": "INT",
        "y": "INT",
        "resize_source": "BOOLEAN",
        "mask": "MASK",
    }

    def __init__(
        self,
        node_id: str,
        destination: tuple[str, int],
        source: tuple[str, int],
        x: int | tuple[str, int] = 0,
        y: int | tuple[str, int] = 0,
        resize_source: bool | tuple[str, int] = False,
        mask: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.destination = destination
        self.source = source
        self.x = x
        self.y = y
        self.resize_source = resize_source
        if mask is not None:
            self.mask = mask
        self.__validate()

    def __validate(self) -> None:
        validate_range("x", self.x, 0, 16_384)
        validate_range("y", self.y, 0, 16_384)
        validate_bool("resize_source", self.resize_source)


class EmptyLatentAudio(BaseNode):
    __slots__ = ("seconds", "batch_size")
    CATEGORY = "model/latent"
    OUTPUTS = {"LATENT": 0}
    RETURN_TYPES = ("LATENT",)
    INPUT_SOCKET_TYPES = {"seconds": "FLOAT", "batch_size": "INT"}

    def __init__(
        self,
        node_id: str,
        seconds: float | tuple[str, int] = 47.6,
        batch_size: int | tuple[str, int] = 1,
    ) -> None:
        super().__init__(node_id=node_id)
        self.seconds = seconds
        self.batch_size = batch_size
        self.__validate()

    def __validate(self) -> None:
        validate_range("seconds", self.seconds, 1.0, 1000.0)
        validate_range("batch_size", self.batch_size, 1, 4096)


class VAEEncodeAudio(BaseNode):
    __slots__ = ("audio", "vae")
    CATEGORY = "model/latent"
    OUTPUTS = {"LATENT": 0}
    RETURN_TYPES = ("LATENT",)
    INPUT_SOCKET_TYPES = {"audio": "AUDIO", "vae": "VAE"}

    def __init__(
        self, node_id: str, audio: tuple[str, int], vae: tuple[str, int]
    ) -> None:
        super().__init__(node_id=node_id)
        self.audio = audio
        self.vae = vae


class VAEDecodeAudio(BaseNode):
    __slots__ = ("samples", "vae")
    CATEGORY = "model/latent"
    OUTPUTS = {"AUDIO": 0}
    RETURN_TYPES = ("AUDIO",)
    INPUT_SOCKET_TYPES = {"samples": "LATENT", "vae": "VAE"}

    def __init__(
        self, node_id: str, samples: tuple[str, int], vae: tuple[str, int]
    ) -> None:
        super().__init__(node_id=node_id)
        self.samples = samples
        self.vae = vae


class VAEDecodeAudioTiled(BaseNode):
    __slots__ = ("samples", "vae", "tile_size", "overlap")
    CATEGORY = "model/latent"
    OUTPUTS = {"AUDIO": 0}
    RETURN_TYPES = ("AUDIO",)
    INPUT_SOCKET_TYPES = {
        "samples": "LATENT",
        "vae": "VAE",
        "tile_size": "INT",
        "overlap": "INT",
    }

    def __init__(
        self,
        node_id: str,
        samples: tuple[str, int],
        vae: tuple[str, int],
        tile_size: int | tuple[str, int] = 512,
        overlap: int | tuple[str, int] = 64,
    ) -> None:
        super().__init__(node_id=node_id)
        self.samples = samples
        self.vae = vae
        self.tile_size = tile_size
        self.overlap = overlap
        self.__validate()

    def __validate(self) -> None:
        validate_range("tile_size", self.tile_size, 32, 8192)
        validate_range("overlap", self.overlap, 0, 1024)


class TrimVideoLatent(BaseNode):
    __slots__ = ("samples", "trim_amount")
    CATEGORY = "model/latent"
    OUTPUTS = {"LATENT": 0}
    RETURN_TYPES = ("LATENT",)
    INPUT_SOCKET_TYPES = {"samples": "LATENT", "trim_amount": "INT"}

    def __init__(
        self,
        node_id: str,
        samples: tuple[str, int],
        trim_amount: int | tuple[str, int] = 0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.samples = samples
        self.trim_amount = trim_amount
        self.__validate()

    def __validate(self) -> None:
        validate_range("trim_amount", self.trim_amount, 0, 99_999)


__getattr__ = submodules_of(__name__)
