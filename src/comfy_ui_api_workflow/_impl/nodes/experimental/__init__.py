from typing import Literal, get_args

from ..base import BaseNode
from ..validation import validate_bool, validate_choice, validate_range
from ...lazy import submodules_of


class LatentBlend(BaseNode):
    __slots__ = ("samples1", "samples2", "blend_factor")
    CATEGORY = "experimental"
    OUTPUTS = {"LATENT": 0}
    RETURN_TYPES = ("LATENT",)
    INPUT_SOCKET_TYPES = {
        "samples1": "LATENT",
        "samples2": "LATENT",
        "blend_factor": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        samples1: tuple[str, int],
        samples2: tuple[str, int],
        blend_factor: float | tuple[str, int] = 0.5,
    ) -> None:
        super().__init__(node_id=node_id)
        self.samples1 = samples1
        self.samples2 = samples2
        self.blend_factor = blend_factor
        self.__validate()

    def __validate(self) -> None:
        validate_range("blend_factor", self.blend_factor, 0, 1)


class SelfAttentionGuidance(BaseNode):
    __slots__ = ("model", "scale", "blur_sigma")
    CATEGORY = "experimental"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {"model": "MODEL", "scale": "FLOAT", "blur_sigma": "FLOAT"}

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        scale: float | tuple[str, int] = 0.5,
        blur_sigma: float | tuple[str, int] = 2.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.scale = scale
        self.blur_sigma = blur_sigma
        self.__validate()

    def __validate(self) -> None:
        validate_range("scale", self.scale, -2.0, 5.0)
        validate_range("blur_sigma", self.blur_sigma, 0.0, 10.0)


class PerpNegGuider(BaseNode):
    __slots__ = (
        "model",
        "positive",
        "negative",
        "empty_conditioning",
        "cfg",
        "neg_scale",
    )
    CATEGORY = "experimental"
    OUTPUTS = {"GUIDER": 0}
    RETURN_TYPES = ("GUIDER",)
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "positive": "CONDITIONING",
        "negative": "CONDITIONING",
        "empty_conditioning": "CONDITIONING",
        "cfg": "FLOAT",
        "neg_scale": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        positive: tuple[str, int],
        negative: tuple[str, int],
        empty_conditioning: tuple[str, int],
        cfg: float | tuple[str, int] = 8.0,
        neg_scale: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.positive = positive
        self.negative = negative
        self.empty_conditioning = empty_conditioning
        self.cfg = cfg
        self.neg_scale = neg_scale
        self.__validate()

    def __validate(self) -> None:
        validate_range("cfg", self.cfg, 0.0, 100.0)
        validate_range("neg_scale", self.neg_scale, 0.0, 100.0)


class DifferentialDiffusion(BaseNode):
    __slots__ = ("model", "strength")
    CATEGORY = "experimental"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {"model": "MODEL", "strength": "FLOAT"}

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        strength: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.strength = strength
        self.__validate()

    def __validate(self) -> None:
        validate_range("strength", self.strength, 0.0, 1.0)


SamplerEulerCFGpp_Versions = Literal["regular", "alternative"]
SAMPLEREULERCFGPP_VERSIONS = frozenset(get_args(SamplerEulerCFGpp_Versions))


class SamplerEulerCFGpp(BaseNode):
    __slots__ = ("version",)
    CATEGORY = "experimental"
    OUTPUTS = {"SAMPLER": 0}
    RETURN_TYPES = ("SAMPLER",)
    INPUT_SOCKET_TYPES = {"version": "COMBO"}

    def __init__(
        self, node_id: str, version: SamplerEulerCFGpp_Versions | tuple[str, int]
    ) -> None:
        super().__init__(node_id=node_id)
        self.version = version
        self.__validate()

    def __validate(self) -> None:
        validate_choice("version", self.version, SAMPLEREULERCFGPP_VERSIONS)


class FluxKVCache(BaseNode):
    __slots__ = ("model",)
    CATEGORY = "experimental"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {"model": "MODEL"}

    def __init__(self, node_id: str, model: tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.model = model


LoraSave_Lora_Types = Literal["standard", "full_diff"]
LORASAVE_LORA_TYPES = frozenset(get_args(LoraSave_Lora_Types))


class LoraSave(BaseNode):
    __slots__ = (
        "lora_type",
        "filename_prefix",
        "rank",
        "bias_diff",
        "model_diff",
        "text_encoder_diff",
    )
    CATEGORY = "experimental"
    INPUT_SOCKET_TYPES = {
        "filename_prefix": "STRING",
        "rank": "INT",
        "lora_type": "COMBO",
        "bias_diff": "BOOLEAN",
        "model_diff": "MODEL",
        "text_encoder_diff": "CLIP",
    }

    def __init__(
        self,
        node_id: str,
        lora_type: LoraSave_Lora_Types | tuple[str, int],
        filename_prefix: str | tuple[str, int] = "loras/ComfyUI_extracted_lora",
        rank: int | tuple[str, int] = 8,
        bias_diff: bool | tuple[str, int] = True,
        model_diff: tuple[str, int] | None = None,
        text_encoder_diff: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.lora_type = lora_type
        self.filename_prefix = filename_prefix
        self.rank = rank
        self.bias_diff = bias_diff
        if model_diff is not None:
            self.model_diff = model_diff
        if text_encoder_diff is not None:
            self.text_encoder_diff = text_encoder_diff
        self.__validate()

    def __validate(self) -> None:
        validate_choice("lora_type", self.lora_type, LORASAVE_LORA_TYPES)
        validate_range("rank", self.rank, 1, 4096)
        validate_bool("bias_diff", self.bias_diff)


TorchCompileModel_Backends = Literal["inductor", "cudagraphs"]
TORCHCOMPILEMODEL_BACKENDS = frozenset(get_args(TorchCompileModel_Backends))


class TorchCompileModel(BaseNode):
    __slots__ = ("model", "backend")
    CATEGORY = "experimental"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {"model": "MODEL", "backend": "COMBO"}

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        backend: TorchCompileModel_Backends | tuple[str, int],
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.backend = backend
        self.__validate()

    def __validate(self) -> None:
        validate_choice("backend", self.backend, TORCHCOMPILEMODEL_BACKENDS)


class Mahiro(BaseNode):
    __slots__ = ("model",)
    CATEGORY = "experimental"
    OUTPUTS = {"patched_model": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {"model": "MODEL"}

    def __init__(self, node_id: str, model: tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.model = model


class FreSca(BaseNode):
    __slots__ = ("model", "scale_low", "scale_high", "freq_cutoff")
    CATEGORY = "experimental"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "scale_low": "FLOAT",
        "scale_high": "FLOAT",
        "freq_cutoff": "INT",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        scale_low: float | tuple[str, int] = 1.0,
        scale_high: float | tuple[str, int] = 1.25,
        freq_cutoff: int | tuple[str, int] = 20,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.freq_cutoff = freq_cutoff
        self.__validate()

    def __validate(self) -> None:
        validate_range("scale_low", self.scale_low, 0, 10)
        validate_range("scale_high", self.scale_high, 0, 10)
        validate_range("freq_cutoff", self.freq_cutoff, 1, 10_000)


__getattr__ = submodules_of(__name__)
