from typing import Literal, get_args

from ...base import BaseNode
from ...validation import validate_bool, validate_choice, validate_range
from ....lazy import submodules_of
from ...dynamic import Combo, DynamicInputs, Number, validate_dynamic

ModelSamplingDiscrete_Samplings = Literal[
    "eps", "v_prediction", "lcm", "x0", "img_to_img", "img_to_img_flow"
]
MODELSAMPLINGDISCRETE_SAMPLINGS = frozenset(get_args(ModelSamplingDiscrete_Samplings))


class ModelSamplingDiscrete(BaseNode):
    __slots__ = ("model", "sampling", "zsnr")
    CATEGORY = "model/patch"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {"model": "MODEL", "sampling": "V1 COMBO", "zsnr": "BOOLEAN"}

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        sampling: ModelSamplingDiscrete_Samplings | tuple[str, int],
        zsnr: bool | tuple[str, int] = False,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.sampling = sampling
        self.zsnr = zsnr
        self.__validate()

    def __validate(self) -> None:
        validate_choice("sampling", self.sampling, MODELSAMPLINGDISCRETE_SAMPLINGS)
        validate_bool("zsnr", self.zsnr)


ModelSamplingContinuousEDM_Samplings = Literal[
    "v_prediction", "edm", "edm_playground_v2.5", "eps", "cosmos_rflow"
]
MODELSAMPLINGCONTINUOUSEDM_SAMPLINGS = frozenset(
    get_args(ModelSamplingContinuousEDM_Samplings)
)


class ModelSamplingContinuousEDM(BaseNode):
    __slots__ = ("model", "sampling", "sigma_max", "sigma_min")
    CATEGORY = "model/patch"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "sampling": "V1 COMBO",
        "sigma_max": "FLOAT",
        "sigma_min": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        sampling: ModelSamplingContinuousEDM_Samplings | tuple[str, int],
        sigma_max: float | tuple[str, int] = 120.0,
        sigma_min: float | tuple[str, int] = 0.002,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.sampling = sampling
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.__validate()

    def __validate(self) -> None:
        validate_choice("sampling", self.sampling, MODELSAMPLINGCONTINUOUSEDM_SAMPLINGS)
        validate_range("sigma_max", self.sigma_max, 0.0, 1000.0)
        validate_range("sigma_min", self.sigma_min, 0.0, 1000.0)


ModelSamplingContinuousV_Samplings = Literal["v_prediction"]
MODELSAMPLINGCONTINUOUSV_SAMPLINGS = frozenset(
    get_args(ModelSamplingContinuousV_Samplings)
)


class ModelSamplingContinuousV(BaseNode):
    __slots__ = ("model", "sampling", "sigma_max", "sigma_min")
    CATEGORY = "model/patch"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "sampling": "V1 COMBO",
        "sigma_max": "FLOAT",
        "sigma_min": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        sampling: ModelSamplingContinuousV_Samplings | tuple[str, int],
        sigma_max: float | tuple[str, int] = 500.0,
        sigma_min: float | tuple[str, int] = 0.03,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.sampling = sampling
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.__validate()

    def __validate(self) -> None:
        validate_choice("sampling", self.sampling, MODELSAMPLINGCONTINUOUSV_SAMPLINGS)
        validate_range("sigma_max", self.sigma_max, 0.0, 1000.0)
        validate_range("sigma_min", self.sigma_min, 0.0, 1000.0)


ModelSamplingAuraFlow_Samplings = Literal["flow", "img_to_img_velocity"]
MODELSAMPLINGAURAFLOW_SAMPLINGS = frozenset(get_args(ModelSamplingAuraFlow_Samplings))


class ModelSamplingAuraFlow(BaseNode):
    __slots__ = ("model", "shift", "sampling")
    CATEGORY = "model/patch"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {"model": "MODEL", "shift": "FLOAT", "sampling": "V1 COMBO"}

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        shift: float | tuple[str, int] = 1.73,
        sampling: ModelSamplingAuraFlow_Samplings | tuple[str, int] = "flow",
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.shift = shift
        self.sampling = sampling
        self.__validate()

    def __validate(self) -> None:
        validate_range("shift", self.shift, 0.0, 100.0)
        validate_choice("sampling", self.sampling, MODELSAMPLINGAURAFLOW_SAMPLINGS)


class ModelNoiseScale(BaseNode):
    __slots__ = ("model", "noise_scale")
    CATEGORY = "model/patch"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {"model": "MODEL", "noise_scale": "FLOAT"}

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        noise_scale: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.noise_scale = noise_scale
        self.__validate()

    def __validate(self) -> None:
        validate_range("noise_scale", self.noise_scale, 0.0, 64.0)


class RescaleCFG(BaseNode):
    __slots__ = ("model", "multiplier")
    CATEGORY = "model/patch"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {"model": "MODEL", "multiplier": "FLOAT"}

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        multiplier: float | tuple[str, int] = 0.7,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.multiplier = multiplier
        self.__validate()

    def __validate(self) -> None:
        validate_range("multiplier", self.multiplier, 0.0, 1.0)


ModelAttentionBackend_Attentions = Literal[
    "pytorch attention", "comfy kitchen attention"
]
MODELATTENTIONBACKEND_ATTENTIONS = frozenset(get_args(ModelAttentionBackend_Attentions))


class ModelAttentionBackend(BaseNode):
    __slots__ = ("model", "attention")
    CATEGORY = "model/patch"
    OUTPUTS = {"model": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {"model": "MODEL", "attention": "COMBO"}

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        attention: (
            ModelAttentionBackend_Attentions | tuple[str, int]
        ) = "pytorch attention",
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.attention = attention
        self.__validate()

    def __validate(self) -> None:
        validate_choice("attention", self.attention, MODELATTENTIONBACKEND_ATTENTIONS)


class RenormCFG(BaseNode):
    __slots__ = ("model", "cfg_trunc", "renorm_cfg")
    CATEGORY = "model/patch"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {"model": "MODEL", "cfg_trunc": "FLOAT", "renorm_cfg": "FLOAT"}

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        cfg_trunc: float | tuple[str, int] = 100.0,
        renorm_cfg: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.cfg_trunc = cfg_trunc
        self.renorm_cfg = renorm_cfg
        self.__validate()

    def __validate(self) -> None:
        validate_range("cfg_trunc", self.cfg_trunc, 0.0, 100.0)
        validate_range("renorm_cfg", self.renorm_cfg, 0.0, 100.0)


ContextWindowsManual_Context_Schedules = Literal[
    "standard_static", "standard_uniform", "looped_uniform", "batched"
]
ContextWindowsManual_Fuse_Methods = Literal[
    "pyramid", "relative", "flat", "overlap-linear"
]
CONTEXTWINDOWSMANUAL_CONTEXT_SCHEDULES = frozenset(
    get_args(ContextWindowsManual_Context_Schedules)
)
CONTEXTWINDOWSMANUAL_FUSE_METHODS = frozenset(
    get_args(ContextWindowsManual_Fuse_Methods)
)


class ContextWindowsManual(BaseNode):
    __slots__ = (
        "model",
        "context_length",
        "context_overlap",
        "context_schedule",
        "context_stride",
        "closed_loop",
        "fuse_method",
        "dim",
        "freenoise",
        "cond_retain_index_list",
        "split_conds_to_windows",
        "latent_retain_index_list",
        "causal_window_fix",
    )
    CATEGORY = "model/patch"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "context_length": "INT",
        "context_overlap": "INT",
        "context_schedule": "COMBO",
        "context_stride": "INT",
        "closed_loop": "BOOLEAN",
        "fuse_method": "COMBO",
        "dim": "INT",
        "freenoise": "BOOLEAN",
        "cond_retain_index_list": "STRING",
        "split_conds_to_windows": "BOOLEAN",
        "latent_retain_index_list": "STRING",
        "causal_window_fix": "BOOLEAN",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        context_length: int | tuple[str, int] = 16,
        context_overlap: int | tuple[str, int] = 4,
        context_schedule: (
            ContextWindowsManual_Context_Schedules | tuple[str, int]
        ) = "standard_static",
        context_stride: int | tuple[str, int] = 1,
        closed_loop: bool | tuple[str, int] = False,
        fuse_method: ContextWindowsManual_Fuse_Methods | tuple[str, int] = "pyramid",
        dim: int | tuple[str, int] = 0,
        freenoise: bool | tuple[str, int] = False,
        cond_retain_index_list: str | tuple[str, int] = "",
        split_conds_to_windows: bool | tuple[str, int] = False,
        latent_retain_index_list: str | tuple[str, int] = "",
        causal_window_fix: bool | tuple[str, int] = True,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.context_length = context_length
        self.context_overlap = context_overlap
        self.context_schedule = context_schedule
        self.context_stride = context_stride
        self.closed_loop = closed_loop
        self.fuse_method = fuse_method
        self.dim = dim
        self.freenoise = freenoise
        self.cond_retain_index_list = cond_retain_index_list
        self.split_conds_to_windows = split_conds_to_windows
        self.latent_retain_index_list = latent_retain_index_list
        self.causal_window_fix = causal_window_fix
        self.__validate()

    def __validate(self) -> None:
        validate_range("context_length", self.context_length, 1)
        validate_range("context_overlap", self.context_overlap, 0)
        validate_choice(
            "context_schedule",
            self.context_schedule,
            CONTEXTWINDOWSMANUAL_CONTEXT_SCHEDULES,
        )
        validate_range("context_stride", self.context_stride, 1)
        validate_bool("closed_loop", self.closed_loop)
        validate_choice(
            "fuse_method", self.fuse_method, CONTEXTWINDOWSMANUAL_FUSE_METHODS
        )
        validate_range("dim", self.dim, 0, 5)
        validate_bool("freenoise", self.freenoise)
        validate_bool("split_conds_to_windows", self.split_conds_to_windows)
        validate_bool("causal_window_fix", self.causal_window_fix)


class ZImageFunControlnet(BaseNode):
    __slots__ = (
        "model",
        "model_patch",
        "vae",
        "strength",
        "image",
        "inpaint_image",
        "mask",
        "start_percent",
        "end_percent",
    )
    CATEGORY = "model/patch"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "model_patch": "MODEL_PATCH",
        "vae": "VAE",
        "strength": "FLOAT",
        "image": "IMAGE",
        "inpaint_image": "IMAGE",
        "mask": "MASK",
        "start_percent": "FLOAT",
        "end_percent": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        model_patch: tuple[str, int],
        vae: tuple[str, int],
        strength: float | tuple[str, int] = 1.0,
        image: tuple[str, int] | None = None,
        inpaint_image: tuple[str, int] | None = None,
        mask: tuple[str, int] | None = None,
        start_percent: float | tuple[str, int] = 0.0,
        end_percent: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.model_patch = model_patch
        self.vae = vae
        self.strength = strength
        if image is not None:
            self.image = image
        if inpaint_image is not None:
            self.inpaint_image = inpaint_image
        if mask is not None:
            self.mask = mask
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.__validate()

    def __validate(self) -> None:
        validate_range("strength", self.strength, -10.0, 10.0)
        validate_range("start_percent", self.start_percent, 0.0, 1.0)
        validate_range("end_percent", self.end_percent, 0.0, 1.0)


class ScaleROPE(BaseNode):
    __slots__ = (
        "model",
        "scale_x",
        "shift_x",
        "scale_y",
        "shift_y",
        "scale_t",
        "shift_t",
    )
    CATEGORY = "model/patch"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "scale_x": "FLOAT",
        "shift_x": "FLOAT",
        "scale_y": "FLOAT",
        "shift_y": "FLOAT",
        "scale_t": "FLOAT",
        "shift_t": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        scale_x: float | tuple[str, int] = 1.0,
        shift_x: float | tuple[str, int] = 0.0,
        scale_y: float | tuple[str, int] = 1.0,
        shift_y: float | tuple[str, int] = 0.0,
        scale_t: float | tuple[str, int] = 1.0,
        shift_t: float | tuple[str, int] = 0.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.scale_x = scale_x
        self.shift_x = shift_x
        self.scale_y = scale_y
        self.shift_y = shift_y
        self.scale_t = scale_t
        self.shift_t = shift_t
        self.__validate()

    def __validate(self) -> None:
        validate_range("scale_x", self.scale_x, 0.0, 100.0)
        validate_range("shift_x", self.shift_x, -256.0, 256.0)
        validate_range("scale_y", self.scale_y, 0.0, 100.0)
        validate_range("shift_y", self.shift_y, -256.0, 256.0)
        validate_range("scale_t", self.scale_t, 0.0, 100.0)
        validate_range("shift_t", self.shift_t, -256.0, 256.0)


BlockSparseAttention_Selections = Literal["sol-attn", "sla", "vsa"]


BlockSparseAttention_Sink_Conditionings = Literal[
    "exact_kv", "exact_kv_and_rows", "off"
]


BLOCKSPARSEATTENTION_SINK_CONDITIONINGS = frozenset(
    get_args(BlockSparseAttention_Sink_Conditionings)
)


class BlockSparseAttention(DynamicInputs, BaseNode):
    __slots__ = (
        "model",
        "selection",
        "selection_tau",
        "selection_keep_percent",
        "start_percent",
        "end_percent",
        "dense_blocks",
        "min_tokens",
        "extra_tokens",
        "sink_conditioning",
        "verbose",
    )
    CATEGORY = "model/patch"
    OUTPUTS = {"model": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_NAMES = {
        "selection_tau": "selection.tau",
        "selection_keep_percent": "selection.keep_percent",
    }
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "selection": "COMFY_DYNAMICCOMBO_V3",
        "selection_tau": "FLOAT",
        "selection_keep_percent": "FLOAT",
        "start_percent": "FLOAT",
        "end_percent": "FLOAT",
        "dense_blocks": "STRING",
        "min_tokens": "INT",
        "extra_tokens": "INT",
        "sink_conditioning": "COMBO",
        "verbose": "BOOLEAN",
    }
    DYNAMIC_INPUTS = {
        "selection": Combo(
            {
                "sol-attn": {"selection_tau": Number(default=1.3, low=0.0, high=4.0)},
                "sla": {
                    "selection_keep_percent": Number(default=10.0, low=0.5, high=95.0)
                },
                "vsa": {
                    "selection_keep_percent": Number(default=10.0, low=0.5, high=95.0)
                },
            }
        )
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        selection: BlockSparseAttention_Selections = "sol-attn",
        selection_tau: float | tuple[str, int] | None = None,
        selection_keep_percent: float | tuple[str, int] | None = None,
        start_percent: float | tuple[str, int] = 0.2,
        end_percent: float | tuple[str, int] = 1.0,
        dense_blocks: str | tuple[str, int] = "",
        min_tokens: int | tuple[str, int] = 12288,
        extra_tokens: int | tuple[str, int] = 256,
        sink_conditioning: (
            BlockSparseAttention_Sink_Conditionings | tuple[str, int]
        ) = "exact_kv_and_rows",
        verbose: bool | tuple[str, int] = False,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.selection = selection
        if selection_tau is not None:
            self.selection_tau = selection_tau
        if selection_keep_percent is not None:
            self.selection_keep_percent = selection_keep_percent
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.dense_blocks = dense_blocks
        self.min_tokens = min_tokens
        self.extra_tokens = extra_tokens
        self.sink_conditioning = sink_conditioning
        self.verbose = verbose
        self.__validate()

    def __validate(self) -> None:
        validate_range("start_percent", self.start_percent, 0.0, 1.0)
        validate_range("end_percent", self.end_percent, 0.0, 1.0)
        validate_range("min_tokens", self.min_tokens, 0, 1048576)
        validate_range("extra_tokens", self.extra_tokens, 0, 256)
        validate_choice(
            "sink_conditioning",
            self.sink_conditioning,
            BLOCKSPARSEATTENTION_SINK_CONDITIONINGS,
        )
        validate_bool("verbose", self.verbose)
        validate_dynamic(self)


__getattr__ = submodules_of(__name__)
