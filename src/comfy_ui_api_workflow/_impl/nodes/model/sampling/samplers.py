from typing import Literal, get_args

from ...base import BaseNode
from ...model.sampling import KSAMPLER_SAMPLER_NAMES, KSampler_Sampler_Names
from ...validation import validate_bool, validate_choice, validate_range


class KSamplerSelect(BaseNode):
    __slots__ = ("sampler_name",)
    CATEGORY = "model/sampling/samplers"
    OUTPUTS = {"SAMPLER": 0}
    RETURN_TYPES = ("SAMPLER",)
    INPUT_SOCKET_TYPES = {"sampler_name": "COMBO"}

    def __init__(
        self, node_id: str, sampler_name: KSampler_Sampler_Names | tuple[str, int]
    ) -> None:
        super().__init__(node_id=node_id)
        self.sampler_name = sampler_name
        self.__validate()

    def __validate(self) -> None:
        validate_choice("sampler_name", self.sampler_name, KSAMPLER_SAMPLER_NAMES)


class SamplerEulerAncestral(BaseNode):
    __slots__ = ("eta", "s_noise")
    CATEGORY = "model/sampling/samplers"
    OUTPUTS = {"SAMPLER": 0}
    RETURN_TYPES = ("SAMPLER",)
    INPUT_SOCKET_TYPES = {"eta": "FLOAT", "s_noise": "FLOAT"}

    def __init__(
        self,
        node_id: str,
        eta: float | tuple[str, int] = 1.0,
        s_noise: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.eta = eta
        self.s_noise = s_noise
        self.__validate()

    def __validate(self) -> None:
        validate_range("eta", self.eta, 0.0, 100.0)
        validate_range("s_noise", self.s_noise, 0.0, 100.0)


class SamplerEulerAncestralCFGPP(BaseNode):
    __slots__ = ("eta", "s_noise")
    CATEGORY = "model/sampling/samplers"
    OUTPUTS = {"SAMPLER": 0}
    RETURN_TYPES = ("SAMPLER",)
    INPUT_SOCKET_TYPES = {"eta": "FLOAT", "s_noise": "FLOAT"}

    def __init__(
        self,
        node_id: str,
        eta: float | tuple[str, int] = 1.0,
        s_noise: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.eta = eta
        self.s_noise = s_noise
        self.__validate()

    def __validate(self) -> None:
        validate_range("eta", self.eta, 0.0, 1.0)
        validate_range("s_noise", self.s_noise, 0.0, 10.0)


class SamplerLMS(BaseNode):
    __slots__ = ("order",)
    CATEGORY = "model/sampling/samplers"
    OUTPUTS = {"SAMPLER": 0}
    RETURN_TYPES = ("SAMPLER",)
    INPUT_SOCKET_TYPES = {"order": "INT"}

    def __init__(self, node_id: str, order: int | tuple[str, int] = 4) -> None:
        super().__init__(node_id=node_id)
        self.order = order
        self.__validate()

    def __validate(self) -> None:
        validate_range("order", self.order, 1, 100)


SamplerDPMPP_3M_SDE_Noise_Devices = Literal["gpu", "cpu"]
SAMPLERDPMPP_3M_SDE_NOISE_DEVICES = frozenset(
    get_args(SamplerDPMPP_3M_SDE_Noise_Devices)
)


class SamplerDPMPP_3M_SDE(BaseNode):
    __slots__ = ("noise_device", "eta", "s_noise")
    CATEGORY = "model/sampling/samplers"
    OUTPUTS = {"SAMPLER": 0}
    RETURN_TYPES = ("SAMPLER",)
    INPUT_SOCKET_TYPES = {"eta": "FLOAT", "s_noise": "FLOAT", "noise_device": "COMBO"}

    def __init__(
        self,
        node_id: str,
        noise_device: SamplerDPMPP_3M_SDE_Noise_Devices | tuple[str, int],
        eta: float | tuple[str, int] = 1.0,
        s_noise: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.noise_device = noise_device
        self.eta = eta
        self.s_noise = s_noise
        self.__validate()

    def __validate(self) -> None:
        validate_choice(
            "noise_device", self.noise_device, SAMPLERDPMPP_3M_SDE_NOISE_DEVICES
        )
        validate_range("eta", self.eta, 0.0, 100.0)
        validate_range("s_noise", self.s_noise, 0.0, 100.0)


SamplerDPMPP_2M_SDE_Solver_Types = Literal["midpoint", "heun"]
SamplerDPMPP_2M_SDE_Noise_Devices = Literal["gpu", "cpu"]
SAMPLERDPMPP_2M_SDE_SOLVER_TYPES = frozenset(get_args(SamplerDPMPP_2M_SDE_Solver_Types))
SAMPLERDPMPP_2M_SDE_NOISE_DEVICES = frozenset(
    get_args(SamplerDPMPP_2M_SDE_Noise_Devices)
)


class SamplerDPMPP_2M_SDE(BaseNode):
    __slots__ = ("solver_type", "noise_device", "eta", "s_noise")
    CATEGORY = "model/sampling/samplers"
    OUTPUTS = {"SAMPLER": 0}
    RETURN_TYPES = ("SAMPLER",)
    INPUT_SOCKET_TYPES = {
        "solver_type": "COMBO",
        "eta": "FLOAT",
        "s_noise": "FLOAT",
        "noise_device": "COMBO",
    }

    def __init__(
        self,
        node_id: str,
        solver_type: SamplerDPMPP_2M_SDE_Solver_Types | tuple[str, int],
        noise_device: SamplerDPMPP_2M_SDE_Noise_Devices | tuple[str, int],
        eta: float | tuple[str, int] = 1.0,
        s_noise: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.solver_type = solver_type
        self.noise_device = noise_device
        self.eta = eta
        self.s_noise = s_noise
        self.__validate()

    def __validate(self) -> None:
        validate_choice(
            "solver_type", self.solver_type, SAMPLERDPMPP_2M_SDE_SOLVER_TYPES
        )
        validate_choice(
            "noise_device", self.noise_device, SAMPLERDPMPP_2M_SDE_NOISE_DEVICES
        )
        validate_range("eta", self.eta, 0.0, 100.0)
        validate_range("s_noise", self.s_noise, 0.0, 100.0)


SamplerDPMPP_SDE_Noise_Devices = Literal["gpu", "cpu"]
SAMPLERDPMPP_SDE_NOISE_DEVICES = frozenset(get_args(SamplerDPMPP_SDE_Noise_Devices))


class SamplerDPMPP_SDE(BaseNode):
    __slots__ = ("noise_device", "eta", "s_noise", "r")
    CATEGORY = "model/sampling/samplers"
    OUTPUTS = {"SAMPLER": 0}
    RETURN_TYPES = ("SAMPLER",)
    INPUT_SOCKET_TYPES = {
        "eta": "FLOAT",
        "s_noise": "FLOAT",
        "r": "FLOAT",
        "noise_device": "COMBO",
    }

    def __init__(
        self,
        node_id: str,
        noise_device: SamplerDPMPP_SDE_Noise_Devices | tuple[str, int],
        eta: float | tuple[str, int] = 1.0,
        s_noise: float | tuple[str, int] = 1.0,
        r: float | tuple[str, int] = 0.5,
    ) -> None:
        super().__init__(node_id=node_id)
        self.noise_device = noise_device
        self.eta = eta
        self.s_noise = s_noise
        self.r = r
        self.__validate()

    def __validate(self) -> None:
        validate_choice(
            "noise_device", self.noise_device, SAMPLERDPMPP_SDE_NOISE_DEVICES
        )
        validate_range("eta", self.eta, 0.0, 100.0)
        validate_range("s_noise", self.s_noise, 0.0, 100.0)
        validate_range("r", self.r, 0.0, 100.0)


class SamplerDPMPP_2S_Ancestral(BaseNode):
    __slots__ = ("eta", "s_noise")
    CATEGORY = "model/sampling/samplers"
    OUTPUTS = {"SAMPLER": 0}
    RETURN_TYPES = ("SAMPLER",)
    INPUT_SOCKET_TYPES = {"eta": "FLOAT", "s_noise": "FLOAT"}

    def __init__(
        self,
        node_id: str,
        eta: float | tuple[str, int] = 1.0,
        s_noise: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.eta = eta
        self.s_noise = s_noise
        self.__validate()

    def __validate(self) -> None:
        validate_range("eta", self.eta, 0.0, 100.0)
        validate_range("s_noise", self.s_noise, 0.0, 100.0)


class SamplerDPMAdaptative(BaseNode):
    __slots__ = (
        "order",
        "rtol",
        "atol",
        "h_init",
        "pcoeff",
        "icoeff",
        "dcoeff",
        "accept_safety",
        "eta",
        "s_noise",
    )
    CATEGORY = "model/sampling/samplers"
    OUTPUTS = {"SAMPLER": 0}
    RETURN_TYPES = ("SAMPLER",)
    INPUT_SOCKET_TYPES = {
        "order": "INT",
        "rtol": "FLOAT",
        "atol": "FLOAT",
        "h_init": "FLOAT",
        "pcoeff": "FLOAT",
        "icoeff": "FLOAT",
        "dcoeff": "FLOAT",
        "accept_safety": "FLOAT",
        "eta": "FLOAT",
        "s_noise": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        order: int | tuple[str, int] = 3,
        rtol: float | tuple[str, int] = 0.05,
        atol: float | tuple[str, int] = 0.0078,
        h_init: float | tuple[str, int] = 0.05,
        pcoeff: float | tuple[str, int] = 0.0,
        icoeff: float | tuple[str, int] = 1.0,
        dcoeff: float | tuple[str, int] = 0.0,
        accept_safety: float | tuple[str, int] = 0.81,
        eta: float | tuple[str, int] = 0.0,
        s_noise: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.order = order
        self.rtol = rtol
        self.atol = atol
        self.h_init = h_init
        self.pcoeff = pcoeff
        self.icoeff = icoeff
        self.dcoeff = dcoeff
        self.accept_safety = accept_safety
        self.eta = eta
        self.s_noise = s_noise
        self.__validate()

    def __validate(self) -> None:
        validate_range("order", self.order, 2, 3)
        validate_range("rtol", self.rtol, 0.0, 100.0)
        validate_range("atol", self.atol, 0.0, 100.0)
        validate_range("h_init", self.h_init, 0.0, 100.0)
        validate_range("pcoeff", self.pcoeff, 0.0, 100.0)
        validate_range("icoeff", self.icoeff, 0.0, 100.0)
        validate_range("dcoeff", self.dcoeff, 0.0, 100.0)
        validate_range("accept_safety", self.accept_safety, 0.0, 100.0)
        validate_range("eta", self.eta, 0.0, 100.0)
        validate_range("s_noise", self.s_noise, 0.0, 100.0)


SamplerER_SDE_Solver_Types = Literal["ER-SDE", "Reverse-time SDE", "ODE"]
SAMPLERER_SDE_SOLVER_TYPES = frozenset(get_args(SamplerER_SDE_Solver_Types))


class SamplerER_SDE(BaseNode):
    __slots__ = ("solver_type", "max_stage", "eta", "s_noise")
    CATEGORY = "model/sampling/samplers"
    OUTPUTS = {"SAMPLER": 0}
    RETURN_TYPES = ("SAMPLER",)
    INPUT_SOCKET_TYPES = {
        "solver_type": "COMBO",
        "max_stage": "INT",
        "eta": "FLOAT",
        "s_noise": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        solver_type: SamplerER_SDE_Solver_Types | tuple[str, int],
        max_stage: int | tuple[str, int] = 3,
        eta: float | tuple[str, int] = 1.0,
        s_noise: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.solver_type = solver_type
        self.max_stage = max_stage
        self.eta = eta
        self.s_noise = s_noise
        self.__validate()

    def __validate(self) -> None:
        validate_choice("solver_type", self.solver_type, SAMPLERER_SDE_SOLVER_TYPES)
        validate_range("max_stage", self.max_stage, 1, 3)
        validate_range("eta", self.eta, 0.0, 10.0)
        validate_range("s_noise", self.s_noise, 0.0, 100.0)


class SamplerSASolver(BaseNode):
    __slots__ = (
        "model",
        "use_pece",
        "simple_order_2",
        "eta",
        "sde_start_percent",
        "sde_end_percent",
        "s_noise",
        "predictor_order",
        "corrector_order",
    )
    CATEGORY = "model/sampling/samplers"
    OUTPUTS = {"SAMPLER": 0}
    RETURN_TYPES = ("SAMPLER",)
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "eta": "FLOAT",
        "sde_start_percent": "FLOAT",
        "sde_end_percent": "FLOAT",
        "s_noise": "FLOAT",
        "predictor_order": "INT",
        "corrector_order": "INT",
        "use_pece": "BOOLEAN",
        "simple_order_2": "BOOLEAN",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        use_pece: bool | tuple[str, int],
        simple_order_2: bool | tuple[str, int],
        eta: float | tuple[str, int] = 1.0,
        sde_start_percent: float | tuple[str, int] = 0.2,
        sde_end_percent: float | tuple[str, int] = 0.8,
        s_noise: float | tuple[str, int] = 1.0,
        predictor_order: int | tuple[str, int] = 3,
        corrector_order: int | tuple[str, int] = 4,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.use_pece = use_pece
        self.simple_order_2 = simple_order_2
        self.eta = eta
        self.sde_start_percent = sde_start_percent
        self.sde_end_percent = sde_end_percent
        self.s_noise = s_noise
        self.predictor_order = predictor_order
        self.corrector_order = corrector_order
        self.__validate()

    def __validate(self) -> None:
        validate_bool("use_pece", self.use_pece)
        validate_bool("simple_order_2", self.simple_order_2)
        validate_range("eta", self.eta, 0.0, 10.0)
        validate_range("sde_start_percent", self.sde_start_percent, 0.0, 1.0)
        validate_range("sde_end_percent", self.sde_end_percent, 0.0, 1.0)
        validate_range("s_noise", self.s_noise, 0.0, 100.0)
        validate_range("predictor_order", self.predictor_order, 1, 6)
        validate_range("corrector_order", self.corrector_order, 0, 6)


SamplerSEEDS2_Solver_Types = Literal["phi_1", "phi_2"]
SAMPLERSEEDS2_SOLVER_TYPES = frozenset(get_args(SamplerSEEDS2_Solver_Types))


class SamplerSEEDS2(BaseNode):
    __slots__ = ("solver_type", "eta", "s_noise", "r")
    CATEGORY = "model/sampling/samplers"
    OUTPUTS = {"SAMPLER": 0}
    RETURN_TYPES = ("SAMPLER",)
    INPUT_SOCKET_TYPES = {
        "solver_type": "COMBO",
        "eta": "FLOAT",
        "s_noise": "FLOAT",
        "r": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        solver_type: SamplerSEEDS2_Solver_Types | tuple[str, int],
        eta: float | tuple[str, int] = 1.0,
        s_noise: float | tuple[str, int] = 1.0,
        r: float | tuple[str, int] = 0.5,
    ) -> None:
        super().__init__(node_id=node_id)
        self.solver_type = solver_type
        self.eta = eta
        self.s_noise = s_noise
        self.r = r
        self.__validate()

    def __validate(self) -> None:
        validate_choice("solver_type", self.solver_type, SAMPLERSEEDS2_SOLVER_TYPES)
        validate_range("eta", self.eta, 0.0, 100.0)
        validate_range("s_noise", self.s_noise, 0.0, 100.0)
        validate_range("r", self.r, 0.01, 1.0)


SamplerLCMUpscale_Upscale_Methods = Literal[
    "bislerp", "nearest-exact", "bilinear", "area", "bicubic"
]
SAMPLERLCMUPSCALE_UPSCALE_METHODS = frozenset(
    get_args(SamplerLCMUpscale_Upscale_Methods)
)


class SamplerLCMUpscale(BaseNode):
    __slots__ = ("upscale_method", "scale_ratio", "scale_steps")
    CATEGORY = "model/sampling/samplers"
    OUTPUTS = {"SAMPLER": 0}
    RETURN_TYPES = ("SAMPLER",)
    INPUT_SOCKET_TYPES = {
        "scale_ratio": "FLOAT",
        "scale_steps": "INT",
        "upscale_method": "COMBO",
    }

    def __init__(
        self,
        node_id: str,
        upscale_method: SamplerLCMUpscale_Upscale_Methods | tuple[str, int],
        scale_ratio: float | tuple[str, int] = 1.0,
        scale_steps: int | tuple[str, int] = -1,
    ) -> None:
        super().__init__(node_id=node_id)
        self.upscale_method = upscale_method
        self.scale_ratio = scale_ratio
        self.scale_steps = scale_steps
        self.__validate()

    def __validate(self) -> None:
        validate_choice(
            "upscale_method", self.upscale_method, SAMPLERLCMUPSCALE_UPSCALE_METHODS
        )
        validate_range("scale_ratio", self.scale_ratio, 0.1, 20.0)
        validate_range("scale_steps", self.scale_steps, -1, 1000)


class SamplerLCM(BaseNode):
    __slots__ = ("s_noise", "s_noise_end", "noise_clip_std")
    CATEGORY = "model/sampling/samplers"
    OUTPUTS = {"SAMPLER": 0}
    RETURN_TYPES = ("SAMPLER",)
    INPUT_SOCKET_TYPES = {
        "s_noise": "FLOAT",
        "s_noise_end": "FLOAT",
        "noise_clip_std": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        s_noise: float | tuple[str, int] = 1.0,
        s_noise_end: float | tuple[str, int] = 1.0,
        noise_clip_std: float | tuple[str, int] = 0.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.s_noise = s_noise
        self.s_noise_end = s_noise_end
        self.noise_clip_std = noise_clip_std
        self.__validate()

    def __validate(self) -> None:
        validate_range("s_noise", self.s_noise, 0.0, 64.0)
        validate_range("s_noise_end", self.s_noise_end, 0.0, 64.0)
        validate_range("noise_clip_std", self.noise_clip_std, 0.0, 10.0)


class SamplerARVideo(BaseNode):
    __slots__ = ("num_frame_per_block",)
    CATEGORY = "model/sampling/samplers"
    OUTPUTS = {"SAMPLER": 0}
    RETURN_TYPES = ("SAMPLER",)
    INPUT_SOCKET_TYPES = {"num_frame_per_block": "INT"}

    def __init__(
        self, node_id: str, num_frame_per_block: int | tuple[str, int] = 1
    ) -> None:
        super().__init__(node_id=node_id)
        self.num_frame_per_block = num_frame_per_block
        self.__validate()

    def __validate(self) -> None:
        validate_range("num_frame_per_block", self.num_frame_per_block, 1, 64)


class VOIDSampler(BaseNode):
    __slots__ = ()
    CATEGORY = "model/sampling/samplers"
    OUTPUTS = {"SAMPLER": 0}
    RETURN_TYPES = ("SAMPLER",)
