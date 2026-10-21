from typing import Literal, get_args

from ...base import BaseNode
from ...validation import validate_choice, validate_range

KSampler_Sampler_Names = Literal[
    "euler",
    "euler_cfg_pp",
    "euler_ancestral",
    "euler_ancestral_cfg_pp",
    "heun",
    "heunpp2",
    "exp_heun_2_x0",
    "exp_heun_2_x0_sde",
    "dpm_2",
    "dpm_2_ancestral",
    "lms",
    "dpm_fast",
    "dpm_adaptive",
    "dpmpp_2s_ancestral",
    "dpmpp_2s_ancestral_cfg_pp",
    "dpmpp_sde",
    "dpmpp_sde_gpu",
    "dpmpp_2m",
    "dpmpp_2m_cfg_pp",
    "dpmpp_2m_sde",
    "dpmpp_2m_sde_gpu",
    "dpmpp_2m_sde_heun",
    "dpmpp_2m_sde_heun_gpu",
    "dpmpp_3m_sde",
    "dpmpp_3m_sde_gpu",
    "ddpm",
    "lcm",
    "ipndm",
    "ipndm_v",
    "deis",
    "cfgpp_ud10_ab",
    "res_multistep",
    "res_multistep_cfg_pp",
    "res_multistep_ancestral",
    "res_multistep_ancestral_cfg_pp",
    "gradient_estimation",
    "gradient_estimation_cfg_pp",
    "er_sde",
    "seeds_2",
    "seeds_3",
    "sa_solver",
    "sa_solver_pece",
    "ddim",
    "uni_pc",
    "uni_pc_bh2",
]
KSampler_Schedulers = Literal[
    "simple",
    "sgm_uniform",
    "karras",
    "exponential",
    "ddim_uniform",
    "beta",
    "normal",
    "linear_quadratic",
    "kl_optimal",
]
KSAMPLER_SAMPLER_NAMES = frozenset(get_args(KSampler_Sampler_Names))
KSAMPLER_SCHEDULERS = frozenset(get_args(KSampler_Schedulers))


class KSampler(BaseNode):
    __slots__ = (
        "model",
        "sampler_name",
        "scheduler",
        "positive",
        "negative",
        "latent_image",
        "seed",
        "steps",
        "cfg",
        "denoise",
    )
    CATEGORY = "model/sampling"
    OUTPUTS = {"LATENT": 0}
    RETURN_TYPES = ("LATENT",)
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "seed": "INT",
        "steps": "INT",
        "cfg": "FLOAT",
        "sampler_name": "V1 COMBO",
        "scheduler": "V1 COMBO",
        "positive": "CONDITIONING",
        "negative": "CONDITIONING",
        "latent_image": "LATENT",
        "denoise": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        sampler_name: KSampler_Sampler_Names | tuple[str, int],
        scheduler: KSampler_Schedulers | tuple[str, int],
        positive: tuple[str, int],
        negative: tuple[str, int],
        latent_image: tuple[str, int],
        seed: int | tuple[str, int] = 0,
        steps: int | tuple[str, int] = 20,
        cfg: float | tuple[str, int] = 8.0,
        denoise: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.sampler_name = sampler_name
        self.scheduler = scheduler
        self.positive = positive
        self.negative = negative
        self.latent_image = latent_image
        self.seed = seed
        self.steps = steps
        self.cfg = cfg
        self.denoise = denoise
        self.__validate()

    def __validate(self) -> None:
        validate_range("seed", self.seed, 0, 18_446_744_073_709_551_615)
        validate_range("steps", self.steps, 1, 10_000)
        validate_range("cfg", self.cfg, 0.0, 100.0)
        validate_range("denoise", self.denoise, 0.0, 1.0)
        validate_choice("sampler_name", self.sampler_name, KSAMPLER_SAMPLER_NAMES)
        validate_choice("scheduler", self.scheduler, KSAMPLER_SCHEDULERS)


KSamplerAdvanced_Add_Noises = Literal["enable", "disable"]
KSamplerAdvanced_Return_With_Leftover_Noises = Literal["disable", "enable"]
KSAMPLERADVANCED_ADD_NOISES = frozenset(get_args(KSamplerAdvanced_Add_Noises))
KSAMPLERADVANCED_RETURN_WITH_LEFTOVER_NOISES = frozenset(
    get_args(KSamplerAdvanced_Return_With_Leftover_Noises)
)


class KSamplerAdvanced(BaseNode):
    __slots__ = (
        "model",
        "add_noise",
        "sampler_name",
        "scheduler",
        "positive",
        "negative",
        "latent_image",
        "return_with_leftover_noise",
        "noise_seed",
        "steps",
        "cfg",
        "start_at_step",
        "end_at_step",
    )
    CATEGORY = "model/sampling"
    OUTPUTS = {"LATENT": 0}
    RETURN_TYPES = ("LATENT",)
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "add_noise": "V1 COMBO",
        "noise_seed": "INT",
        "steps": "INT",
        "cfg": "FLOAT",
        "sampler_name": "V1 COMBO",
        "scheduler": "V1 COMBO",
        "positive": "CONDITIONING",
        "negative": "CONDITIONING",
        "latent_image": "LATENT",
        "start_at_step": "INT",
        "end_at_step": "INT",
        "return_with_leftover_noise": "V1 COMBO",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        add_noise: KSamplerAdvanced_Add_Noises | tuple[str, int],
        sampler_name: KSampler_Sampler_Names | tuple[str, int],
        scheduler: KSampler_Schedulers | tuple[str, int],
        positive: tuple[str, int],
        negative: tuple[str, int],
        latent_image: tuple[str, int],
        return_with_leftover_noise: (
            KSamplerAdvanced_Return_With_Leftover_Noises | tuple[str, int]
        ),
        noise_seed: int | tuple[str, int] = 0,
        steps: int | tuple[str, int] = 20,
        cfg: float | tuple[str, int] = 8.0,
        start_at_step: int | tuple[str, int] = 0,
        end_at_step: int | tuple[str, int] = 10_000,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.add_noise = add_noise
        self.sampler_name = sampler_name
        self.scheduler = scheduler
        self.positive = positive
        self.negative = negative
        self.latent_image = latent_image
        self.return_with_leftover_noise = return_with_leftover_noise
        self.noise_seed = noise_seed
        self.steps = steps
        self.cfg = cfg
        self.start_at_step = start_at_step
        self.end_at_step = end_at_step
        self.__validate()

    def __validate(self) -> None:
        validate_choice("add_noise", self.add_noise, KSAMPLERADVANCED_ADD_NOISES)
        validate_choice("sampler_name", self.sampler_name, KSAMPLER_SAMPLER_NAMES)
        validate_choice("scheduler", self.scheduler, KSAMPLER_SCHEDULERS)
        validate_choice(
            "return_with_leftover_noise",
            self.return_with_leftover_noise,
            KSAMPLERADVANCED_RETURN_WITH_LEFTOVER_NOISES,
        )
        validate_range("noise_seed", self.noise_seed, 0, 18_446_744_073_709_551_615)
        validate_range("steps", self.steps, 1, 10_000)
        validate_range("cfg", self.cfg, 0.0, 100.0)
        validate_range("start_at_step", self.start_at_step, 0, 10_000)
        validate_range("end_at_step", self.end_at_step, 0, 10_000)


from . import custom, guiders, noise, samplers, schedulers, sigmas
