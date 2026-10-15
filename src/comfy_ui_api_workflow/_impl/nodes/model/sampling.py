from typing import Literal, get_args

from ..base import BaseNode, NodeOutput
from ...enums import OutputTypes
from ...errors.node import InvalidNodeParameterValue

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
        "output",
    )
    CATEGORY = "model/sampling"

    def __init__(
        self,
        node_id: str,
        model: NodeOutput,
        sampler_name: KSampler_Sampler_Names,
        scheduler: KSampler_Schedulers,
        positive: NodeOutput,
        negative: NodeOutput,
        latent_image: NodeOutput,
        seed: int = 0,
        steps: int = 20,
        cfg: float = 8.0,
        denoise: float = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model.get(output=OutputTypes.MODEL)
        self.sampler_name = sampler_name
        self.scheduler = scheduler
        self.positive = positive.get(output=OutputTypes.CONDITIONING)
        self.negative = negative.get(output=OutputTypes.CONDITIONING)
        self.latent_image = latent_image.get(output=OutputTypes.LATENT)
        self.seed = seed
        self.steps = steps
        self.cfg = cfg
        self.denoise = denoise
        self.output = NodeOutput(node_id=self.node_id, outputs={OutputTypes.LATENT: 0})
        self.__validate()

    def __validate(self) -> None:
        self.__validate_range(
            name="seed",
            range=self.seed,
            range_min=0,
            range_max=18_446_744_073_709_551_615,
        )
        self.__validate_range(
            name="steps", range=self.steps, range_min=1, range_max=10_000
        )
        self.__validate_range(
            name="cfg", range=self.cfg, range_min=0.0, range_max=100.0
        )
        self.__validate_range(
            name="denoise", range=self.denoise, range_min=0.0, range_max=1.0
        )
        self.__validate_sampler_name()
        self.__validate_scheduler()

    def __validate_range(
        self,
        name: str,
        range: int | float,
        range_min: int | float,
        range_max: int | float,
    ) -> None:
        if isinstance(range, bool):
            raise InvalidNodeParameterValue(f"{name} must be a number, not bool")
        if not (range_min <= range <= range_max):
            raise InvalidNodeParameterValue(
                f"{name} must be in closed range: [{range_min}, {range_max}]"
            )

    def __validate_sampler_name(self) -> None:
        if not self.sampler_name in KSAMPLER_SAMPLER_NAMES:
            raise InvalidNodeParameterValue(
                f"Invalid sampler name, must be one of the following strings: {KSAMPLER_SAMPLER_NAMES}"
            )

    def __validate_scheduler(self) -> None:
        if not self.scheduler in KSAMPLER_SCHEDULERS:
            raise InvalidNodeParameterValue(
                f"Invalid scheduler, must be one of the following strings: {KSAMPLER_SCHEDULERS}"
            )
