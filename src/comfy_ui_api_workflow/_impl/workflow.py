from typing import Any, Literal

from .nodes.base import BaseNode, NodeOutput
from .nodes.model.sampling import KSampler
from .errors.node import NodeNotHaveOutput


class Workflow:
    def __init__(self) -> None:
        self.comfy_ui_workflow: dict[str, Any] = {}
        self.nodes: dict[str, BaseNode] = {}
        self.__next_index = 0

    def get_node_output(self, node_id: str) -> NodeOutput:
        try:
            return getattr(self.nodes[node_id], "output")
        except AttributeError:
            raise NodeNotHaveOutput(f"node: {node_id} don't have a output")

    def add_node(self, node_type: type[BaseNode], **kwargs: Any) -> str:
        node_id = node_type.__name__ + f"_{self.__next_index}"
        self.__next_index += 1
        self.nodes[node_id] = node_type(node_id=node_id, **kwargs)
        self.comfy_ui_workflow[node_id] = self.nodes[node_id].get_comfy_ui_workflow_format()
        return node_id

    def add_ksampler(
        self,
        model: NodeOutput,
        sampler_name: Literal[
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
        ],
        scheduler: Literal[
            "simple",
            "sgm_uniform",
            "karras",
            "exponential",
            "ddim_uniform",
            "beta",
            "normal",
            "linear_quadratic",
            "kl_optimal",
        ],
        positive: NodeOutput,
        negative: NodeOutput,
        latent_image: NodeOutput,
        seed: int = 0,
        steps: int = 20,
        cfg: float = 8.0,
        denoise: float = 1.0,
    ) -> str:
        node_id = KSampler.__name__ + str(self.__next_index)
        self.__next_index += 1
        self.nodes[node_id] = KSampler(
            node_id=node_id,
            model=model,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            denoise=denoise,
            seed=seed,
            steps=steps,
            cfg=cfg,
            latent_image=latent_image,
        )
        self.comfy_ui_workflow[node_id] = self.nodes[node_id]
        return node_id
