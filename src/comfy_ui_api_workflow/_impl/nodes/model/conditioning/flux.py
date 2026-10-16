from typing import Literal, get_args

from ...base import BaseNode
from ...validation import validate_choice, validate_range


class CLIPTextEncodeFlux(BaseNode):
    __slots__ = ("clip", "clip_l", "t5xxl", "guidance")
    CATEGORY = "model/conditioning/flux"
    OUTPUTS = {"CONDITIONING": 0}
    RETURN_TYPES = ("CONDITIONING",)
    INPUT_SOCKET_TYPES = {
        "clip": "CLIP",
        "clip_l": "STRING",
        "t5xxl": "STRING",
        "guidance": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        clip: tuple[str, int],
        clip_l: str | tuple[str, int],
        t5xxl: str | tuple[str, int],
        guidance: float | tuple[str, int] = 3.5,
    ) -> None:
        super().__init__(node_id=node_id)
        self.clip = clip
        self.clip_l = clip_l
        self.t5xxl = t5xxl
        self.guidance = guidance
        self.__validate()

    def __validate(self) -> None:
        validate_range("guidance", self.guidance, 0.0, 100.0)


class FluxGuidance(BaseNode):
    __slots__ = ("conditioning", "guidance")
    CATEGORY = "model/conditioning/flux"
    OUTPUTS = {"CONDITIONING": 0}
    RETURN_TYPES = ("CONDITIONING",)
    INPUT_SOCKET_TYPES = {"conditioning": "CONDITIONING", "guidance": "FLOAT"}

    def __init__(
        self,
        node_id: str,
        conditioning: tuple[str, int],
        guidance: float | tuple[str, int] = 3.5,
    ) -> None:
        super().__init__(node_id=node_id)
        self.conditioning = conditioning
        self.guidance = guidance
        self.__validate()

    def __validate(self) -> None:
        validate_range("guidance", self.guidance, 0.0, 100.0)


class FluxDisableGuidance(BaseNode):
    __slots__ = ("conditioning",)
    CATEGORY = "model/conditioning/flux"
    OUTPUTS = {"CONDITIONING": 0}
    RETURN_TYPES = ("CONDITIONING",)
    INPUT_SOCKET_TYPES = {"conditioning": "CONDITIONING"}

    def __init__(self, node_id: str, conditioning: tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.conditioning = conditioning


class FluxKontextImageScale(BaseNode):
    __slots__ = ("image",)
    CATEGORY = "model/conditioning/flux"
    OUTPUTS = {"IMAGE": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_SOCKET_TYPES = {"image": "IMAGE"}

    def __init__(self, node_id: str, image: tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.image = image


FluxKontextMultiReferenceLatentMethod_Reference_Latents_Methods = Literal[
    "offset", "index", "uxo/uno", "index_timestep_zero"
]
FLUXKONTEXTMULTIREFERENCELATENTMETHOD_REFERENCE_LATENTS_METHODS = frozenset(
    get_args(FluxKontextMultiReferenceLatentMethod_Reference_Latents_Methods)
)


class FluxKontextMultiReferenceLatentMethod(BaseNode):
    __slots__ = ("conditioning", "reference_latents_method")
    CATEGORY = "model/conditioning/flux"
    OUTPUTS = {"CONDITIONING": 0}
    RETURN_TYPES = ("CONDITIONING",)
    INPUT_SOCKET_TYPES = {
        "conditioning": "CONDITIONING",
        "reference_latents_method": "COMBO",
    }

    def __init__(
        self,
        node_id: str,
        conditioning: tuple[str, int],
        reference_latents_method: (
            FluxKontextMultiReferenceLatentMethod_Reference_Latents_Methods
            | tuple[str, int]
        ),
    ) -> None:
        super().__init__(node_id=node_id)
        self.conditioning = conditioning
        self.reference_latents_method = reference_latents_method
        self.__validate()

    def __validate(self) -> None:
        validate_choice(
            "reference_latents_method",
            self.reference_latents_method,
            FLUXKONTEXTMULTIREFERENCELATENTMETHOD_REFERENCE_LATENTS_METHODS,
        )
