from typing import Literal, get_args

from ..base import BaseNode
from ..validation import validate_choice

SeedVR2PostProcessing_Color_Correction_Methods = Literal[
    "lab", "wavelet", "adain", "none"
]
SEEDVR2POSTPROCESSING_COLOR_CORRECTION_METHODS = frozenset(
    get_args(SeedVR2PostProcessing_Color_Correction_Methods)
)


class SeedVR2PostProcessing(BaseNode):
    __slots__ = ("images", "original_resized_images", "color_correction_method")
    CATEGORY = "image/post-processors"
    OUTPUTS = {"images": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_SOCKET_TYPES = {
        "images": "IMAGE",
        "original_resized_images": "IMAGE",
        "color_correction_method": "COMBO",
    }

    def __init__(
        self,
        node_id: str,
        images: tuple[str, int],
        original_resized_images: tuple[str, int],
        color_correction_method: (
            SeedVR2PostProcessing_Color_Correction_Methods | tuple[str, int]
        ) = "lab",
    ) -> None:
        super().__init__(node_id=node_id)
        self.images = images
        self.original_resized_images = original_resized_images
        self.color_correction_method = color_correction_method
        self.__validate()

    def __validate(self) -> None:
        validate_choice(
            "color_correction_method",
            self.color_correction_method,
            SEEDVR2POSTPROCESSING_COLOR_CORRECTION_METHODS,
        )
