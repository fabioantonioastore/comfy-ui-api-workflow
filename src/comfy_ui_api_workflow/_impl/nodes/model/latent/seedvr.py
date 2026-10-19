from typing import Literal
from ...base import BaseNode
from ...dynamic import Combo, DynamicInputs, Number, validate_dynamic
from ...validation import validate_range


class SeedVR2TemporalMerge(BaseNode):
    __slots__ = ("latents", "temporal_overlap")
    CATEGORY = "model/latent/seedvr"
    OUTPUTS = {"latent": 0}
    RETURN_TYPES = ("LATENT",)
    INPUT_SOCKET_TYPES = {"latents": "LATENT", "temporal_overlap": "INT"}

    def __init__(
        self, node_id: str, latents: tuple[str, int], temporal_overlap: tuple[str, int]
    ) -> None:
        super().__init__(node_id=node_id)
        self.latents = latents
        self.temporal_overlap = temporal_overlap


SeedVR2TemporalChunk_Chunking_Modes = Literal["auto", "manual"]


class SeedVR2TemporalChunk(DynamicInputs, BaseNode):
    __slots__ = (
        "latent",
        "temporal_overlap",
        "chunking_mode",
        "chunking_mode_frames_per_chunk",
    )
    CATEGORY = "model/latent/seedvr"
    OUTPUTS = {"latents": 0, "temporal_overlap": 1}
    RETURN_TYPES = (
        "LATENT",
        "INT",
    )
    INPUT_NAMES = {"chunking_mode_frames_per_chunk": "chunking_mode.frames_per_chunk"}
    INPUT_SOCKET_TYPES = {
        "latent": "LATENT",
        "temporal_overlap": "INT",
        "chunking_mode": "COMFY_DYNAMICCOMBO_V3",
        "chunking_mode_frames_per_chunk": "INT",
    }
    DYNAMIC_INPUTS = {
        "chunking_mode": Combo(
            {
                "auto": {},
                "manual": {
                    "chunking_mode_frames_per_chunk": Number(
                        default=21, low=1, high=16384
                    )
                },
            }
        )
    }

    def __init__(
        self,
        node_id: str,
        latent: tuple[str, int],
        temporal_overlap: int | tuple[str, int] = 0,
        chunking_mode: SeedVR2TemporalChunk_Chunking_Modes = "auto",
        chunking_mode_frames_per_chunk: int | tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.latent = latent
        self.temporal_overlap = temporal_overlap
        self.chunking_mode = chunking_mode
        if chunking_mode_frames_per_chunk is not None:
            self.chunking_mode_frames_per_chunk = chunking_mode_frames_per_chunk
        self.__validate()

    def __validate(self) -> None:
        validate_range("temporal_overlap", self.temporal_overlap, 0, 16384)
        validate_dynamic(self)
