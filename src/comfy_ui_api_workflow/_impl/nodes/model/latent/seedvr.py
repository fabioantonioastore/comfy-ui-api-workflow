from ...base import BaseNode


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
