from ..base import BaseNode


class StableCascade_SuperResolutionControlnet(BaseNode):
    __slots__ = ("image", "vae")
    CATEGORY = "experimental/stable cascade"
    OUTPUTS = {"controlnet_input": 0, "stage_c": 1, "stage_b": 2}
    RETURN_TYPES = ("IMAGE", "LATENT", "LATENT")
    INPUT_SOCKET_TYPES = {"image": "IMAGE", "vae": "VAE"}

    def __init__(
        self, node_id: str, image: tuple[str, int], vae: tuple[str, int]
    ) -> None:
        super().__init__(node_id=node_id)
        self.image = image
        self.vae = vae
