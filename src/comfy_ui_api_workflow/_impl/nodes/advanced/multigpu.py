from ..base import BaseNode
from ..validation import validate_not_empty, validate_range


class MultiGPU_WorkUnits(BaseNode):
    __slots__ = ("model", "max_gpus")
    CATEGORY = "advanced/multigpu"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {"model": "MODEL", "max_gpus": "INT"}

    def __init__(
        self, node_id: str, model: tuple[str, int], max_gpus: int | tuple[str, int] = 2
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.max_gpus = max_gpus
        self.__validate()

    def __validate(self) -> None:
        validate_range("max_gpus", self.max_gpus, 1)


class SelectModelDevice(BaseNode):
    __slots__ = ("model", "device")
    CATEGORY = "advanced/multigpu"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {"model": "MODEL", "device": "COMBO"}

    def __init__(
        self, node_id: str, model: tuple[str, int], device: str | tuple[str, int]
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.device = device
        self.__validate()

    def __validate(self) -> None:
        validate_not_empty("device", self.device)


class SelectCLIPDevice(BaseNode):
    __slots__ = ("clip", "device")
    CATEGORY = "advanced/multigpu"
    OUTPUTS = {"CLIP": 0}
    RETURN_TYPES = ("CLIP",)
    INPUT_SOCKET_TYPES = {"clip": "CLIP", "device": "COMBO"}

    def __init__(
        self, node_id: str, clip: tuple[str, int], device: str | tuple[str, int]
    ) -> None:
        super().__init__(node_id=node_id)
        self.clip = clip
        self.device = device
        self.__validate()

    def __validate(self) -> None:
        validate_not_empty("device", self.device)


class SelectVAEDevice(BaseNode):
    __slots__ = ("vae", "device")
    CATEGORY = "advanced/multigpu"
    OUTPUTS = {"VAE": 0}
    RETURN_TYPES = ("VAE",)
    INPUT_SOCKET_TYPES = {"vae": "VAE", "device": "COMBO"}

    def __init__(
        self, node_id: str, vae: tuple[str, int], device: str | tuple[str, int]
    ) -> None:
        super().__init__(node_id=node_id)
        self.vae = vae
        self.device = device
        self.__validate()

    def __validate(self) -> None:
        validate_not_empty("device", self.device)
