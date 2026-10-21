from ...base import BaseNode
from ...validation import validate_range


class ModelMergeSimple(BaseNode):
    __slots__ = ("model1", "model2", "ratio")
    CATEGORY = "model/merging"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {"model1": "MODEL", "model2": "MODEL", "ratio": "FLOAT"}

    def __init__(
        self,
        node_id: str,
        model1: tuple[str, int],
        model2: tuple[str, int],
        ratio: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model1 = model1
        self.model2 = model2
        self.ratio = ratio
        self.__validate()

    def __validate(self) -> None:
        validate_range("ratio", self.ratio, 0.0, 1.0)


class ModelMergeBlocks(BaseNode):
    __slots__ = ("model1", "model2", "input", "middle", "out")
    CATEGORY = "model/merging"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {
        "model1": "MODEL",
        "model2": "MODEL",
        "input": "FLOAT",
        "middle": "FLOAT",
        "out": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        model1: tuple[str, int],
        model2: tuple[str, int],
        input: float | tuple[str, int] = 1.0,
        middle: float | tuple[str, int] = 1.0,
        out: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model1 = model1
        self.model2 = model2
        self.input = input
        self.middle = middle
        self.out = out
        self.__validate()

    def __validate(self) -> None:
        validate_range("input", self.input, 0.0, 1.0)
        validate_range("middle", self.middle, 0.0, 1.0)
        validate_range("out", self.out, 0.0, 1.0)


class ModelMergeSubtract(BaseNode):
    __slots__ = ("model1", "model2", "multiplier")
    CATEGORY = "model/merging"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {"model1": "MODEL", "model2": "MODEL", "multiplier": "FLOAT"}

    def __init__(
        self,
        node_id: str,
        model1: tuple[str, int],
        model2: tuple[str, int],
        multiplier: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model1 = model1
        self.model2 = model2
        self.multiplier = multiplier
        self.__validate()

    def __validate(self) -> None:
        validate_range("multiplier", self.multiplier, -10.0, 10.0)


class ModelMergeAdd(BaseNode):
    __slots__ = ("model1", "model2")
    CATEGORY = "model/merging"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {"model1": "MODEL", "model2": "MODEL"}

    def __init__(
        self, node_id: str, model1: tuple[str, int], model2: tuple[str, int]
    ) -> None:
        super().__init__(node_id=node_id)
        self.model1 = model1
        self.model2 = model2


class CheckpointSave(BaseNode):
    __slots__ = ("model", "clip", "vae", "filename_prefix")
    CATEGORY = "model/merging"
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "clip": "CLIP",
        "vae": "VAE",
        "filename_prefix": "STRING",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        clip: tuple[str, int],
        vae: tuple[str, int],
        filename_prefix: str | tuple[str, int] = "checkpoints/ComfyUI",
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.clip = clip
        self.vae = vae
        self.filename_prefix = filename_prefix


class CLIPMergeSimple(BaseNode):
    __slots__ = ("clip1", "clip2", "ratio")
    CATEGORY = "model/merging"
    OUTPUTS = {"CLIP": 0}
    RETURN_TYPES = ("CLIP",)
    INPUT_SOCKET_TYPES = {"clip1": "CLIP", "clip2": "CLIP", "ratio": "FLOAT"}

    def __init__(
        self,
        node_id: str,
        clip1: tuple[str, int],
        clip2: tuple[str, int],
        ratio: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.clip1 = clip1
        self.clip2 = clip2
        self.ratio = ratio
        self.__validate()

    def __validate(self) -> None:
        validate_range("ratio", self.ratio, 0.0, 1.0)


class CLIPMergeSubtract(BaseNode):
    __slots__ = ("clip1", "clip2", "multiplier")
    CATEGORY = "model/merging"
    OUTPUTS = {"CLIP": 0}
    RETURN_TYPES = ("CLIP",)
    INPUT_SOCKET_TYPES = {"clip1": "CLIP", "clip2": "CLIP", "multiplier": "FLOAT"}

    def __init__(
        self,
        node_id: str,
        clip1: tuple[str, int],
        clip2: tuple[str, int],
        multiplier: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.clip1 = clip1
        self.clip2 = clip2
        self.multiplier = multiplier
        self.__validate()

    def __validate(self) -> None:
        validate_range("multiplier", self.multiplier, -10.0, 10.0)


class CLIPMergeAdd(BaseNode):
    __slots__ = ("clip1", "clip2")
    CATEGORY = "model/merging"
    OUTPUTS = {"CLIP": 0}
    RETURN_TYPES = ("CLIP",)
    INPUT_SOCKET_TYPES = {"clip1": "CLIP", "clip2": "CLIP"}

    def __init__(
        self, node_id: str, clip1: tuple[str, int], clip2: tuple[str, int]
    ) -> None:
        super().__init__(node_id=node_id)
        self.clip1 = clip1
        self.clip2 = clip2


class CLIPSave(BaseNode):
    __slots__ = ("clip", "filename_prefix")
    CATEGORY = "model/merging"
    INPUT_SOCKET_TYPES = {"clip": "CLIP", "filename_prefix": "STRING"}

    def __init__(
        self,
        node_id: str,
        clip: tuple[str, int],
        filename_prefix: str | tuple[str, int] = "clip/ComfyUI",
    ) -> None:
        super().__init__(node_id=node_id)
        self.clip = clip
        self.filename_prefix = filename_prefix


class VAESave(BaseNode):
    __slots__ = ("vae", "filename_prefix")
    CATEGORY = "model/merging"
    INPUT_SOCKET_TYPES = {"vae": "VAE", "filename_prefix": "STRING"}

    def __init__(
        self,
        node_id: str,
        vae: tuple[str, int],
        filename_prefix: str | tuple[str, int] = "vae/ComfyUI_vae",
    ) -> None:
        super().__init__(node_id=node_id)
        self.vae = vae
        self.filename_prefix = filename_prefix


class ModelSave(BaseNode):
    __slots__ = ("model", "filename_prefix")
    CATEGORY = "model/merging"
    INPUT_SOCKET_TYPES = {"model": "MODEL", "filename_prefix": "STRING"}

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        filename_prefix: str | tuple[str, int] = "diffusion_models/ComfyUI",
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.filename_prefix = filename_prefix


class ImageOnlyCheckpointSave(BaseNode):
    __slots__ = ("model", "clip_vision", "vae", "filename_prefix")
    CATEGORY = "model/merging"
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "clip_vision": "CLIP_VISION",
        "vae": "VAE",
        "filename_prefix": "STRING",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        clip_vision: tuple[str, int],
        vae: tuple[str, int],
        filename_prefix: str | tuple[str, int] = "checkpoints/ComfyUI",
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.clip_vision = clip_vision
        self.vae = vae
        self.filename_prefix = filename_prefix


class SaveLoRA(BaseNode):
    __slots__ = ("lora", "prefix", "steps")
    CATEGORY = "model/merging"
    INPUT_SOCKET_TYPES = {"lora": "LORA_MODEL", "prefix": "STRING", "steps": "INT"}

    def __init__(
        self,
        node_id: str,
        lora: tuple[str, int],
        prefix: str | tuple[str, int] = "loras/ComfyUI_trained_lora",
        steps: int | tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.lora = lora
        self.prefix = prefix
        if steps is not None:
            self.steps = steps
        self.__validate()

    def __validate(self) -> None:
        if hasattr(self, "steps"):
            validate_range("steps", self.steps)


from . import model_specific
