from ..base import BaseNode
from ..validation import validate_bool, validate_range


class SkipLayerGuidanceSD3(BaseNode):
    __slots__ = ("model", "layers", "scale", "start_percent", "end_percent")
    CATEGORY = "advanced/guidance"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "layers": "STRING",
        "scale": "FLOAT",
        "start_percent": "FLOAT",
        "end_percent": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        layers: str | tuple[str, int] = "7, 8, 9",
        scale: float | tuple[str, int] = 3.0,
        start_percent: float | tuple[str, int] = 0.01,
        end_percent: float | tuple[str, int] = 0.15,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.layers = layers
        self.scale = scale
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.__validate()

    def __validate(self) -> None:
        validate_range("scale", self.scale, 0.0, 10.0)
        validate_range("start_percent", self.start_percent, 0.0, 1.0)
        validate_range("end_percent", self.end_percent, 0.0, 1.0)


class SkipLayerGuidanceDiT(BaseNode):
    __slots__ = (
        "model",
        "double_layers",
        "single_layers",
        "scale",
        "start_percent",
        "end_percent",
        "rescaling_scale",
    )
    CATEGORY = "advanced/guidance"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "double_layers": "STRING",
        "single_layers": "STRING",
        "scale": "FLOAT",
        "start_percent": "FLOAT",
        "end_percent": "FLOAT",
        "rescaling_scale": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        double_layers: str | tuple[str, int] = "7, 8, 9",
        single_layers: str | tuple[str, int] = "7, 8, 9",
        scale: float | tuple[str, int] = 3.0,
        start_percent: float | tuple[str, int] = 0.01,
        end_percent: float | tuple[str, int] = 0.15,
        rescaling_scale: float | tuple[str, int] = 0.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.double_layers = double_layers
        self.single_layers = single_layers
        self.scale = scale
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.rescaling_scale = rescaling_scale
        self.__validate()

    def __validate(self) -> None:
        validate_range("scale", self.scale, 0.0, 10.0)
        validate_range("start_percent", self.start_percent, 0.0, 1.0)
        validate_range("end_percent", self.end_percent, 0.0, 1.0)
        validate_range("rescaling_scale", self.rescaling_scale, 0.0, 10.0)


class SkipLayerGuidanceDiTSimple(BaseNode):
    __slots__ = (
        "model",
        "double_layers",
        "single_layers",
        "start_percent",
        "end_percent",
    )
    CATEGORY = "advanced/guidance"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "double_layers": "STRING",
        "single_layers": "STRING",
        "start_percent": "FLOAT",
        "end_percent": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        double_layers: str | tuple[str, int] = "7, 8, 9",
        single_layers: str | tuple[str, int] = "7, 8, 9",
        start_percent: float | tuple[str, int] = 0.0,
        end_percent: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.double_layers = double_layers
        self.single_layers = single_layers
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.__validate()

    def __validate(self) -> None:
        validate_range("start_percent", self.start_percent, 0.0, 1.0)
        validate_range("end_percent", self.end_percent, 0.0, 1.0)


class LTXVModalityGuidance(BaseNode):
    __slots__ = ("model", "modality_scale", "start_percent", "end_percent")
    CATEGORY = "advanced/guidance"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "modality_scale": "FLOAT",
        "start_percent": "FLOAT",
        "end_percent": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        modality_scale: float | tuple[str, int] = 3.0,
        start_percent: float | tuple[str, int] = 0.0,
        end_percent: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.modality_scale = modality_scale
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.__validate()

    def __validate(self) -> None:
        validate_range("modality_scale", self.modality_scale, 1.0, 100.0)
        validate_range("start_percent", self.start_percent, 0.0, 1.0)
        validate_range("end_percent", self.end_percent, 0.0, 1.0)


class LTXVSpatioTemporalGuidance(BaseNode):
    __slots__ = ("model", "scale", "blocks", "start_percent", "end_percent")
    CATEGORY = "advanced/guidance"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "scale": "FLOAT",
        "blocks": "STRING",
        "start_percent": "FLOAT",
        "end_percent": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        scale: float | tuple[str, int] = 1.0,
        blocks: str | tuple[str, int] = "29",
        start_percent: float | tuple[str, int] = 0.0,
        end_percent: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.scale = scale
        self.blocks = blocks
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.__validate()

    def __validate(self) -> None:
        validate_range("scale", self.scale, 0.0, 100.0)
        validate_range("start_percent", self.start_percent, 0.0, 1.0)
        validate_range("end_percent", self.end_percent, 0.0, 1.0)


class CFGZeroStar(BaseNode):
    __slots__ = ("model",)
    CATEGORY = "advanced/guidance"
    OUTPUTS = {"patched_model": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {"model": "MODEL"}

    def __init__(self, node_id: str, model: tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.model = model


class CFGNorm(BaseNode):
    __slots__ = ("model", "strength", "pre_cfg")
    CATEGORY = "advanced/guidance"
    OUTPUTS = {"patched_model": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {"model": "MODEL", "strength": "FLOAT", "pre_cfg": "BOOLEAN"}

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        strength: float | tuple[str, int] = 1.0,
        pre_cfg: bool | tuple[str, int] = False,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.strength = strength
        self.pre_cfg = pre_cfg
        self.__validate()

    def __validate(self) -> None:
        validate_range("strength", self.strength, 0.0, 100.0)
        validate_bool("pre_cfg", self.pre_cfg)


class TCFG(BaseNode):
    __slots__ = ("model",)
    CATEGORY = "advanced/guidance"
    OUTPUTS = {"patched_model": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {"model": "MODEL"}

    def __init__(self, node_id: str, model: tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.model = model


class NAGuidance(BaseNode):
    __slots__ = ("model", "nag_scale", "nag_alpha", "nag_tau")
    CATEGORY = "advanced/guidance"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "nag_scale": "FLOAT",
        "nag_alpha": "FLOAT",
        "nag_tau": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        nag_scale: float | tuple[str, int] = 5.0,
        nag_alpha: float | tuple[str, int] = 0.5,
        nag_tau: float | tuple[str, int] = 1.5,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.nag_scale = nag_scale
        self.nag_alpha = nag_alpha
        self.nag_tau = nag_tau
        self.__validate()

    def __validate(self) -> None:
        validate_range("nag_scale", self.nag_scale, 0.0, 50.0)
        validate_range("nag_alpha", self.nag_alpha, 0.0, 1.0)
        validate_range("nag_tau", self.nag_tau, 1.0, 10.0)
