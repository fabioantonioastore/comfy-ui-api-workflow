from typing import Literal, get_args

from ...base import BaseNode
from ...validation import validate_choice, validate_range


class CFGOverride(BaseNode):
    __slots__ = ("model", "cfg", "start_percent", "end_percent")
    CATEGORY = "model/sampling/guiders"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "cfg": "FLOAT",
        "start_percent": "FLOAT",
        "end_percent": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        cfg: float | tuple[str, int] = 1.0,
        start_percent: float | tuple[str, int] = 0.0,
        end_percent: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.cfg = cfg
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.__validate()

    def __validate(self) -> None:
        validate_range("cfg", self.cfg, 0.0, 100.0)
        validate_range("start_percent", self.start_percent, 0.0, 1.0)
        validate_range("end_percent", self.end_percent, 0.0, 1.0)


class CFGGuider(BaseNode):
    __slots__ = ("model", "positive", "negative", "cfg")
    CATEGORY = "model/sampling/guiders"
    OUTPUTS = {"GUIDER": 0}
    RETURN_TYPES = ("GUIDER",)
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "positive": "CONDITIONING",
        "negative": "CONDITIONING",
        "cfg": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        positive: tuple[str, int],
        negative: tuple[str, int],
        cfg: float | tuple[str, int] = 8.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.positive = positive
        self.negative = negative
        self.cfg = cfg
        self.__validate()

    def __validate(self) -> None:
        validate_range("cfg", self.cfg, 0.0, 100.0)


DualCFGGuider_Styles = Literal["regular", "nested"]
DUALCFGGUIDER_STYLES = frozenset(get_args(DualCFGGuider_Styles))


class DualCFGGuider(BaseNode):
    __slots__ = (
        "model",
        "cond1",
        "cond2",
        "negative",
        "style",
        "cfg_conds",
        "cfg_cond2_negative",
    )
    CATEGORY = "model/sampling/guiders"
    OUTPUTS = {"GUIDER": 0}
    RETURN_TYPES = ("GUIDER",)
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "cond1": "CONDITIONING",
        "cond2": "CONDITIONING",
        "negative": "CONDITIONING",
        "cfg_conds": "FLOAT",
        "cfg_cond2_negative": "FLOAT",
        "style": "COMBO",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        cond1: tuple[str, int],
        cond2: tuple[str, int],
        negative: tuple[str, int],
        style: DualCFGGuider_Styles | tuple[str, int],
        cfg_conds: float | tuple[str, int] = 8.0,
        cfg_cond2_negative: float | tuple[str, int] = 8.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.cond1 = cond1
        self.cond2 = cond2
        self.negative = negative
        self.style = style
        self.cfg_conds = cfg_conds
        self.cfg_cond2_negative = cfg_cond2_negative
        self.__validate()

    def __validate(self) -> None:
        validate_choice("style", self.style, DUALCFGGUIDER_STYLES)
        validate_range("cfg_conds", self.cfg_conds, 0.0, 100.0)
        validate_range("cfg_cond2_negative", self.cfg_cond2_negative, 0.0, 100.0)


class DualModelGuider(BaseNode):
    __slots__ = ("model", "positive", "cfg", "model_negative", "negative")
    CATEGORY = "model/sampling/guiders"
    OUTPUTS = {"GUIDER": 0}
    RETURN_TYPES = ("GUIDER",)
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "positive": "CONDITIONING",
        "cfg": "FLOAT",
        "model_negative": "MODEL",
        "negative": "CONDITIONING",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        positive: tuple[str, int],
        cfg: float | tuple[str, int] = 4.0,
        model_negative: tuple[str, int] | None = None,
        negative: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.positive = positive
        self.cfg = cfg
        if model_negative is not None:
            self.model_negative = model_negative
        if negative is not None:
            self.negative = negative
        self.__validate()

    def __validate(self) -> None:
        validate_range("cfg", self.cfg, 0.0, 100.0)


class BasicGuider(BaseNode):
    __slots__ = ("model", "conditioning")
    CATEGORY = "model/sampling/guiders"
    OUTPUTS = {"GUIDER": 0}
    RETURN_TYPES = ("GUIDER",)
    INPUT_SOCKET_TYPES = {"model": "MODEL", "conditioning": "CONDITIONING"}

    def __init__(
        self, node_id: str, model: tuple[str, int], conditioning: tuple[str, int]
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.conditioning = conditioning


class VideoLinearCFGGuidance(BaseNode):
    __slots__ = ("model", "min_cfg")
    CATEGORY = "model/sampling/guiders"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {"model": "MODEL", "min_cfg": "FLOAT"}

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        min_cfg: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.min_cfg = min_cfg
        self.__validate()

    def __validate(self) -> None:
        validate_range("min_cfg", self.min_cfg, 0.0, 100.0)


class VideoTriangleCFGGuidance(BaseNode):
    __slots__ = ("model", "min_cfg")
    CATEGORY = "model/sampling/guiders"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {"model": "MODEL", "min_cfg": "FLOAT"}

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        min_cfg: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.min_cfg = min_cfg
        self.__validate()

    def __validate(self) -> None:
        validate_range("min_cfg", self.min_cfg, 0.0, 100.0)


class LTXVDualCFGGuider(BaseNode):
    __slots__ = ("model", "positive", "negative", "video_cfg", "audio_cfg")
    CATEGORY = "model/sampling/guiders"
    OUTPUTS = {"GUIDER": 0}
    RETURN_TYPES = ("GUIDER",)
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "positive": "CONDITIONING",
        "negative": "CONDITIONING",
        "video_cfg": "FLOAT",
        "audio_cfg": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        positive: tuple[str, int],
        negative: tuple[str, int],
        video_cfg: float | tuple[str, int] = 3.0,
        audio_cfg: float | tuple[str, int] = 7.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.positive = positive
        self.negative = negative
        self.video_cfg = video_cfg
        self.audio_cfg = audio_cfg
        self.__validate()

    def __validate(self) -> None:
        validate_range("video_cfg", self.video_cfg, 0.0, 100.0)
        validate_range("audio_cfg", self.audio_cfg, 0.0, 100.0)
