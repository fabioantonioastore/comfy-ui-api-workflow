from ..base import BaseNode
from ..validation import validate_range


class UNetSelfAttentionMultiply(BaseNode):
    __slots__ = ("model", "q", "k", "v", "out")
    CATEGORY = "experimental/attention_experiments"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "q": "FLOAT",
        "k": "FLOAT",
        "v": "FLOAT",
        "out": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        q: float | tuple[str, int] = 1.0,
        k: float | tuple[str, int] = 1.0,
        v: float | tuple[str, int] = 1.0,
        out: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.q = q
        self.k = k
        self.v = v
        self.out = out
        self.__validate()

    def __validate(self) -> None:
        validate_range("q", self.q, 0.0, 10.0)
        validate_range("k", self.k, 0.0, 10.0)
        validate_range("v", self.v, 0.0, 10.0)
        validate_range("out", self.out, 0.0, 10.0)


class UNetCrossAttentionMultiply(BaseNode):
    __slots__ = ("model", "q", "k", "v", "out")
    CATEGORY = "experimental/attention_experiments"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "q": "FLOAT",
        "k": "FLOAT",
        "v": "FLOAT",
        "out": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        q: float | tuple[str, int] = 1.0,
        k: float | tuple[str, int] = 1.0,
        v: float | tuple[str, int] = 1.0,
        out: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.q = q
        self.k = k
        self.v = v
        self.out = out
        self.__validate()

    def __validate(self) -> None:
        validate_range("q", self.q, 0.0, 10.0)
        validate_range("k", self.k, 0.0, 10.0)
        validate_range("v", self.v, 0.0, 10.0)
        validate_range("out", self.out, 0.0, 10.0)


class CLIPAttentionMultiply(BaseNode):
    __slots__ = ("clip", "q", "k", "v", "out")
    CATEGORY = "experimental/attention_experiments"
    OUTPUTS = {"CLIP": 0}
    RETURN_TYPES = ("CLIP",)
    INPUT_SOCKET_TYPES = {
        "clip": "CLIP",
        "q": "FLOAT",
        "k": "FLOAT",
        "v": "FLOAT",
        "out": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        clip: tuple[str, int],
        q: float | tuple[str, int] = 1.0,
        k: float | tuple[str, int] = 1.0,
        v: float | tuple[str, int] = 1.0,
        out: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.clip = clip
        self.q = q
        self.k = k
        self.v = v
        self.out = out
        self.__validate()

    def __validate(self) -> None:
        validate_range("q", self.q, 0.0, 10.0)
        validate_range("k", self.k, 0.0, 10.0)
        validate_range("v", self.v, 0.0, 10.0)
        validate_range("out", self.out, 0.0, 10.0)


class UNetTemporalAttentionMultiply(BaseNode):
    __slots__ = (
        "model",
        "self_structural",
        "self_temporal",
        "cross_structural",
        "cross_temporal",
    )
    CATEGORY = "experimental/attention_experiments"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "self_structural": "FLOAT",
        "self_temporal": "FLOAT",
        "cross_structural": "FLOAT",
        "cross_temporal": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        self_structural: float | tuple[str, int] = 1.0,
        self_temporal: float | tuple[str, int] = 1.0,
        cross_structural: float | tuple[str, int] = 1.0,
        cross_temporal: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.self_structural = self_structural
        self.self_temporal = self_temporal
        self.cross_structural = cross_structural
        self.cross_temporal = cross_temporal
        self.__validate()

    def __validate(self) -> None:
        validate_range("self_structural", self.self_structural, 0.0, 10.0)
        validate_range("self_temporal", self.self_temporal, 0.0, 10.0)
        validate_range("cross_structural", self.cross_structural, 0.0, 10.0)
        validate_range("cross_temporal", self.cross_temporal, 0.0, 10.0)
