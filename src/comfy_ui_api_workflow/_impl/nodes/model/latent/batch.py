from ...base import BaseNode
from ...validation import validate_range


class LatentFromBatch(BaseNode):
    __slots__ = ("samples", "batch_index", "length")
    CATEGORY = "model/latent/batch"
    OUTPUTS = {"LATENT": 0}
    RETURN_TYPES = ("LATENT",)
    INPUT_SOCKET_TYPES = {"samples": "LATENT", "batch_index": "INT", "length": "INT"}

    def __init__(
        self,
        node_id: str,
        samples: tuple[str, int],
        batch_index: int | tuple[str, int] = 0,
        length: int | tuple[str, int] = 1,
    ) -> None:
        super().__init__(node_id=node_id)
        self.samples = samples
        self.batch_index = batch_index
        self.length = length
        self.__validate()

    def __validate(self) -> None:
        validate_range("batch_index", self.batch_index, -16_384, 16_384)
        validate_range("length", self.length, 1, 64)


class RepeatLatentBatch(BaseNode):
    __slots__ = ("samples", "amount")
    CATEGORY = "model/latent/batch"
    OUTPUTS = {"LATENT": 0}
    RETURN_TYPES = ("LATENT",)
    INPUT_SOCKET_TYPES = {"samples": "LATENT", "amount": "INT"}

    def __init__(
        self, node_id: str, samples: tuple[str, int], amount: int | tuple[str, int] = 1
    ) -> None:
        super().__init__(node_id=node_id)
        self.samples = samples
        self.amount = amount
        self.__validate()

    def __validate(self) -> None:
        validate_range("amount", self.amount, 1, 64)


class ReplaceVideoLatentFrames(BaseNode):
    __slots__ = ("destination", "index", "source")
    CATEGORY = "model/latent/batch"
    OUTPUTS = {"LATENT": 0}
    RETURN_TYPES = ("LATENT",)
    INPUT_SOCKET_TYPES = {"destination": "LATENT", "index": "INT", "source": "LATENT"}

    def __init__(
        self,
        node_id: str,
        destination: tuple[str, int],
        index: int | tuple[str, int] = 0,
        source: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.destination = destination
        self.index = index
        if source is not None:
            self.source = source
        self.__validate()

    def __validate(self) -> None:
        validate_range("index", self.index, -16_384, 16_384)


class RebatchLatents(BaseNode):
    __slots__ = ("latents", "batch_size")
    CATEGORY = "model/latent/batch"
    OUTPUTS = {"LATENT": 0}
    RETURN_TYPES = ("LATENT",)
    INPUT_SOCKET_TYPES = {"latents": "LATENT", "batch_size": "INT"}

    def __init__(
        self,
        node_id: str,
        latents: tuple[str, int],
        batch_size: int | tuple[str, int] = 1,
    ) -> None:
        super().__init__(node_id=node_id)
        self.latents = latents
        self.batch_size = batch_size
        self.__validate()

    def __validate(self) -> None:
        validate_range("batch_size", self.batch_size, 1, 4096)
