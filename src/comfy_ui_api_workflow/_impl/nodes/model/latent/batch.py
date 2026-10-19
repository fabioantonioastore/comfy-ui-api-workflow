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


class BatchLatentsNode(BaseNode):
    __slots__ = (
        "latents_latent0",
        "latents_latent1",
        "latents_latent2",
        "latents_latent3",
        "latents_latent4",
        "latents_latent5",
        "latents_latent6",
        "latents_latent7",
        "latents_latent8",
        "latents_latent9",
        "latents_latent10",
        "latents_latent11",
        "latents_latent12",
        "latents_latent13",
        "latents_latent14",
        "latents_latent15",
        "latents_latent16",
        "latents_latent17",
        "latents_latent18",
        "latents_latent19",
        "latents_latent20",
        "latents_latent21",
        "latents_latent22",
        "latents_latent23",
        "latents_latent24",
        "latents_latent25",
        "latents_latent26",
        "latents_latent27",
        "latents_latent28",
        "latents_latent29",
        "latents_latent30",
        "latents_latent31",
        "latents_latent32",
        "latents_latent33",
        "latents_latent34",
        "latents_latent35",
        "latents_latent36",
        "latents_latent37",
        "latents_latent38",
        "latents_latent39",
        "latents_latent40",
        "latents_latent41",
        "latents_latent42",
        "latents_latent43",
        "latents_latent44",
        "latents_latent45",
        "latents_latent46",
        "latents_latent47",
        "latents_latent48",
        "latents_latent49",
    )
    CATEGORY = "model/latent/batch"
    OUTPUTS = {"LATENT": 0}
    RETURN_TYPES = ("LATENT",)
    INPUT_NAMES = {
        "latents_latent0": "latents.latent0",
        "latents_latent1": "latents.latent1",
        "latents_latent2": "latents.latent2",
        "latents_latent3": "latents.latent3",
        "latents_latent4": "latents.latent4",
        "latents_latent5": "latents.latent5",
        "latents_latent6": "latents.latent6",
        "latents_latent7": "latents.latent7",
        "latents_latent8": "latents.latent8",
        "latents_latent9": "latents.latent9",
        "latents_latent10": "latents.latent10",
        "latents_latent11": "latents.latent11",
        "latents_latent12": "latents.latent12",
        "latents_latent13": "latents.latent13",
        "latents_latent14": "latents.latent14",
        "latents_latent15": "latents.latent15",
        "latents_latent16": "latents.latent16",
        "latents_latent17": "latents.latent17",
        "latents_latent18": "latents.latent18",
        "latents_latent19": "latents.latent19",
        "latents_latent20": "latents.latent20",
        "latents_latent21": "latents.latent21",
        "latents_latent22": "latents.latent22",
        "latents_latent23": "latents.latent23",
        "latents_latent24": "latents.latent24",
        "latents_latent25": "latents.latent25",
        "latents_latent26": "latents.latent26",
        "latents_latent27": "latents.latent27",
        "latents_latent28": "latents.latent28",
        "latents_latent29": "latents.latent29",
        "latents_latent30": "latents.latent30",
        "latents_latent31": "latents.latent31",
        "latents_latent32": "latents.latent32",
        "latents_latent33": "latents.latent33",
        "latents_latent34": "latents.latent34",
        "latents_latent35": "latents.latent35",
        "latents_latent36": "latents.latent36",
        "latents_latent37": "latents.latent37",
        "latents_latent38": "latents.latent38",
        "latents_latent39": "latents.latent39",
        "latents_latent40": "latents.latent40",
        "latents_latent41": "latents.latent41",
        "latents_latent42": "latents.latent42",
        "latents_latent43": "latents.latent43",
        "latents_latent44": "latents.latent44",
        "latents_latent45": "latents.latent45",
        "latents_latent46": "latents.latent46",
        "latents_latent47": "latents.latent47",
        "latents_latent48": "latents.latent48",
        "latents_latent49": "latents.latent49",
    }
    INPUT_SOCKET_TYPES = {
        "latents_latent0": "LATENT",
        "latents_latent1": "LATENT",
        "latents_latent2": "LATENT",
        "latents_latent3": "LATENT",
        "latents_latent4": "LATENT",
        "latents_latent5": "LATENT",
        "latents_latent6": "LATENT",
        "latents_latent7": "LATENT",
        "latents_latent8": "LATENT",
        "latents_latent9": "LATENT",
        "latents_latent10": "LATENT",
        "latents_latent11": "LATENT",
        "latents_latent12": "LATENT",
        "latents_latent13": "LATENT",
        "latents_latent14": "LATENT",
        "latents_latent15": "LATENT",
        "latents_latent16": "LATENT",
        "latents_latent17": "LATENT",
        "latents_latent18": "LATENT",
        "latents_latent19": "LATENT",
        "latents_latent20": "LATENT",
        "latents_latent21": "LATENT",
        "latents_latent22": "LATENT",
        "latents_latent23": "LATENT",
        "latents_latent24": "LATENT",
        "latents_latent25": "LATENT",
        "latents_latent26": "LATENT",
        "latents_latent27": "LATENT",
        "latents_latent28": "LATENT",
        "latents_latent29": "LATENT",
        "latents_latent30": "LATENT",
        "latents_latent31": "LATENT",
        "latents_latent32": "LATENT",
        "latents_latent33": "LATENT",
        "latents_latent34": "LATENT",
        "latents_latent35": "LATENT",
        "latents_latent36": "LATENT",
        "latents_latent37": "LATENT",
        "latents_latent38": "LATENT",
        "latents_latent39": "LATENT",
        "latents_latent40": "LATENT",
        "latents_latent41": "LATENT",
        "latents_latent42": "LATENT",
        "latents_latent43": "LATENT",
        "latents_latent44": "LATENT",
        "latents_latent45": "LATENT",
        "latents_latent46": "LATENT",
        "latents_latent47": "LATENT",
        "latents_latent48": "LATENT",
        "latents_latent49": "LATENT",
    }

    def __init__(
        self,
        node_id: str,
        latents_latent0: tuple[str, int],
        latents_latent1: tuple[str, int] | None = None,
        latents_latent2: tuple[str, int] | None = None,
        latents_latent3: tuple[str, int] | None = None,
        latents_latent4: tuple[str, int] | None = None,
        latents_latent5: tuple[str, int] | None = None,
        latents_latent6: tuple[str, int] | None = None,
        latents_latent7: tuple[str, int] | None = None,
        latents_latent8: tuple[str, int] | None = None,
        latents_latent9: tuple[str, int] | None = None,
        latents_latent10: tuple[str, int] | None = None,
        latents_latent11: tuple[str, int] | None = None,
        latents_latent12: tuple[str, int] | None = None,
        latents_latent13: tuple[str, int] | None = None,
        latents_latent14: tuple[str, int] | None = None,
        latents_latent15: tuple[str, int] | None = None,
        latents_latent16: tuple[str, int] | None = None,
        latents_latent17: tuple[str, int] | None = None,
        latents_latent18: tuple[str, int] | None = None,
        latents_latent19: tuple[str, int] | None = None,
        latents_latent20: tuple[str, int] | None = None,
        latents_latent21: tuple[str, int] | None = None,
        latents_latent22: tuple[str, int] | None = None,
        latents_latent23: tuple[str, int] | None = None,
        latents_latent24: tuple[str, int] | None = None,
        latents_latent25: tuple[str, int] | None = None,
        latents_latent26: tuple[str, int] | None = None,
        latents_latent27: tuple[str, int] | None = None,
        latents_latent28: tuple[str, int] | None = None,
        latents_latent29: tuple[str, int] | None = None,
        latents_latent30: tuple[str, int] | None = None,
        latents_latent31: tuple[str, int] | None = None,
        latents_latent32: tuple[str, int] | None = None,
        latents_latent33: tuple[str, int] | None = None,
        latents_latent34: tuple[str, int] | None = None,
        latents_latent35: tuple[str, int] | None = None,
        latents_latent36: tuple[str, int] | None = None,
        latents_latent37: tuple[str, int] | None = None,
        latents_latent38: tuple[str, int] | None = None,
        latents_latent39: tuple[str, int] | None = None,
        latents_latent40: tuple[str, int] | None = None,
        latents_latent41: tuple[str, int] | None = None,
        latents_latent42: tuple[str, int] | None = None,
        latents_latent43: tuple[str, int] | None = None,
        latents_latent44: tuple[str, int] | None = None,
        latents_latent45: tuple[str, int] | None = None,
        latents_latent46: tuple[str, int] | None = None,
        latents_latent47: tuple[str, int] | None = None,
        latents_latent48: tuple[str, int] | None = None,
        latents_latent49: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.latents_latent0 = latents_latent0
        if latents_latent1 is not None:
            self.latents_latent1 = latents_latent1
        if latents_latent2 is not None:
            self.latents_latent2 = latents_latent2
        if latents_latent3 is not None:
            self.latents_latent3 = latents_latent3
        if latents_latent4 is not None:
            self.latents_latent4 = latents_latent4
        if latents_latent5 is not None:
            self.latents_latent5 = latents_latent5
        if latents_latent6 is not None:
            self.latents_latent6 = latents_latent6
        if latents_latent7 is not None:
            self.latents_latent7 = latents_latent7
        if latents_latent8 is not None:
            self.latents_latent8 = latents_latent8
        if latents_latent9 is not None:
            self.latents_latent9 = latents_latent9
        if latents_latent10 is not None:
            self.latents_latent10 = latents_latent10
        if latents_latent11 is not None:
            self.latents_latent11 = latents_latent11
        if latents_latent12 is not None:
            self.latents_latent12 = latents_latent12
        if latents_latent13 is not None:
            self.latents_latent13 = latents_latent13
        if latents_latent14 is not None:
            self.latents_latent14 = latents_latent14
        if latents_latent15 is not None:
            self.latents_latent15 = latents_latent15
        if latents_latent16 is not None:
            self.latents_latent16 = latents_latent16
        if latents_latent17 is not None:
            self.latents_latent17 = latents_latent17
        if latents_latent18 is not None:
            self.latents_latent18 = latents_latent18
        if latents_latent19 is not None:
            self.latents_latent19 = latents_latent19
        if latents_latent20 is not None:
            self.latents_latent20 = latents_latent20
        if latents_latent21 is not None:
            self.latents_latent21 = latents_latent21
        if latents_latent22 is not None:
            self.latents_latent22 = latents_latent22
        if latents_latent23 is not None:
            self.latents_latent23 = latents_latent23
        if latents_latent24 is not None:
            self.latents_latent24 = latents_latent24
        if latents_latent25 is not None:
            self.latents_latent25 = latents_latent25
        if latents_latent26 is not None:
            self.latents_latent26 = latents_latent26
        if latents_latent27 is not None:
            self.latents_latent27 = latents_latent27
        if latents_latent28 is not None:
            self.latents_latent28 = latents_latent28
        if latents_latent29 is not None:
            self.latents_latent29 = latents_latent29
        if latents_latent30 is not None:
            self.latents_latent30 = latents_latent30
        if latents_latent31 is not None:
            self.latents_latent31 = latents_latent31
        if latents_latent32 is not None:
            self.latents_latent32 = latents_latent32
        if latents_latent33 is not None:
            self.latents_latent33 = latents_latent33
        if latents_latent34 is not None:
            self.latents_latent34 = latents_latent34
        if latents_latent35 is not None:
            self.latents_latent35 = latents_latent35
        if latents_latent36 is not None:
            self.latents_latent36 = latents_latent36
        if latents_latent37 is not None:
            self.latents_latent37 = latents_latent37
        if latents_latent38 is not None:
            self.latents_latent38 = latents_latent38
        if latents_latent39 is not None:
            self.latents_latent39 = latents_latent39
        if latents_latent40 is not None:
            self.latents_latent40 = latents_latent40
        if latents_latent41 is not None:
            self.latents_latent41 = latents_latent41
        if latents_latent42 is not None:
            self.latents_latent42 = latents_latent42
        if latents_latent43 is not None:
            self.latents_latent43 = latents_latent43
        if latents_latent44 is not None:
            self.latents_latent44 = latents_latent44
        if latents_latent45 is not None:
            self.latents_latent45 = latents_latent45
        if latents_latent46 is not None:
            self.latents_latent46 = latents_latent46
        if latents_latent47 is not None:
            self.latents_latent47 = latents_latent47
        if latents_latent48 is not None:
            self.latents_latent48 = latents_latent48
        if latents_latent49 is not None:
            self.latents_latent49 = latents_latent49
