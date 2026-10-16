from ...base import BaseNode
from ...validation import validate_range


class ModelMergeSD1(BaseNode):
    __slots__ = (
        "model1",
        "model2",
        "time_embed",
        "label_emb",
        "input_blocks_0",
        "input_blocks_1",
        "input_blocks_2",
        "input_blocks_3",
        "input_blocks_4",
        "input_blocks_5",
        "input_blocks_6",
        "input_blocks_7",
        "input_blocks_8",
        "input_blocks_9",
        "input_blocks_10",
        "input_blocks_11",
        "middle_block_0",
        "middle_block_1",
        "middle_block_2",
        "output_blocks_0",
        "output_blocks_1",
        "output_blocks_2",
        "output_blocks_3",
        "output_blocks_4",
        "output_blocks_5",
        "output_blocks_6",
        "output_blocks_7",
        "output_blocks_8",
        "output_blocks_9",
        "output_blocks_10",
        "output_blocks_11",
        "out",
    )
    CATEGORY = "model/merging/model specific"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {
        "model1": "MODEL",
        "model2": "MODEL",
        "time_embed": "FLOAT",
        "label_emb": "FLOAT",
        "input_blocks_0": "FLOAT",
        "input_blocks_1": "FLOAT",
        "input_blocks_2": "FLOAT",
        "input_blocks_3": "FLOAT",
        "input_blocks_4": "FLOAT",
        "input_blocks_5": "FLOAT",
        "input_blocks_6": "FLOAT",
        "input_blocks_7": "FLOAT",
        "input_blocks_8": "FLOAT",
        "input_blocks_9": "FLOAT",
        "input_blocks_10": "FLOAT",
        "input_blocks_11": "FLOAT",
        "middle_block_0": "FLOAT",
        "middle_block_1": "FLOAT",
        "middle_block_2": "FLOAT",
        "output_blocks_0": "FLOAT",
        "output_blocks_1": "FLOAT",
        "output_blocks_2": "FLOAT",
        "output_blocks_3": "FLOAT",
        "output_blocks_4": "FLOAT",
        "output_blocks_5": "FLOAT",
        "output_blocks_6": "FLOAT",
        "output_blocks_7": "FLOAT",
        "output_blocks_8": "FLOAT",
        "output_blocks_9": "FLOAT",
        "output_blocks_10": "FLOAT",
        "output_blocks_11": "FLOAT",
        "out": "FLOAT",
    }
    INPUT_NAMES = {
        "time_embed": "time_embed.",
        "label_emb": "label_emb.",
        "input_blocks_0": "input_blocks.0.",
        "input_blocks_1": "input_blocks.1.",
        "input_blocks_2": "input_blocks.2.",
        "input_blocks_3": "input_blocks.3.",
        "input_blocks_4": "input_blocks.4.",
        "input_blocks_5": "input_blocks.5.",
        "input_blocks_6": "input_blocks.6.",
        "input_blocks_7": "input_blocks.7.",
        "input_blocks_8": "input_blocks.8.",
        "input_blocks_9": "input_blocks.9.",
        "input_blocks_10": "input_blocks.10.",
        "input_blocks_11": "input_blocks.11.",
        "middle_block_0": "middle_block.0.",
        "middle_block_1": "middle_block.1.",
        "middle_block_2": "middle_block.2.",
        "output_blocks_0": "output_blocks.0.",
        "output_blocks_1": "output_blocks.1.",
        "output_blocks_2": "output_blocks.2.",
        "output_blocks_3": "output_blocks.3.",
        "output_blocks_4": "output_blocks.4.",
        "output_blocks_5": "output_blocks.5.",
        "output_blocks_6": "output_blocks.6.",
        "output_blocks_7": "output_blocks.7.",
        "output_blocks_8": "output_blocks.8.",
        "output_blocks_9": "output_blocks.9.",
        "output_blocks_10": "output_blocks.10.",
        "output_blocks_11": "output_blocks.11.",
        "out": "out.",
    }

    def __init__(
        self,
        node_id: str,
        model1: tuple[str, int],
        model2: tuple[str, int],
        time_embed: float | tuple[str, int] = 1.0,
        label_emb: float | tuple[str, int] = 1.0,
        input_blocks_0: float | tuple[str, int] = 1.0,
        input_blocks_1: float | tuple[str, int] = 1.0,
        input_blocks_2: float | tuple[str, int] = 1.0,
        input_blocks_3: float | tuple[str, int] = 1.0,
        input_blocks_4: float | tuple[str, int] = 1.0,
        input_blocks_5: float | tuple[str, int] = 1.0,
        input_blocks_6: float | tuple[str, int] = 1.0,
        input_blocks_7: float | tuple[str, int] = 1.0,
        input_blocks_8: float | tuple[str, int] = 1.0,
        input_blocks_9: float | tuple[str, int] = 1.0,
        input_blocks_10: float | tuple[str, int] = 1.0,
        input_blocks_11: float | tuple[str, int] = 1.0,
        middle_block_0: float | tuple[str, int] = 1.0,
        middle_block_1: float | tuple[str, int] = 1.0,
        middle_block_2: float | tuple[str, int] = 1.0,
        output_blocks_0: float | tuple[str, int] = 1.0,
        output_blocks_1: float | tuple[str, int] = 1.0,
        output_blocks_2: float | tuple[str, int] = 1.0,
        output_blocks_3: float | tuple[str, int] = 1.0,
        output_blocks_4: float | tuple[str, int] = 1.0,
        output_blocks_5: float | tuple[str, int] = 1.0,
        output_blocks_6: float | tuple[str, int] = 1.0,
        output_blocks_7: float | tuple[str, int] = 1.0,
        output_blocks_8: float | tuple[str, int] = 1.0,
        output_blocks_9: float | tuple[str, int] = 1.0,
        output_blocks_10: float | tuple[str, int] = 1.0,
        output_blocks_11: float | tuple[str, int] = 1.0,
        out: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model1 = model1
        self.model2 = model2
        self.time_embed = time_embed
        self.label_emb = label_emb
        self.input_blocks_0 = input_blocks_0
        self.input_blocks_1 = input_blocks_1
        self.input_blocks_2 = input_blocks_2
        self.input_blocks_3 = input_blocks_3
        self.input_blocks_4 = input_blocks_4
        self.input_blocks_5 = input_blocks_5
        self.input_blocks_6 = input_blocks_6
        self.input_blocks_7 = input_blocks_7
        self.input_blocks_8 = input_blocks_8
        self.input_blocks_9 = input_blocks_9
        self.input_blocks_10 = input_blocks_10
        self.input_blocks_11 = input_blocks_11
        self.middle_block_0 = middle_block_0
        self.middle_block_1 = middle_block_1
        self.middle_block_2 = middle_block_2
        self.output_blocks_0 = output_blocks_0
        self.output_blocks_1 = output_blocks_1
        self.output_blocks_2 = output_blocks_2
        self.output_blocks_3 = output_blocks_3
        self.output_blocks_4 = output_blocks_4
        self.output_blocks_5 = output_blocks_5
        self.output_blocks_6 = output_blocks_6
        self.output_blocks_7 = output_blocks_7
        self.output_blocks_8 = output_blocks_8
        self.output_blocks_9 = output_blocks_9
        self.output_blocks_10 = output_blocks_10
        self.output_blocks_11 = output_blocks_11
        self.out = out
        self.__validate()

    def __validate(self) -> None:
        validate_range("time_embed", self.time_embed, 0.0, 1.0)
        validate_range("label_emb", self.label_emb, 0.0, 1.0)
        validate_range("input_blocks_0", self.input_blocks_0, 0.0, 1.0)
        validate_range("input_blocks_1", self.input_blocks_1, 0.0, 1.0)
        validate_range("input_blocks_2", self.input_blocks_2, 0.0, 1.0)
        validate_range("input_blocks_3", self.input_blocks_3, 0.0, 1.0)
        validate_range("input_blocks_4", self.input_blocks_4, 0.0, 1.0)
        validate_range("input_blocks_5", self.input_blocks_5, 0.0, 1.0)
        validate_range("input_blocks_6", self.input_blocks_6, 0.0, 1.0)
        validate_range("input_blocks_7", self.input_blocks_7, 0.0, 1.0)
        validate_range("input_blocks_8", self.input_blocks_8, 0.0, 1.0)
        validate_range("input_blocks_9", self.input_blocks_9, 0.0, 1.0)
        validate_range("input_blocks_10", self.input_blocks_10, 0.0, 1.0)
        validate_range("input_blocks_11", self.input_blocks_11, 0.0, 1.0)
        validate_range("middle_block_0", self.middle_block_0, 0.0, 1.0)
        validate_range("middle_block_1", self.middle_block_1, 0.0, 1.0)
        validate_range("middle_block_2", self.middle_block_2, 0.0, 1.0)
        validate_range("output_blocks_0", self.output_blocks_0, 0.0, 1.0)
        validate_range("output_blocks_1", self.output_blocks_1, 0.0, 1.0)
        validate_range("output_blocks_2", self.output_blocks_2, 0.0, 1.0)
        validate_range("output_blocks_3", self.output_blocks_3, 0.0, 1.0)
        validate_range("output_blocks_4", self.output_blocks_4, 0.0, 1.0)
        validate_range("output_blocks_5", self.output_blocks_5, 0.0, 1.0)
        validate_range("output_blocks_6", self.output_blocks_6, 0.0, 1.0)
        validate_range("output_blocks_7", self.output_blocks_7, 0.0, 1.0)
        validate_range("output_blocks_8", self.output_blocks_8, 0.0, 1.0)
        validate_range("output_blocks_9", self.output_blocks_9, 0.0, 1.0)
        validate_range("output_blocks_10", self.output_blocks_10, 0.0, 1.0)
        validate_range("output_blocks_11", self.output_blocks_11, 0.0, 1.0)
        validate_range("out", self.out, 0.0, 1.0)


class ModelMergeSD2(BaseNode):
    __slots__ = (
        "model1",
        "model2",
        "time_embed",
        "label_emb",
        "input_blocks_0",
        "input_blocks_1",
        "input_blocks_2",
        "input_blocks_3",
        "input_blocks_4",
        "input_blocks_5",
        "input_blocks_6",
        "input_blocks_7",
        "input_blocks_8",
        "input_blocks_9",
        "input_blocks_10",
        "input_blocks_11",
        "middle_block_0",
        "middle_block_1",
        "middle_block_2",
        "output_blocks_0",
        "output_blocks_1",
        "output_blocks_2",
        "output_blocks_3",
        "output_blocks_4",
        "output_blocks_5",
        "output_blocks_6",
        "output_blocks_7",
        "output_blocks_8",
        "output_blocks_9",
        "output_blocks_10",
        "output_blocks_11",
        "out",
    )
    CATEGORY = "model/merging/model specific"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {
        "model1": "MODEL",
        "model2": "MODEL",
        "time_embed": "FLOAT",
        "label_emb": "FLOAT",
        "input_blocks_0": "FLOAT",
        "input_blocks_1": "FLOAT",
        "input_blocks_2": "FLOAT",
        "input_blocks_3": "FLOAT",
        "input_blocks_4": "FLOAT",
        "input_blocks_5": "FLOAT",
        "input_blocks_6": "FLOAT",
        "input_blocks_7": "FLOAT",
        "input_blocks_8": "FLOAT",
        "input_blocks_9": "FLOAT",
        "input_blocks_10": "FLOAT",
        "input_blocks_11": "FLOAT",
        "middle_block_0": "FLOAT",
        "middle_block_1": "FLOAT",
        "middle_block_2": "FLOAT",
        "output_blocks_0": "FLOAT",
        "output_blocks_1": "FLOAT",
        "output_blocks_2": "FLOAT",
        "output_blocks_3": "FLOAT",
        "output_blocks_4": "FLOAT",
        "output_blocks_5": "FLOAT",
        "output_blocks_6": "FLOAT",
        "output_blocks_7": "FLOAT",
        "output_blocks_8": "FLOAT",
        "output_blocks_9": "FLOAT",
        "output_blocks_10": "FLOAT",
        "output_blocks_11": "FLOAT",
        "out": "FLOAT",
    }
    INPUT_NAMES = {
        "time_embed": "time_embed.",
        "label_emb": "label_emb.",
        "input_blocks_0": "input_blocks.0.",
        "input_blocks_1": "input_blocks.1.",
        "input_blocks_2": "input_blocks.2.",
        "input_blocks_3": "input_blocks.3.",
        "input_blocks_4": "input_blocks.4.",
        "input_blocks_5": "input_blocks.5.",
        "input_blocks_6": "input_blocks.6.",
        "input_blocks_7": "input_blocks.7.",
        "input_blocks_8": "input_blocks.8.",
        "input_blocks_9": "input_blocks.9.",
        "input_blocks_10": "input_blocks.10.",
        "input_blocks_11": "input_blocks.11.",
        "middle_block_0": "middle_block.0.",
        "middle_block_1": "middle_block.1.",
        "middle_block_2": "middle_block.2.",
        "output_blocks_0": "output_blocks.0.",
        "output_blocks_1": "output_blocks.1.",
        "output_blocks_2": "output_blocks.2.",
        "output_blocks_3": "output_blocks.3.",
        "output_blocks_4": "output_blocks.4.",
        "output_blocks_5": "output_blocks.5.",
        "output_blocks_6": "output_blocks.6.",
        "output_blocks_7": "output_blocks.7.",
        "output_blocks_8": "output_blocks.8.",
        "output_blocks_9": "output_blocks.9.",
        "output_blocks_10": "output_blocks.10.",
        "output_blocks_11": "output_blocks.11.",
        "out": "out.",
    }

    def __init__(
        self,
        node_id: str,
        model1: tuple[str, int],
        model2: tuple[str, int],
        time_embed: float | tuple[str, int] = 1.0,
        label_emb: float | tuple[str, int] = 1.0,
        input_blocks_0: float | tuple[str, int] = 1.0,
        input_blocks_1: float | tuple[str, int] = 1.0,
        input_blocks_2: float | tuple[str, int] = 1.0,
        input_blocks_3: float | tuple[str, int] = 1.0,
        input_blocks_4: float | tuple[str, int] = 1.0,
        input_blocks_5: float | tuple[str, int] = 1.0,
        input_blocks_6: float | tuple[str, int] = 1.0,
        input_blocks_7: float | tuple[str, int] = 1.0,
        input_blocks_8: float | tuple[str, int] = 1.0,
        input_blocks_9: float | tuple[str, int] = 1.0,
        input_blocks_10: float | tuple[str, int] = 1.0,
        input_blocks_11: float | tuple[str, int] = 1.0,
        middle_block_0: float | tuple[str, int] = 1.0,
        middle_block_1: float | tuple[str, int] = 1.0,
        middle_block_2: float | tuple[str, int] = 1.0,
        output_blocks_0: float | tuple[str, int] = 1.0,
        output_blocks_1: float | tuple[str, int] = 1.0,
        output_blocks_2: float | tuple[str, int] = 1.0,
        output_blocks_3: float | tuple[str, int] = 1.0,
        output_blocks_4: float | tuple[str, int] = 1.0,
        output_blocks_5: float | tuple[str, int] = 1.0,
        output_blocks_6: float | tuple[str, int] = 1.0,
        output_blocks_7: float | tuple[str, int] = 1.0,
        output_blocks_8: float | tuple[str, int] = 1.0,
        output_blocks_9: float | tuple[str, int] = 1.0,
        output_blocks_10: float | tuple[str, int] = 1.0,
        output_blocks_11: float | tuple[str, int] = 1.0,
        out: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model1 = model1
        self.model2 = model2
        self.time_embed = time_embed
        self.label_emb = label_emb
        self.input_blocks_0 = input_blocks_0
        self.input_blocks_1 = input_blocks_1
        self.input_blocks_2 = input_blocks_2
        self.input_blocks_3 = input_blocks_3
        self.input_blocks_4 = input_blocks_4
        self.input_blocks_5 = input_blocks_5
        self.input_blocks_6 = input_blocks_6
        self.input_blocks_7 = input_blocks_7
        self.input_blocks_8 = input_blocks_8
        self.input_blocks_9 = input_blocks_9
        self.input_blocks_10 = input_blocks_10
        self.input_blocks_11 = input_blocks_11
        self.middle_block_0 = middle_block_0
        self.middle_block_1 = middle_block_1
        self.middle_block_2 = middle_block_2
        self.output_blocks_0 = output_blocks_0
        self.output_blocks_1 = output_blocks_1
        self.output_blocks_2 = output_blocks_2
        self.output_blocks_3 = output_blocks_3
        self.output_blocks_4 = output_blocks_4
        self.output_blocks_5 = output_blocks_5
        self.output_blocks_6 = output_blocks_6
        self.output_blocks_7 = output_blocks_7
        self.output_blocks_8 = output_blocks_8
        self.output_blocks_9 = output_blocks_9
        self.output_blocks_10 = output_blocks_10
        self.output_blocks_11 = output_blocks_11
        self.out = out
        self.__validate()

    def __validate(self) -> None:
        validate_range("time_embed", self.time_embed, 0.0, 1.0)
        validate_range("label_emb", self.label_emb, 0.0, 1.0)
        validate_range("input_blocks_0", self.input_blocks_0, 0.0, 1.0)
        validate_range("input_blocks_1", self.input_blocks_1, 0.0, 1.0)
        validate_range("input_blocks_2", self.input_blocks_2, 0.0, 1.0)
        validate_range("input_blocks_3", self.input_blocks_3, 0.0, 1.0)
        validate_range("input_blocks_4", self.input_blocks_4, 0.0, 1.0)
        validate_range("input_blocks_5", self.input_blocks_5, 0.0, 1.0)
        validate_range("input_blocks_6", self.input_blocks_6, 0.0, 1.0)
        validate_range("input_blocks_7", self.input_blocks_7, 0.0, 1.0)
        validate_range("input_blocks_8", self.input_blocks_8, 0.0, 1.0)
        validate_range("input_blocks_9", self.input_blocks_9, 0.0, 1.0)
        validate_range("input_blocks_10", self.input_blocks_10, 0.0, 1.0)
        validate_range("input_blocks_11", self.input_blocks_11, 0.0, 1.0)
        validate_range("middle_block_0", self.middle_block_0, 0.0, 1.0)
        validate_range("middle_block_1", self.middle_block_1, 0.0, 1.0)
        validate_range("middle_block_2", self.middle_block_2, 0.0, 1.0)
        validate_range("output_blocks_0", self.output_blocks_0, 0.0, 1.0)
        validate_range("output_blocks_1", self.output_blocks_1, 0.0, 1.0)
        validate_range("output_blocks_2", self.output_blocks_2, 0.0, 1.0)
        validate_range("output_blocks_3", self.output_blocks_3, 0.0, 1.0)
        validate_range("output_blocks_4", self.output_blocks_4, 0.0, 1.0)
        validate_range("output_blocks_5", self.output_blocks_5, 0.0, 1.0)
        validate_range("output_blocks_6", self.output_blocks_6, 0.0, 1.0)
        validate_range("output_blocks_7", self.output_blocks_7, 0.0, 1.0)
        validate_range("output_blocks_8", self.output_blocks_8, 0.0, 1.0)
        validate_range("output_blocks_9", self.output_blocks_9, 0.0, 1.0)
        validate_range("output_blocks_10", self.output_blocks_10, 0.0, 1.0)
        validate_range("output_blocks_11", self.output_blocks_11, 0.0, 1.0)
        validate_range("out", self.out, 0.0, 1.0)


class ModelMergeSDXL(BaseNode):
    __slots__ = (
        "model1",
        "model2",
        "time_embed",
        "label_emb",
        "input_blocks_0",
        "input_blocks_1",
        "input_blocks_2",
        "input_blocks_3",
        "input_blocks_4",
        "input_blocks_5",
        "input_blocks_6",
        "input_blocks_7",
        "input_blocks_8",
        "middle_block_0",
        "middle_block_1",
        "middle_block_2",
        "output_blocks_0",
        "output_blocks_1",
        "output_blocks_2",
        "output_blocks_3",
        "output_blocks_4",
        "output_blocks_5",
        "output_blocks_6",
        "output_blocks_7",
        "output_blocks_8",
        "out",
    )
    CATEGORY = "model/merging/model specific"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {
        "model1": "MODEL",
        "model2": "MODEL",
        "time_embed": "FLOAT",
        "label_emb": "FLOAT",
        "input_blocks_0": "FLOAT",
        "input_blocks_1": "FLOAT",
        "input_blocks_2": "FLOAT",
        "input_blocks_3": "FLOAT",
        "input_blocks_4": "FLOAT",
        "input_blocks_5": "FLOAT",
        "input_blocks_6": "FLOAT",
        "input_blocks_7": "FLOAT",
        "input_blocks_8": "FLOAT",
        "middle_block_0": "FLOAT",
        "middle_block_1": "FLOAT",
        "middle_block_2": "FLOAT",
        "output_blocks_0": "FLOAT",
        "output_blocks_1": "FLOAT",
        "output_blocks_2": "FLOAT",
        "output_blocks_3": "FLOAT",
        "output_blocks_4": "FLOAT",
        "output_blocks_5": "FLOAT",
        "output_blocks_6": "FLOAT",
        "output_blocks_7": "FLOAT",
        "output_blocks_8": "FLOAT",
        "out": "FLOAT",
    }
    INPUT_NAMES = {
        "time_embed": "time_embed.",
        "label_emb": "label_emb.",
        "input_blocks_0": "input_blocks.0",
        "input_blocks_1": "input_blocks.1",
        "input_blocks_2": "input_blocks.2",
        "input_blocks_3": "input_blocks.3",
        "input_blocks_4": "input_blocks.4",
        "input_blocks_5": "input_blocks.5",
        "input_blocks_6": "input_blocks.6",
        "input_blocks_7": "input_blocks.7",
        "input_blocks_8": "input_blocks.8",
        "middle_block_0": "middle_block.0",
        "middle_block_1": "middle_block.1",
        "middle_block_2": "middle_block.2",
        "output_blocks_0": "output_blocks.0",
        "output_blocks_1": "output_blocks.1",
        "output_blocks_2": "output_blocks.2",
        "output_blocks_3": "output_blocks.3",
        "output_blocks_4": "output_blocks.4",
        "output_blocks_5": "output_blocks.5",
        "output_blocks_6": "output_blocks.6",
        "output_blocks_7": "output_blocks.7",
        "output_blocks_8": "output_blocks.8",
        "out": "out.",
    }

    def __init__(
        self,
        node_id: str,
        model1: tuple[str, int],
        model2: tuple[str, int],
        time_embed: float | tuple[str, int] = 1.0,
        label_emb: float | tuple[str, int] = 1.0,
        input_blocks_0: float | tuple[str, int] = 1.0,
        input_blocks_1: float | tuple[str, int] = 1.0,
        input_blocks_2: float | tuple[str, int] = 1.0,
        input_blocks_3: float | tuple[str, int] = 1.0,
        input_blocks_4: float | tuple[str, int] = 1.0,
        input_blocks_5: float | tuple[str, int] = 1.0,
        input_blocks_6: float | tuple[str, int] = 1.0,
        input_blocks_7: float | tuple[str, int] = 1.0,
        input_blocks_8: float | tuple[str, int] = 1.0,
        middle_block_0: float | tuple[str, int] = 1.0,
        middle_block_1: float | tuple[str, int] = 1.0,
        middle_block_2: float | tuple[str, int] = 1.0,
        output_blocks_0: float | tuple[str, int] = 1.0,
        output_blocks_1: float | tuple[str, int] = 1.0,
        output_blocks_2: float | tuple[str, int] = 1.0,
        output_blocks_3: float | tuple[str, int] = 1.0,
        output_blocks_4: float | tuple[str, int] = 1.0,
        output_blocks_5: float | tuple[str, int] = 1.0,
        output_blocks_6: float | tuple[str, int] = 1.0,
        output_blocks_7: float | tuple[str, int] = 1.0,
        output_blocks_8: float | tuple[str, int] = 1.0,
        out: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model1 = model1
        self.model2 = model2
        self.time_embed = time_embed
        self.label_emb = label_emb
        self.input_blocks_0 = input_blocks_0
        self.input_blocks_1 = input_blocks_1
        self.input_blocks_2 = input_blocks_2
        self.input_blocks_3 = input_blocks_3
        self.input_blocks_4 = input_blocks_4
        self.input_blocks_5 = input_blocks_5
        self.input_blocks_6 = input_blocks_6
        self.input_blocks_7 = input_blocks_7
        self.input_blocks_8 = input_blocks_8
        self.middle_block_0 = middle_block_0
        self.middle_block_1 = middle_block_1
        self.middle_block_2 = middle_block_2
        self.output_blocks_0 = output_blocks_0
        self.output_blocks_1 = output_blocks_1
        self.output_blocks_2 = output_blocks_2
        self.output_blocks_3 = output_blocks_3
        self.output_blocks_4 = output_blocks_4
        self.output_blocks_5 = output_blocks_5
        self.output_blocks_6 = output_blocks_6
        self.output_blocks_7 = output_blocks_7
        self.output_blocks_8 = output_blocks_8
        self.out = out
        self.__validate()

    def __validate(self) -> None:
        validate_range("time_embed", self.time_embed, 0.0, 1.0)
        validate_range("label_emb", self.label_emb, 0.0, 1.0)
        validate_range("input_blocks_0", self.input_blocks_0, 0.0, 1.0)
        validate_range("input_blocks_1", self.input_blocks_1, 0.0, 1.0)
        validate_range("input_blocks_2", self.input_blocks_2, 0.0, 1.0)
        validate_range("input_blocks_3", self.input_blocks_3, 0.0, 1.0)
        validate_range("input_blocks_4", self.input_blocks_4, 0.0, 1.0)
        validate_range("input_blocks_5", self.input_blocks_5, 0.0, 1.0)
        validate_range("input_blocks_6", self.input_blocks_6, 0.0, 1.0)
        validate_range("input_blocks_7", self.input_blocks_7, 0.0, 1.0)
        validate_range("input_blocks_8", self.input_blocks_8, 0.0, 1.0)
        validate_range("middle_block_0", self.middle_block_0, 0.0, 1.0)
        validate_range("middle_block_1", self.middle_block_1, 0.0, 1.0)
        validate_range("middle_block_2", self.middle_block_2, 0.0, 1.0)
        validate_range("output_blocks_0", self.output_blocks_0, 0.0, 1.0)
        validate_range("output_blocks_1", self.output_blocks_1, 0.0, 1.0)
        validate_range("output_blocks_2", self.output_blocks_2, 0.0, 1.0)
        validate_range("output_blocks_3", self.output_blocks_3, 0.0, 1.0)
        validate_range("output_blocks_4", self.output_blocks_4, 0.0, 1.0)
        validate_range("output_blocks_5", self.output_blocks_5, 0.0, 1.0)
        validate_range("output_blocks_6", self.output_blocks_6, 0.0, 1.0)
        validate_range("output_blocks_7", self.output_blocks_7, 0.0, 1.0)
        validate_range("output_blocks_8", self.output_blocks_8, 0.0, 1.0)
        validate_range("out", self.out, 0.0, 1.0)


class ModelMergeSD3_2B(BaseNode):
    __slots__ = (
        "model1",
        "model2",
        "pos_embed",
        "x_embedder",
        "context_embedder",
        "y_embedder",
        "t_embedder",
        "joint_blocks_0",
        "joint_blocks_1",
        "joint_blocks_2",
        "joint_blocks_3",
        "joint_blocks_4",
        "joint_blocks_5",
        "joint_blocks_6",
        "joint_blocks_7",
        "joint_blocks_8",
        "joint_blocks_9",
        "joint_blocks_10",
        "joint_blocks_11",
        "joint_blocks_12",
        "joint_blocks_13",
        "joint_blocks_14",
        "joint_blocks_15",
        "joint_blocks_16",
        "joint_blocks_17",
        "joint_blocks_18",
        "joint_blocks_19",
        "joint_blocks_20",
        "joint_blocks_21",
        "joint_blocks_22",
        "joint_blocks_23",
        "final_layer",
    )
    CATEGORY = "model/merging/model specific"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {
        "model1": "MODEL",
        "model2": "MODEL",
        "pos_embed": "FLOAT",
        "x_embedder": "FLOAT",
        "context_embedder": "FLOAT",
        "y_embedder": "FLOAT",
        "t_embedder": "FLOAT",
        "joint_blocks_0": "FLOAT",
        "joint_blocks_1": "FLOAT",
        "joint_blocks_2": "FLOAT",
        "joint_blocks_3": "FLOAT",
        "joint_blocks_4": "FLOAT",
        "joint_blocks_5": "FLOAT",
        "joint_blocks_6": "FLOAT",
        "joint_blocks_7": "FLOAT",
        "joint_blocks_8": "FLOAT",
        "joint_blocks_9": "FLOAT",
        "joint_blocks_10": "FLOAT",
        "joint_blocks_11": "FLOAT",
        "joint_blocks_12": "FLOAT",
        "joint_blocks_13": "FLOAT",
        "joint_blocks_14": "FLOAT",
        "joint_blocks_15": "FLOAT",
        "joint_blocks_16": "FLOAT",
        "joint_blocks_17": "FLOAT",
        "joint_blocks_18": "FLOAT",
        "joint_blocks_19": "FLOAT",
        "joint_blocks_20": "FLOAT",
        "joint_blocks_21": "FLOAT",
        "joint_blocks_22": "FLOAT",
        "joint_blocks_23": "FLOAT",
        "final_layer": "FLOAT",
    }
    INPUT_NAMES = {
        "pos_embed": "pos_embed.",
        "x_embedder": "x_embedder.",
        "context_embedder": "context_embedder.",
        "y_embedder": "y_embedder.",
        "t_embedder": "t_embedder.",
        "joint_blocks_0": "joint_blocks.0.",
        "joint_blocks_1": "joint_blocks.1.",
        "joint_blocks_2": "joint_blocks.2.",
        "joint_blocks_3": "joint_blocks.3.",
        "joint_blocks_4": "joint_blocks.4.",
        "joint_blocks_5": "joint_blocks.5.",
        "joint_blocks_6": "joint_blocks.6.",
        "joint_blocks_7": "joint_blocks.7.",
        "joint_blocks_8": "joint_blocks.8.",
        "joint_blocks_9": "joint_blocks.9.",
        "joint_blocks_10": "joint_blocks.10.",
        "joint_blocks_11": "joint_blocks.11.",
        "joint_blocks_12": "joint_blocks.12.",
        "joint_blocks_13": "joint_blocks.13.",
        "joint_blocks_14": "joint_blocks.14.",
        "joint_blocks_15": "joint_blocks.15.",
        "joint_blocks_16": "joint_blocks.16.",
        "joint_blocks_17": "joint_blocks.17.",
        "joint_blocks_18": "joint_blocks.18.",
        "joint_blocks_19": "joint_blocks.19.",
        "joint_blocks_20": "joint_blocks.20.",
        "joint_blocks_21": "joint_blocks.21.",
        "joint_blocks_22": "joint_blocks.22.",
        "joint_blocks_23": "joint_blocks.23.",
        "final_layer": "final_layer.",
    }

    def __init__(
        self,
        node_id: str,
        model1: tuple[str, int],
        model2: tuple[str, int],
        pos_embed: float | tuple[str, int] = 1.0,
        x_embedder: float | tuple[str, int] = 1.0,
        context_embedder: float | tuple[str, int] = 1.0,
        y_embedder: float | tuple[str, int] = 1.0,
        t_embedder: float | tuple[str, int] = 1.0,
        joint_blocks_0: float | tuple[str, int] = 1.0,
        joint_blocks_1: float | tuple[str, int] = 1.0,
        joint_blocks_2: float | tuple[str, int] = 1.0,
        joint_blocks_3: float | tuple[str, int] = 1.0,
        joint_blocks_4: float | tuple[str, int] = 1.0,
        joint_blocks_5: float | tuple[str, int] = 1.0,
        joint_blocks_6: float | tuple[str, int] = 1.0,
        joint_blocks_7: float | tuple[str, int] = 1.0,
        joint_blocks_8: float | tuple[str, int] = 1.0,
        joint_blocks_9: float | tuple[str, int] = 1.0,
        joint_blocks_10: float | tuple[str, int] = 1.0,
        joint_blocks_11: float | tuple[str, int] = 1.0,
        joint_blocks_12: float | tuple[str, int] = 1.0,
        joint_blocks_13: float | tuple[str, int] = 1.0,
        joint_blocks_14: float | tuple[str, int] = 1.0,
        joint_blocks_15: float | tuple[str, int] = 1.0,
        joint_blocks_16: float | tuple[str, int] = 1.0,
        joint_blocks_17: float | tuple[str, int] = 1.0,
        joint_blocks_18: float | tuple[str, int] = 1.0,
        joint_blocks_19: float | tuple[str, int] = 1.0,
        joint_blocks_20: float | tuple[str, int] = 1.0,
        joint_blocks_21: float | tuple[str, int] = 1.0,
        joint_blocks_22: float | tuple[str, int] = 1.0,
        joint_blocks_23: float | tuple[str, int] = 1.0,
        final_layer: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model1 = model1
        self.model2 = model2
        self.pos_embed = pos_embed
        self.x_embedder = x_embedder
        self.context_embedder = context_embedder
        self.y_embedder = y_embedder
        self.t_embedder = t_embedder
        self.joint_blocks_0 = joint_blocks_0
        self.joint_blocks_1 = joint_blocks_1
        self.joint_blocks_2 = joint_blocks_2
        self.joint_blocks_3 = joint_blocks_3
        self.joint_blocks_4 = joint_blocks_4
        self.joint_blocks_5 = joint_blocks_5
        self.joint_blocks_6 = joint_blocks_6
        self.joint_blocks_7 = joint_blocks_7
        self.joint_blocks_8 = joint_blocks_8
        self.joint_blocks_9 = joint_blocks_9
        self.joint_blocks_10 = joint_blocks_10
        self.joint_blocks_11 = joint_blocks_11
        self.joint_blocks_12 = joint_blocks_12
        self.joint_blocks_13 = joint_blocks_13
        self.joint_blocks_14 = joint_blocks_14
        self.joint_blocks_15 = joint_blocks_15
        self.joint_blocks_16 = joint_blocks_16
        self.joint_blocks_17 = joint_blocks_17
        self.joint_blocks_18 = joint_blocks_18
        self.joint_blocks_19 = joint_blocks_19
        self.joint_blocks_20 = joint_blocks_20
        self.joint_blocks_21 = joint_blocks_21
        self.joint_blocks_22 = joint_blocks_22
        self.joint_blocks_23 = joint_blocks_23
        self.final_layer = final_layer
        self.__validate()

    def __validate(self) -> None:
        validate_range("pos_embed", self.pos_embed, 0.0, 1.0)
        validate_range("x_embedder", self.x_embedder, 0.0, 1.0)
        validate_range("context_embedder", self.context_embedder, 0.0, 1.0)
        validate_range("y_embedder", self.y_embedder, 0.0, 1.0)
        validate_range("t_embedder", self.t_embedder, 0.0, 1.0)
        validate_range("joint_blocks_0", self.joint_blocks_0, 0.0, 1.0)
        validate_range("joint_blocks_1", self.joint_blocks_1, 0.0, 1.0)
        validate_range("joint_blocks_2", self.joint_blocks_2, 0.0, 1.0)
        validate_range("joint_blocks_3", self.joint_blocks_3, 0.0, 1.0)
        validate_range("joint_blocks_4", self.joint_blocks_4, 0.0, 1.0)
        validate_range("joint_blocks_5", self.joint_blocks_5, 0.0, 1.0)
        validate_range("joint_blocks_6", self.joint_blocks_6, 0.0, 1.0)
        validate_range("joint_blocks_7", self.joint_blocks_7, 0.0, 1.0)
        validate_range("joint_blocks_8", self.joint_blocks_8, 0.0, 1.0)
        validate_range("joint_blocks_9", self.joint_blocks_9, 0.0, 1.0)
        validate_range("joint_blocks_10", self.joint_blocks_10, 0.0, 1.0)
        validate_range("joint_blocks_11", self.joint_blocks_11, 0.0, 1.0)
        validate_range("joint_blocks_12", self.joint_blocks_12, 0.0, 1.0)
        validate_range("joint_blocks_13", self.joint_blocks_13, 0.0, 1.0)
        validate_range("joint_blocks_14", self.joint_blocks_14, 0.0, 1.0)
        validate_range("joint_blocks_15", self.joint_blocks_15, 0.0, 1.0)
        validate_range("joint_blocks_16", self.joint_blocks_16, 0.0, 1.0)
        validate_range("joint_blocks_17", self.joint_blocks_17, 0.0, 1.0)
        validate_range("joint_blocks_18", self.joint_blocks_18, 0.0, 1.0)
        validate_range("joint_blocks_19", self.joint_blocks_19, 0.0, 1.0)
        validate_range("joint_blocks_20", self.joint_blocks_20, 0.0, 1.0)
        validate_range("joint_blocks_21", self.joint_blocks_21, 0.0, 1.0)
        validate_range("joint_blocks_22", self.joint_blocks_22, 0.0, 1.0)
        validate_range("joint_blocks_23", self.joint_blocks_23, 0.0, 1.0)
        validate_range("final_layer", self.final_layer, 0.0, 1.0)


class ModelMergeAuraflow(BaseNode):
    __slots__ = (
        "model1",
        "model2",
        "init_x_linear",
        "positional_encoding",
        "cond_seq_linear",
        "register_tokens",
        "t_embedder",
        "double_layers_0",
        "double_layers_1",
        "double_layers_2",
        "double_layers_3",
        "single_layers_0",
        "single_layers_1",
        "single_layers_2",
        "single_layers_3",
        "single_layers_4",
        "single_layers_5",
        "single_layers_6",
        "single_layers_7",
        "single_layers_8",
        "single_layers_9",
        "single_layers_10",
        "single_layers_11",
        "single_layers_12",
        "single_layers_13",
        "single_layers_14",
        "single_layers_15",
        "single_layers_16",
        "single_layers_17",
        "single_layers_18",
        "single_layers_19",
        "single_layers_20",
        "single_layers_21",
        "single_layers_22",
        "single_layers_23",
        "single_layers_24",
        "single_layers_25",
        "single_layers_26",
        "single_layers_27",
        "single_layers_28",
        "single_layers_29",
        "single_layers_30",
        "single_layers_31",
        "modF",
        "final_linear",
    )
    CATEGORY = "model/merging/model specific"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {
        "model1": "MODEL",
        "model2": "MODEL",
        "init_x_linear": "FLOAT",
        "positional_encoding": "FLOAT",
        "cond_seq_linear": "FLOAT",
        "register_tokens": "FLOAT",
        "t_embedder": "FLOAT",
        "double_layers_0": "FLOAT",
        "double_layers_1": "FLOAT",
        "double_layers_2": "FLOAT",
        "double_layers_3": "FLOAT",
        "single_layers_0": "FLOAT",
        "single_layers_1": "FLOAT",
        "single_layers_2": "FLOAT",
        "single_layers_3": "FLOAT",
        "single_layers_4": "FLOAT",
        "single_layers_5": "FLOAT",
        "single_layers_6": "FLOAT",
        "single_layers_7": "FLOAT",
        "single_layers_8": "FLOAT",
        "single_layers_9": "FLOAT",
        "single_layers_10": "FLOAT",
        "single_layers_11": "FLOAT",
        "single_layers_12": "FLOAT",
        "single_layers_13": "FLOAT",
        "single_layers_14": "FLOAT",
        "single_layers_15": "FLOAT",
        "single_layers_16": "FLOAT",
        "single_layers_17": "FLOAT",
        "single_layers_18": "FLOAT",
        "single_layers_19": "FLOAT",
        "single_layers_20": "FLOAT",
        "single_layers_21": "FLOAT",
        "single_layers_22": "FLOAT",
        "single_layers_23": "FLOAT",
        "single_layers_24": "FLOAT",
        "single_layers_25": "FLOAT",
        "single_layers_26": "FLOAT",
        "single_layers_27": "FLOAT",
        "single_layers_28": "FLOAT",
        "single_layers_29": "FLOAT",
        "single_layers_30": "FLOAT",
        "single_layers_31": "FLOAT",
        "modF": "FLOAT",
        "final_linear": "FLOAT",
    }
    INPUT_NAMES = {
        "init_x_linear": "init_x_linear.",
        "cond_seq_linear": "cond_seq_linear.",
        "t_embedder": "t_embedder.",
        "double_layers_0": "double_layers.0.",
        "double_layers_1": "double_layers.1.",
        "double_layers_2": "double_layers.2.",
        "double_layers_3": "double_layers.3.",
        "single_layers_0": "single_layers.0.",
        "single_layers_1": "single_layers.1.",
        "single_layers_2": "single_layers.2.",
        "single_layers_3": "single_layers.3.",
        "single_layers_4": "single_layers.4.",
        "single_layers_5": "single_layers.5.",
        "single_layers_6": "single_layers.6.",
        "single_layers_7": "single_layers.7.",
        "single_layers_8": "single_layers.8.",
        "single_layers_9": "single_layers.9.",
        "single_layers_10": "single_layers.10.",
        "single_layers_11": "single_layers.11.",
        "single_layers_12": "single_layers.12.",
        "single_layers_13": "single_layers.13.",
        "single_layers_14": "single_layers.14.",
        "single_layers_15": "single_layers.15.",
        "single_layers_16": "single_layers.16.",
        "single_layers_17": "single_layers.17.",
        "single_layers_18": "single_layers.18.",
        "single_layers_19": "single_layers.19.",
        "single_layers_20": "single_layers.20.",
        "single_layers_21": "single_layers.21.",
        "single_layers_22": "single_layers.22.",
        "single_layers_23": "single_layers.23.",
        "single_layers_24": "single_layers.24.",
        "single_layers_25": "single_layers.25.",
        "single_layers_26": "single_layers.26.",
        "single_layers_27": "single_layers.27.",
        "single_layers_28": "single_layers.28.",
        "single_layers_29": "single_layers.29.",
        "single_layers_30": "single_layers.30.",
        "single_layers_31": "single_layers.31.",
        "modF": "modF.",
        "final_linear": "final_linear.",
    }

    def __init__(
        self,
        node_id: str,
        model1: tuple[str, int],
        model2: tuple[str, int],
        init_x_linear: float | tuple[str, int] = 1.0,
        positional_encoding: float | tuple[str, int] = 1.0,
        cond_seq_linear: float | tuple[str, int] = 1.0,
        register_tokens: float | tuple[str, int] = 1.0,
        t_embedder: float | tuple[str, int] = 1.0,
        double_layers_0: float | tuple[str, int] = 1.0,
        double_layers_1: float | tuple[str, int] = 1.0,
        double_layers_2: float | tuple[str, int] = 1.0,
        double_layers_3: float | tuple[str, int] = 1.0,
        single_layers_0: float | tuple[str, int] = 1.0,
        single_layers_1: float | tuple[str, int] = 1.0,
        single_layers_2: float | tuple[str, int] = 1.0,
        single_layers_3: float | tuple[str, int] = 1.0,
        single_layers_4: float | tuple[str, int] = 1.0,
        single_layers_5: float | tuple[str, int] = 1.0,
        single_layers_6: float | tuple[str, int] = 1.0,
        single_layers_7: float | tuple[str, int] = 1.0,
        single_layers_8: float | tuple[str, int] = 1.0,
        single_layers_9: float | tuple[str, int] = 1.0,
        single_layers_10: float | tuple[str, int] = 1.0,
        single_layers_11: float | tuple[str, int] = 1.0,
        single_layers_12: float | tuple[str, int] = 1.0,
        single_layers_13: float | tuple[str, int] = 1.0,
        single_layers_14: float | tuple[str, int] = 1.0,
        single_layers_15: float | tuple[str, int] = 1.0,
        single_layers_16: float | tuple[str, int] = 1.0,
        single_layers_17: float | tuple[str, int] = 1.0,
        single_layers_18: float | tuple[str, int] = 1.0,
        single_layers_19: float | tuple[str, int] = 1.0,
        single_layers_20: float | tuple[str, int] = 1.0,
        single_layers_21: float | tuple[str, int] = 1.0,
        single_layers_22: float | tuple[str, int] = 1.0,
        single_layers_23: float | tuple[str, int] = 1.0,
        single_layers_24: float | tuple[str, int] = 1.0,
        single_layers_25: float | tuple[str, int] = 1.0,
        single_layers_26: float | tuple[str, int] = 1.0,
        single_layers_27: float | tuple[str, int] = 1.0,
        single_layers_28: float | tuple[str, int] = 1.0,
        single_layers_29: float | tuple[str, int] = 1.0,
        single_layers_30: float | tuple[str, int] = 1.0,
        single_layers_31: float | tuple[str, int] = 1.0,
        modF: float | tuple[str, int] = 1.0,
        final_linear: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model1 = model1
        self.model2 = model2
        self.init_x_linear = init_x_linear
        self.positional_encoding = positional_encoding
        self.cond_seq_linear = cond_seq_linear
        self.register_tokens = register_tokens
        self.t_embedder = t_embedder
        self.double_layers_0 = double_layers_0
        self.double_layers_1 = double_layers_1
        self.double_layers_2 = double_layers_2
        self.double_layers_3 = double_layers_3
        self.single_layers_0 = single_layers_0
        self.single_layers_1 = single_layers_1
        self.single_layers_2 = single_layers_2
        self.single_layers_3 = single_layers_3
        self.single_layers_4 = single_layers_4
        self.single_layers_5 = single_layers_5
        self.single_layers_6 = single_layers_6
        self.single_layers_7 = single_layers_7
        self.single_layers_8 = single_layers_8
        self.single_layers_9 = single_layers_9
        self.single_layers_10 = single_layers_10
        self.single_layers_11 = single_layers_11
        self.single_layers_12 = single_layers_12
        self.single_layers_13 = single_layers_13
        self.single_layers_14 = single_layers_14
        self.single_layers_15 = single_layers_15
        self.single_layers_16 = single_layers_16
        self.single_layers_17 = single_layers_17
        self.single_layers_18 = single_layers_18
        self.single_layers_19 = single_layers_19
        self.single_layers_20 = single_layers_20
        self.single_layers_21 = single_layers_21
        self.single_layers_22 = single_layers_22
        self.single_layers_23 = single_layers_23
        self.single_layers_24 = single_layers_24
        self.single_layers_25 = single_layers_25
        self.single_layers_26 = single_layers_26
        self.single_layers_27 = single_layers_27
        self.single_layers_28 = single_layers_28
        self.single_layers_29 = single_layers_29
        self.single_layers_30 = single_layers_30
        self.single_layers_31 = single_layers_31
        self.modF = modF
        self.final_linear = final_linear
        self.__validate()

    def __validate(self) -> None:
        validate_range("init_x_linear", self.init_x_linear, 0.0, 1.0)
        validate_range("positional_encoding", self.positional_encoding, 0.0, 1.0)
        validate_range("cond_seq_linear", self.cond_seq_linear, 0.0, 1.0)
        validate_range("register_tokens", self.register_tokens, 0.0, 1.0)
        validate_range("t_embedder", self.t_embedder, 0.0, 1.0)
        validate_range("double_layers_0", self.double_layers_0, 0.0, 1.0)
        validate_range("double_layers_1", self.double_layers_1, 0.0, 1.0)
        validate_range("double_layers_2", self.double_layers_2, 0.0, 1.0)
        validate_range("double_layers_3", self.double_layers_3, 0.0, 1.0)
        validate_range("single_layers_0", self.single_layers_0, 0.0, 1.0)
        validate_range("single_layers_1", self.single_layers_1, 0.0, 1.0)
        validate_range("single_layers_2", self.single_layers_2, 0.0, 1.0)
        validate_range("single_layers_3", self.single_layers_3, 0.0, 1.0)
        validate_range("single_layers_4", self.single_layers_4, 0.0, 1.0)
        validate_range("single_layers_5", self.single_layers_5, 0.0, 1.0)
        validate_range("single_layers_6", self.single_layers_6, 0.0, 1.0)
        validate_range("single_layers_7", self.single_layers_7, 0.0, 1.0)
        validate_range("single_layers_8", self.single_layers_8, 0.0, 1.0)
        validate_range("single_layers_9", self.single_layers_9, 0.0, 1.0)
        validate_range("single_layers_10", self.single_layers_10, 0.0, 1.0)
        validate_range("single_layers_11", self.single_layers_11, 0.0, 1.0)
        validate_range("single_layers_12", self.single_layers_12, 0.0, 1.0)
        validate_range("single_layers_13", self.single_layers_13, 0.0, 1.0)
        validate_range("single_layers_14", self.single_layers_14, 0.0, 1.0)
        validate_range("single_layers_15", self.single_layers_15, 0.0, 1.0)
        validate_range("single_layers_16", self.single_layers_16, 0.0, 1.0)
        validate_range("single_layers_17", self.single_layers_17, 0.0, 1.0)
        validate_range("single_layers_18", self.single_layers_18, 0.0, 1.0)
        validate_range("single_layers_19", self.single_layers_19, 0.0, 1.0)
        validate_range("single_layers_20", self.single_layers_20, 0.0, 1.0)
        validate_range("single_layers_21", self.single_layers_21, 0.0, 1.0)
        validate_range("single_layers_22", self.single_layers_22, 0.0, 1.0)
        validate_range("single_layers_23", self.single_layers_23, 0.0, 1.0)
        validate_range("single_layers_24", self.single_layers_24, 0.0, 1.0)
        validate_range("single_layers_25", self.single_layers_25, 0.0, 1.0)
        validate_range("single_layers_26", self.single_layers_26, 0.0, 1.0)
        validate_range("single_layers_27", self.single_layers_27, 0.0, 1.0)
        validate_range("single_layers_28", self.single_layers_28, 0.0, 1.0)
        validate_range("single_layers_29", self.single_layers_29, 0.0, 1.0)
        validate_range("single_layers_30", self.single_layers_30, 0.0, 1.0)
        validate_range("single_layers_31", self.single_layers_31, 0.0, 1.0)
        validate_range("modF", self.modF, 0.0, 1.0)
        validate_range("final_linear", self.final_linear, 0.0, 1.0)


class ModelMergeFlux1(BaseNode):
    __slots__ = (
        "model1",
        "model2",
        "img_in",
        "time_in",
        "guidance_in",
        "vector_in",
        "txt_in",
        "double_blocks_0",
        "double_blocks_1",
        "double_blocks_2",
        "double_blocks_3",
        "double_blocks_4",
        "double_blocks_5",
        "double_blocks_6",
        "double_blocks_7",
        "double_blocks_8",
        "double_blocks_9",
        "double_blocks_10",
        "double_blocks_11",
        "double_blocks_12",
        "double_blocks_13",
        "double_blocks_14",
        "double_blocks_15",
        "double_blocks_16",
        "double_blocks_17",
        "double_blocks_18",
        "single_blocks_0",
        "single_blocks_1",
        "single_blocks_2",
        "single_blocks_3",
        "single_blocks_4",
        "single_blocks_5",
        "single_blocks_6",
        "single_blocks_7",
        "single_blocks_8",
        "single_blocks_9",
        "single_blocks_10",
        "single_blocks_11",
        "single_blocks_12",
        "single_blocks_13",
        "single_blocks_14",
        "single_blocks_15",
        "single_blocks_16",
        "single_blocks_17",
        "single_blocks_18",
        "single_blocks_19",
        "single_blocks_20",
        "single_blocks_21",
        "single_blocks_22",
        "single_blocks_23",
        "single_blocks_24",
        "single_blocks_25",
        "single_blocks_26",
        "single_blocks_27",
        "single_blocks_28",
        "single_blocks_29",
        "single_blocks_30",
        "single_blocks_31",
        "single_blocks_32",
        "single_blocks_33",
        "single_blocks_34",
        "single_blocks_35",
        "single_blocks_36",
        "single_blocks_37",
        "final_layer",
    )
    CATEGORY = "model/merging/model specific"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {
        "model1": "MODEL",
        "model2": "MODEL",
        "img_in": "FLOAT",
        "time_in": "FLOAT",
        "guidance_in": "FLOAT",
        "vector_in": "FLOAT",
        "txt_in": "FLOAT",
        "double_blocks_0": "FLOAT",
        "double_blocks_1": "FLOAT",
        "double_blocks_2": "FLOAT",
        "double_blocks_3": "FLOAT",
        "double_blocks_4": "FLOAT",
        "double_blocks_5": "FLOAT",
        "double_blocks_6": "FLOAT",
        "double_blocks_7": "FLOAT",
        "double_blocks_8": "FLOAT",
        "double_blocks_9": "FLOAT",
        "double_blocks_10": "FLOAT",
        "double_blocks_11": "FLOAT",
        "double_blocks_12": "FLOAT",
        "double_blocks_13": "FLOAT",
        "double_blocks_14": "FLOAT",
        "double_blocks_15": "FLOAT",
        "double_blocks_16": "FLOAT",
        "double_blocks_17": "FLOAT",
        "double_blocks_18": "FLOAT",
        "single_blocks_0": "FLOAT",
        "single_blocks_1": "FLOAT",
        "single_blocks_2": "FLOAT",
        "single_blocks_3": "FLOAT",
        "single_blocks_4": "FLOAT",
        "single_blocks_5": "FLOAT",
        "single_blocks_6": "FLOAT",
        "single_blocks_7": "FLOAT",
        "single_blocks_8": "FLOAT",
        "single_blocks_9": "FLOAT",
        "single_blocks_10": "FLOAT",
        "single_blocks_11": "FLOAT",
        "single_blocks_12": "FLOAT",
        "single_blocks_13": "FLOAT",
        "single_blocks_14": "FLOAT",
        "single_blocks_15": "FLOAT",
        "single_blocks_16": "FLOAT",
        "single_blocks_17": "FLOAT",
        "single_blocks_18": "FLOAT",
        "single_blocks_19": "FLOAT",
        "single_blocks_20": "FLOAT",
        "single_blocks_21": "FLOAT",
        "single_blocks_22": "FLOAT",
        "single_blocks_23": "FLOAT",
        "single_blocks_24": "FLOAT",
        "single_blocks_25": "FLOAT",
        "single_blocks_26": "FLOAT",
        "single_blocks_27": "FLOAT",
        "single_blocks_28": "FLOAT",
        "single_blocks_29": "FLOAT",
        "single_blocks_30": "FLOAT",
        "single_blocks_31": "FLOAT",
        "single_blocks_32": "FLOAT",
        "single_blocks_33": "FLOAT",
        "single_blocks_34": "FLOAT",
        "single_blocks_35": "FLOAT",
        "single_blocks_36": "FLOAT",
        "single_blocks_37": "FLOAT",
        "final_layer": "FLOAT",
    }
    INPUT_NAMES = {
        "img_in": "img_in.",
        "time_in": "time_in.",
        "vector_in": "vector_in.",
        "txt_in": "txt_in.",
        "double_blocks_0": "double_blocks.0.",
        "double_blocks_1": "double_blocks.1.",
        "double_blocks_2": "double_blocks.2.",
        "double_blocks_3": "double_blocks.3.",
        "double_blocks_4": "double_blocks.4.",
        "double_blocks_5": "double_blocks.5.",
        "double_blocks_6": "double_blocks.6.",
        "double_blocks_7": "double_blocks.7.",
        "double_blocks_8": "double_blocks.8.",
        "double_blocks_9": "double_blocks.9.",
        "double_blocks_10": "double_blocks.10.",
        "double_blocks_11": "double_blocks.11.",
        "double_blocks_12": "double_blocks.12.",
        "double_blocks_13": "double_blocks.13.",
        "double_blocks_14": "double_blocks.14.",
        "double_blocks_15": "double_blocks.15.",
        "double_blocks_16": "double_blocks.16.",
        "double_blocks_17": "double_blocks.17.",
        "double_blocks_18": "double_blocks.18.",
        "single_blocks_0": "single_blocks.0.",
        "single_blocks_1": "single_blocks.1.",
        "single_blocks_2": "single_blocks.2.",
        "single_blocks_3": "single_blocks.3.",
        "single_blocks_4": "single_blocks.4.",
        "single_blocks_5": "single_blocks.5.",
        "single_blocks_6": "single_blocks.6.",
        "single_blocks_7": "single_blocks.7.",
        "single_blocks_8": "single_blocks.8.",
        "single_blocks_9": "single_blocks.9.",
        "single_blocks_10": "single_blocks.10.",
        "single_blocks_11": "single_blocks.11.",
        "single_blocks_12": "single_blocks.12.",
        "single_blocks_13": "single_blocks.13.",
        "single_blocks_14": "single_blocks.14.",
        "single_blocks_15": "single_blocks.15.",
        "single_blocks_16": "single_blocks.16.",
        "single_blocks_17": "single_blocks.17.",
        "single_blocks_18": "single_blocks.18.",
        "single_blocks_19": "single_blocks.19.",
        "single_blocks_20": "single_blocks.20.",
        "single_blocks_21": "single_blocks.21.",
        "single_blocks_22": "single_blocks.22.",
        "single_blocks_23": "single_blocks.23.",
        "single_blocks_24": "single_blocks.24.",
        "single_blocks_25": "single_blocks.25.",
        "single_blocks_26": "single_blocks.26.",
        "single_blocks_27": "single_blocks.27.",
        "single_blocks_28": "single_blocks.28.",
        "single_blocks_29": "single_blocks.29.",
        "single_blocks_30": "single_blocks.30.",
        "single_blocks_31": "single_blocks.31.",
        "single_blocks_32": "single_blocks.32.",
        "single_blocks_33": "single_blocks.33.",
        "single_blocks_34": "single_blocks.34.",
        "single_blocks_35": "single_blocks.35.",
        "single_blocks_36": "single_blocks.36.",
        "single_blocks_37": "single_blocks.37.",
        "final_layer": "final_layer.",
    }

    def __init__(
        self,
        node_id: str,
        model1: tuple[str, int],
        model2: tuple[str, int],
        img_in: float | tuple[str, int] = 1.0,
        time_in: float | tuple[str, int] = 1.0,
        guidance_in: float | tuple[str, int] = 1.0,
        vector_in: float | tuple[str, int] = 1.0,
        txt_in: float | tuple[str, int] = 1.0,
        double_blocks_0: float | tuple[str, int] = 1.0,
        double_blocks_1: float | tuple[str, int] = 1.0,
        double_blocks_2: float | tuple[str, int] = 1.0,
        double_blocks_3: float | tuple[str, int] = 1.0,
        double_blocks_4: float | tuple[str, int] = 1.0,
        double_blocks_5: float | tuple[str, int] = 1.0,
        double_blocks_6: float | tuple[str, int] = 1.0,
        double_blocks_7: float | tuple[str, int] = 1.0,
        double_blocks_8: float | tuple[str, int] = 1.0,
        double_blocks_9: float | tuple[str, int] = 1.0,
        double_blocks_10: float | tuple[str, int] = 1.0,
        double_blocks_11: float | tuple[str, int] = 1.0,
        double_blocks_12: float | tuple[str, int] = 1.0,
        double_blocks_13: float | tuple[str, int] = 1.0,
        double_blocks_14: float | tuple[str, int] = 1.0,
        double_blocks_15: float | tuple[str, int] = 1.0,
        double_blocks_16: float | tuple[str, int] = 1.0,
        double_blocks_17: float | tuple[str, int] = 1.0,
        double_blocks_18: float | tuple[str, int] = 1.0,
        single_blocks_0: float | tuple[str, int] = 1.0,
        single_blocks_1: float | tuple[str, int] = 1.0,
        single_blocks_2: float | tuple[str, int] = 1.0,
        single_blocks_3: float | tuple[str, int] = 1.0,
        single_blocks_4: float | tuple[str, int] = 1.0,
        single_blocks_5: float | tuple[str, int] = 1.0,
        single_blocks_6: float | tuple[str, int] = 1.0,
        single_blocks_7: float | tuple[str, int] = 1.0,
        single_blocks_8: float | tuple[str, int] = 1.0,
        single_blocks_9: float | tuple[str, int] = 1.0,
        single_blocks_10: float | tuple[str, int] = 1.0,
        single_blocks_11: float | tuple[str, int] = 1.0,
        single_blocks_12: float | tuple[str, int] = 1.0,
        single_blocks_13: float | tuple[str, int] = 1.0,
        single_blocks_14: float | tuple[str, int] = 1.0,
        single_blocks_15: float | tuple[str, int] = 1.0,
        single_blocks_16: float | tuple[str, int] = 1.0,
        single_blocks_17: float | tuple[str, int] = 1.0,
        single_blocks_18: float | tuple[str, int] = 1.0,
        single_blocks_19: float | tuple[str, int] = 1.0,
        single_blocks_20: float | tuple[str, int] = 1.0,
        single_blocks_21: float | tuple[str, int] = 1.0,
        single_blocks_22: float | tuple[str, int] = 1.0,
        single_blocks_23: float | tuple[str, int] = 1.0,
        single_blocks_24: float | tuple[str, int] = 1.0,
        single_blocks_25: float | tuple[str, int] = 1.0,
        single_blocks_26: float | tuple[str, int] = 1.0,
        single_blocks_27: float | tuple[str, int] = 1.0,
        single_blocks_28: float | tuple[str, int] = 1.0,
        single_blocks_29: float | tuple[str, int] = 1.0,
        single_blocks_30: float | tuple[str, int] = 1.0,
        single_blocks_31: float | tuple[str, int] = 1.0,
        single_blocks_32: float | tuple[str, int] = 1.0,
        single_blocks_33: float | tuple[str, int] = 1.0,
        single_blocks_34: float | tuple[str, int] = 1.0,
        single_blocks_35: float | tuple[str, int] = 1.0,
        single_blocks_36: float | tuple[str, int] = 1.0,
        single_blocks_37: float | tuple[str, int] = 1.0,
        final_layer: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model1 = model1
        self.model2 = model2
        self.img_in = img_in
        self.time_in = time_in
        self.guidance_in = guidance_in
        self.vector_in = vector_in
        self.txt_in = txt_in
        self.double_blocks_0 = double_blocks_0
        self.double_blocks_1 = double_blocks_1
        self.double_blocks_2 = double_blocks_2
        self.double_blocks_3 = double_blocks_3
        self.double_blocks_4 = double_blocks_4
        self.double_blocks_5 = double_blocks_5
        self.double_blocks_6 = double_blocks_6
        self.double_blocks_7 = double_blocks_7
        self.double_blocks_8 = double_blocks_8
        self.double_blocks_9 = double_blocks_9
        self.double_blocks_10 = double_blocks_10
        self.double_blocks_11 = double_blocks_11
        self.double_blocks_12 = double_blocks_12
        self.double_blocks_13 = double_blocks_13
        self.double_blocks_14 = double_blocks_14
        self.double_blocks_15 = double_blocks_15
        self.double_blocks_16 = double_blocks_16
        self.double_blocks_17 = double_blocks_17
        self.double_blocks_18 = double_blocks_18
        self.single_blocks_0 = single_blocks_0
        self.single_blocks_1 = single_blocks_1
        self.single_blocks_2 = single_blocks_2
        self.single_blocks_3 = single_blocks_3
        self.single_blocks_4 = single_blocks_4
        self.single_blocks_5 = single_blocks_5
        self.single_blocks_6 = single_blocks_6
        self.single_blocks_7 = single_blocks_7
        self.single_blocks_8 = single_blocks_8
        self.single_blocks_9 = single_blocks_9
        self.single_blocks_10 = single_blocks_10
        self.single_blocks_11 = single_blocks_11
        self.single_blocks_12 = single_blocks_12
        self.single_blocks_13 = single_blocks_13
        self.single_blocks_14 = single_blocks_14
        self.single_blocks_15 = single_blocks_15
        self.single_blocks_16 = single_blocks_16
        self.single_blocks_17 = single_blocks_17
        self.single_blocks_18 = single_blocks_18
        self.single_blocks_19 = single_blocks_19
        self.single_blocks_20 = single_blocks_20
        self.single_blocks_21 = single_blocks_21
        self.single_blocks_22 = single_blocks_22
        self.single_blocks_23 = single_blocks_23
        self.single_blocks_24 = single_blocks_24
        self.single_blocks_25 = single_blocks_25
        self.single_blocks_26 = single_blocks_26
        self.single_blocks_27 = single_blocks_27
        self.single_blocks_28 = single_blocks_28
        self.single_blocks_29 = single_blocks_29
        self.single_blocks_30 = single_blocks_30
        self.single_blocks_31 = single_blocks_31
        self.single_blocks_32 = single_blocks_32
        self.single_blocks_33 = single_blocks_33
        self.single_blocks_34 = single_blocks_34
        self.single_blocks_35 = single_blocks_35
        self.single_blocks_36 = single_blocks_36
        self.single_blocks_37 = single_blocks_37
        self.final_layer = final_layer
        self.__validate()

    def __validate(self) -> None:
        validate_range("img_in", self.img_in, 0.0, 1.0)
        validate_range("time_in", self.time_in, 0.0, 1.0)
        validate_range("guidance_in", self.guidance_in, 0.0, 1.0)
        validate_range("vector_in", self.vector_in, 0.0, 1.0)
        validate_range("txt_in", self.txt_in, 0.0, 1.0)
        validate_range("double_blocks_0", self.double_blocks_0, 0.0, 1.0)
        validate_range("double_blocks_1", self.double_blocks_1, 0.0, 1.0)
        validate_range("double_blocks_2", self.double_blocks_2, 0.0, 1.0)
        validate_range("double_blocks_3", self.double_blocks_3, 0.0, 1.0)
        validate_range("double_blocks_4", self.double_blocks_4, 0.0, 1.0)
        validate_range("double_blocks_5", self.double_blocks_5, 0.0, 1.0)
        validate_range("double_blocks_6", self.double_blocks_6, 0.0, 1.0)
        validate_range("double_blocks_7", self.double_blocks_7, 0.0, 1.0)
        validate_range("double_blocks_8", self.double_blocks_8, 0.0, 1.0)
        validate_range("double_blocks_9", self.double_blocks_9, 0.0, 1.0)
        validate_range("double_blocks_10", self.double_blocks_10, 0.0, 1.0)
        validate_range("double_blocks_11", self.double_blocks_11, 0.0, 1.0)
        validate_range("double_blocks_12", self.double_blocks_12, 0.0, 1.0)
        validate_range("double_blocks_13", self.double_blocks_13, 0.0, 1.0)
        validate_range("double_blocks_14", self.double_blocks_14, 0.0, 1.0)
        validate_range("double_blocks_15", self.double_blocks_15, 0.0, 1.0)
        validate_range("double_blocks_16", self.double_blocks_16, 0.0, 1.0)
        validate_range("double_blocks_17", self.double_blocks_17, 0.0, 1.0)
        validate_range("double_blocks_18", self.double_blocks_18, 0.0, 1.0)
        validate_range("single_blocks_0", self.single_blocks_0, 0.0, 1.0)
        validate_range("single_blocks_1", self.single_blocks_1, 0.0, 1.0)
        validate_range("single_blocks_2", self.single_blocks_2, 0.0, 1.0)
        validate_range("single_blocks_3", self.single_blocks_3, 0.0, 1.0)
        validate_range("single_blocks_4", self.single_blocks_4, 0.0, 1.0)
        validate_range("single_blocks_5", self.single_blocks_5, 0.0, 1.0)
        validate_range("single_blocks_6", self.single_blocks_6, 0.0, 1.0)
        validate_range("single_blocks_7", self.single_blocks_7, 0.0, 1.0)
        validate_range("single_blocks_8", self.single_blocks_8, 0.0, 1.0)
        validate_range("single_blocks_9", self.single_blocks_9, 0.0, 1.0)
        validate_range("single_blocks_10", self.single_blocks_10, 0.0, 1.0)
        validate_range("single_blocks_11", self.single_blocks_11, 0.0, 1.0)
        validate_range("single_blocks_12", self.single_blocks_12, 0.0, 1.0)
        validate_range("single_blocks_13", self.single_blocks_13, 0.0, 1.0)
        validate_range("single_blocks_14", self.single_blocks_14, 0.0, 1.0)
        validate_range("single_blocks_15", self.single_blocks_15, 0.0, 1.0)
        validate_range("single_blocks_16", self.single_blocks_16, 0.0, 1.0)
        validate_range("single_blocks_17", self.single_blocks_17, 0.0, 1.0)
        validate_range("single_blocks_18", self.single_blocks_18, 0.0, 1.0)
        validate_range("single_blocks_19", self.single_blocks_19, 0.0, 1.0)
        validate_range("single_blocks_20", self.single_blocks_20, 0.0, 1.0)
        validate_range("single_blocks_21", self.single_blocks_21, 0.0, 1.0)
        validate_range("single_blocks_22", self.single_blocks_22, 0.0, 1.0)
        validate_range("single_blocks_23", self.single_blocks_23, 0.0, 1.0)
        validate_range("single_blocks_24", self.single_blocks_24, 0.0, 1.0)
        validate_range("single_blocks_25", self.single_blocks_25, 0.0, 1.0)
        validate_range("single_blocks_26", self.single_blocks_26, 0.0, 1.0)
        validate_range("single_blocks_27", self.single_blocks_27, 0.0, 1.0)
        validate_range("single_blocks_28", self.single_blocks_28, 0.0, 1.0)
        validate_range("single_blocks_29", self.single_blocks_29, 0.0, 1.0)
        validate_range("single_blocks_30", self.single_blocks_30, 0.0, 1.0)
        validate_range("single_blocks_31", self.single_blocks_31, 0.0, 1.0)
        validate_range("single_blocks_32", self.single_blocks_32, 0.0, 1.0)
        validate_range("single_blocks_33", self.single_blocks_33, 0.0, 1.0)
        validate_range("single_blocks_34", self.single_blocks_34, 0.0, 1.0)
        validate_range("single_blocks_35", self.single_blocks_35, 0.0, 1.0)
        validate_range("single_blocks_36", self.single_blocks_36, 0.0, 1.0)
        validate_range("single_blocks_37", self.single_blocks_37, 0.0, 1.0)
        validate_range("final_layer", self.final_layer, 0.0, 1.0)


class ModelMergeSD35_Large(BaseNode):
    __slots__ = (
        "model1",
        "model2",
        "pos_embed",
        "x_embedder",
        "context_embedder",
        "y_embedder",
        "t_embedder",
        "joint_blocks_0",
        "joint_blocks_1",
        "joint_blocks_2",
        "joint_blocks_3",
        "joint_blocks_4",
        "joint_blocks_5",
        "joint_blocks_6",
        "joint_blocks_7",
        "joint_blocks_8",
        "joint_blocks_9",
        "joint_blocks_10",
        "joint_blocks_11",
        "joint_blocks_12",
        "joint_blocks_13",
        "joint_blocks_14",
        "joint_blocks_15",
        "joint_blocks_16",
        "joint_blocks_17",
        "joint_blocks_18",
        "joint_blocks_19",
        "joint_blocks_20",
        "joint_blocks_21",
        "joint_blocks_22",
        "joint_blocks_23",
        "joint_blocks_24",
        "joint_blocks_25",
        "joint_blocks_26",
        "joint_blocks_27",
        "joint_blocks_28",
        "joint_blocks_29",
        "joint_blocks_30",
        "joint_blocks_31",
        "joint_blocks_32",
        "joint_blocks_33",
        "joint_blocks_34",
        "joint_blocks_35",
        "joint_blocks_36",
        "joint_blocks_37",
        "final_layer",
    )
    CATEGORY = "model/merging/model specific"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {
        "model1": "MODEL",
        "model2": "MODEL",
        "pos_embed": "FLOAT",
        "x_embedder": "FLOAT",
        "context_embedder": "FLOAT",
        "y_embedder": "FLOAT",
        "t_embedder": "FLOAT",
        "joint_blocks_0": "FLOAT",
        "joint_blocks_1": "FLOAT",
        "joint_blocks_2": "FLOAT",
        "joint_blocks_3": "FLOAT",
        "joint_blocks_4": "FLOAT",
        "joint_blocks_5": "FLOAT",
        "joint_blocks_6": "FLOAT",
        "joint_blocks_7": "FLOAT",
        "joint_blocks_8": "FLOAT",
        "joint_blocks_9": "FLOAT",
        "joint_blocks_10": "FLOAT",
        "joint_blocks_11": "FLOAT",
        "joint_blocks_12": "FLOAT",
        "joint_blocks_13": "FLOAT",
        "joint_blocks_14": "FLOAT",
        "joint_blocks_15": "FLOAT",
        "joint_blocks_16": "FLOAT",
        "joint_blocks_17": "FLOAT",
        "joint_blocks_18": "FLOAT",
        "joint_blocks_19": "FLOAT",
        "joint_blocks_20": "FLOAT",
        "joint_blocks_21": "FLOAT",
        "joint_blocks_22": "FLOAT",
        "joint_blocks_23": "FLOAT",
        "joint_blocks_24": "FLOAT",
        "joint_blocks_25": "FLOAT",
        "joint_blocks_26": "FLOAT",
        "joint_blocks_27": "FLOAT",
        "joint_blocks_28": "FLOAT",
        "joint_blocks_29": "FLOAT",
        "joint_blocks_30": "FLOAT",
        "joint_blocks_31": "FLOAT",
        "joint_blocks_32": "FLOAT",
        "joint_blocks_33": "FLOAT",
        "joint_blocks_34": "FLOAT",
        "joint_blocks_35": "FLOAT",
        "joint_blocks_36": "FLOAT",
        "joint_blocks_37": "FLOAT",
        "final_layer": "FLOAT",
    }
    INPUT_NAMES = {
        "pos_embed": "pos_embed.",
        "x_embedder": "x_embedder.",
        "context_embedder": "context_embedder.",
        "y_embedder": "y_embedder.",
        "t_embedder": "t_embedder.",
        "joint_blocks_0": "joint_blocks.0.",
        "joint_blocks_1": "joint_blocks.1.",
        "joint_blocks_2": "joint_blocks.2.",
        "joint_blocks_3": "joint_blocks.3.",
        "joint_blocks_4": "joint_blocks.4.",
        "joint_blocks_5": "joint_blocks.5.",
        "joint_blocks_6": "joint_blocks.6.",
        "joint_blocks_7": "joint_blocks.7.",
        "joint_blocks_8": "joint_blocks.8.",
        "joint_blocks_9": "joint_blocks.9.",
        "joint_blocks_10": "joint_blocks.10.",
        "joint_blocks_11": "joint_blocks.11.",
        "joint_blocks_12": "joint_blocks.12.",
        "joint_blocks_13": "joint_blocks.13.",
        "joint_blocks_14": "joint_blocks.14.",
        "joint_blocks_15": "joint_blocks.15.",
        "joint_blocks_16": "joint_blocks.16.",
        "joint_blocks_17": "joint_blocks.17.",
        "joint_blocks_18": "joint_blocks.18.",
        "joint_blocks_19": "joint_blocks.19.",
        "joint_blocks_20": "joint_blocks.20.",
        "joint_blocks_21": "joint_blocks.21.",
        "joint_blocks_22": "joint_blocks.22.",
        "joint_blocks_23": "joint_blocks.23.",
        "joint_blocks_24": "joint_blocks.24.",
        "joint_blocks_25": "joint_blocks.25.",
        "joint_blocks_26": "joint_blocks.26.",
        "joint_blocks_27": "joint_blocks.27.",
        "joint_blocks_28": "joint_blocks.28.",
        "joint_blocks_29": "joint_blocks.29.",
        "joint_blocks_30": "joint_blocks.30.",
        "joint_blocks_31": "joint_blocks.31.",
        "joint_blocks_32": "joint_blocks.32.",
        "joint_blocks_33": "joint_blocks.33.",
        "joint_blocks_34": "joint_blocks.34.",
        "joint_blocks_35": "joint_blocks.35.",
        "joint_blocks_36": "joint_blocks.36.",
        "joint_blocks_37": "joint_blocks.37.",
        "final_layer": "final_layer.",
    }

    def __init__(
        self,
        node_id: str,
        model1: tuple[str, int],
        model2: tuple[str, int],
        pos_embed: float | tuple[str, int] = 1.0,
        x_embedder: float | tuple[str, int] = 1.0,
        context_embedder: float | tuple[str, int] = 1.0,
        y_embedder: float | tuple[str, int] = 1.0,
        t_embedder: float | tuple[str, int] = 1.0,
        joint_blocks_0: float | tuple[str, int] = 1.0,
        joint_blocks_1: float | tuple[str, int] = 1.0,
        joint_blocks_2: float | tuple[str, int] = 1.0,
        joint_blocks_3: float | tuple[str, int] = 1.0,
        joint_blocks_4: float | tuple[str, int] = 1.0,
        joint_blocks_5: float | tuple[str, int] = 1.0,
        joint_blocks_6: float | tuple[str, int] = 1.0,
        joint_blocks_7: float | tuple[str, int] = 1.0,
        joint_blocks_8: float | tuple[str, int] = 1.0,
        joint_blocks_9: float | tuple[str, int] = 1.0,
        joint_blocks_10: float | tuple[str, int] = 1.0,
        joint_blocks_11: float | tuple[str, int] = 1.0,
        joint_blocks_12: float | tuple[str, int] = 1.0,
        joint_blocks_13: float | tuple[str, int] = 1.0,
        joint_blocks_14: float | tuple[str, int] = 1.0,
        joint_blocks_15: float | tuple[str, int] = 1.0,
        joint_blocks_16: float | tuple[str, int] = 1.0,
        joint_blocks_17: float | tuple[str, int] = 1.0,
        joint_blocks_18: float | tuple[str, int] = 1.0,
        joint_blocks_19: float | tuple[str, int] = 1.0,
        joint_blocks_20: float | tuple[str, int] = 1.0,
        joint_blocks_21: float | tuple[str, int] = 1.0,
        joint_blocks_22: float | tuple[str, int] = 1.0,
        joint_blocks_23: float | tuple[str, int] = 1.0,
        joint_blocks_24: float | tuple[str, int] = 1.0,
        joint_blocks_25: float | tuple[str, int] = 1.0,
        joint_blocks_26: float | tuple[str, int] = 1.0,
        joint_blocks_27: float | tuple[str, int] = 1.0,
        joint_blocks_28: float | tuple[str, int] = 1.0,
        joint_blocks_29: float | tuple[str, int] = 1.0,
        joint_blocks_30: float | tuple[str, int] = 1.0,
        joint_blocks_31: float | tuple[str, int] = 1.0,
        joint_blocks_32: float | tuple[str, int] = 1.0,
        joint_blocks_33: float | tuple[str, int] = 1.0,
        joint_blocks_34: float | tuple[str, int] = 1.0,
        joint_blocks_35: float | tuple[str, int] = 1.0,
        joint_blocks_36: float | tuple[str, int] = 1.0,
        joint_blocks_37: float | tuple[str, int] = 1.0,
        final_layer: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model1 = model1
        self.model2 = model2
        self.pos_embed = pos_embed
        self.x_embedder = x_embedder
        self.context_embedder = context_embedder
        self.y_embedder = y_embedder
        self.t_embedder = t_embedder
        self.joint_blocks_0 = joint_blocks_0
        self.joint_blocks_1 = joint_blocks_1
        self.joint_blocks_2 = joint_blocks_2
        self.joint_blocks_3 = joint_blocks_3
        self.joint_blocks_4 = joint_blocks_4
        self.joint_blocks_5 = joint_blocks_5
        self.joint_blocks_6 = joint_blocks_6
        self.joint_blocks_7 = joint_blocks_7
        self.joint_blocks_8 = joint_blocks_8
        self.joint_blocks_9 = joint_blocks_9
        self.joint_blocks_10 = joint_blocks_10
        self.joint_blocks_11 = joint_blocks_11
        self.joint_blocks_12 = joint_blocks_12
        self.joint_blocks_13 = joint_blocks_13
        self.joint_blocks_14 = joint_blocks_14
        self.joint_blocks_15 = joint_blocks_15
        self.joint_blocks_16 = joint_blocks_16
        self.joint_blocks_17 = joint_blocks_17
        self.joint_blocks_18 = joint_blocks_18
        self.joint_blocks_19 = joint_blocks_19
        self.joint_blocks_20 = joint_blocks_20
        self.joint_blocks_21 = joint_blocks_21
        self.joint_blocks_22 = joint_blocks_22
        self.joint_blocks_23 = joint_blocks_23
        self.joint_blocks_24 = joint_blocks_24
        self.joint_blocks_25 = joint_blocks_25
        self.joint_blocks_26 = joint_blocks_26
        self.joint_blocks_27 = joint_blocks_27
        self.joint_blocks_28 = joint_blocks_28
        self.joint_blocks_29 = joint_blocks_29
        self.joint_blocks_30 = joint_blocks_30
        self.joint_blocks_31 = joint_blocks_31
        self.joint_blocks_32 = joint_blocks_32
        self.joint_blocks_33 = joint_blocks_33
        self.joint_blocks_34 = joint_blocks_34
        self.joint_blocks_35 = joint_blocks_35
        self.joint_blocks_36 = joint_blocks_36
        self.joint_blocks_37 = joint_blocks_37
        self.final_layer = final_layer
        self.__validate()

    def __validate(self) -> None:
        validate_range("pos_embed", self.pos_embed, 0.0, 1.0)
        validate_range("x_embedder", self.x_embedder, 0.0, 1.0)
        validate_range("context_embedder", self.context_embedder, 0.0, 1.0)
        validate_range("y_embedder", self.y_embedder, 0.0, 1.0)
        validate_range("t_embedder", self.t_embedder, 0.0, 1.0)
        validate_range("joint_blocks_0", self.joint_blocks_0, 0.0, 1.0)
        validate_range("joint_blocks_1", self.joint_blocks_1, 0.0, 1.0)
        validate_range("joint_blocks_2", self.joint_blocks_2, 0.0, 1.0)
        validate_range("joint_blocks_3", self.joint_blocks_3, 0.0, 1.0)
        validate_range("joint_blocks_4", self.joint_blocks_4, 0.0, 1.0)
        validate_range("joint_blocks_5", self.joint_blocks_5, 0.0, 1.0)
        validate_range("joint_blocks_6", self.joint_blocks_6, 0.0, 1.0)
        validate_range("joint_blocks_7", self.joint_blocks_7, 0.0, 1.0)
        validate_range("joint_blocks_8", self.joint_blocks_8, 0.0, 1.0)
        validate_range("joint_blocks_9", self.joint_blocks_9, 0.0, 1.0)
        validate_range("joint_blocks_10", self.joint_blocks_10, 0.0, 1.0)
        validate_range("joint_blocks_11", self.joint_blocks_11, 0.0, 1.0)
        validate_range("joint_blocks_12", self.joint_blocks_12, 0.0, 1.0)
        validate_range("joint_blocks_13", self.joint_blocks_13, 0.0, 1.0)
        validate_range("joint_blocks_14", self.joint_blocks_14, 0.0, 1.0)
        validate_range("joint_blocks_15", self.joint_blocks_15, 0.0, 1.0)
        validate_range("joint_blocks_16", self.joint_blocks_16, 0.0, 1.0)
        validate_range("joint_blocks_17", self.joint_blocks_17, 0.0, 1.0)
        validate_range("joint_blocks_18", self.joint_blocks_18, 0.0, 1.0)
        validate_range("joint_blocks_19", self.joint_blocks_19, 0.0, 1.0)
        validate_range("joint_blocks_20", self.joint_blocks_20, 0.0, 1.0)
        validate_range("joint_blocks_21", self.joint_blocks_21, 0.0, 1.0)
        validate_range("joint_blocks_22", self.joint_blocks_22, 0.0, 1.0)
        validate_range("joint_blocks_23", self.joint_blocks_23, 0.0, 1.0)
        validate_range("joint_blocks_24", self.joint_blocks_24, 0.0, 1.0)
        validate_range("joint_blocks_25", self.joint_blocks_25, 0.0, 1.0)
        validate_range("joint_blocks_26", self.joint_blocks_26, 0.0, 1.0)
        validate_range("joint_blocks_27", self.joint_blocks_27, 0.0, 1.0)
        validate_range("joint_blocks_28", self.joint_blocks_28, 0.0, 1.0)
        validate_range("joint_blocks_29", self.joint_blocks_29, 0.0, 1.0)
        validate_range("joint_blocks_30", self.joint_blocks_30, 0.0, 1.0)
        validate_range("joint_blocks_31", self.joint_blocks_31, 0.0, 1.0)
        validate_range("joint_blocks_32", self.joint_blocks_32, 0.0, 1.0)
        validate_range("joint_blocks_33", self.joint_blocks_33, 0.0, 1.0)
        validate_range("joint_blocks_34", self.joint_blocks_34, 0.0, 1.0)
        validate_range("joint_blocks_35", self.joint_blocks_35, 0.0, 1.0)
        validate_range("joint_blocks_36", self.joint_blocks_36, 0.0, 1.0)
        validate_range("joint_blocks_37", self.joint_blocks_37, 0.0, 1.0)
        validate_range("final_layer", self.final_layer, 0.0, 1.0)


class ModelMergeMochiPreview(BaseNode):
    __slots__ = (
        "model1",
        "model2",
        "pos_frequencies",
        "t_embedder",
        "t5_y_embedder",
        "t5_yproj",
        "blocks_0",
        "blocks_1",
        "blocks_2",
        "blocks_3",
        "blocks_4",
        "blocks_5",
        "blocks_6",
        "blocks_7",
        "blocks_8",
        "blocks_9",
        "blocks_10",
        "blocks_11",
        "blocks_12",
        "blocks_13",
        "blocks_14",
        "blocks_15",
        "blocks_16",
        "blocks_17",
        "blocks_18",
        "blocks_19",
        "blocks_20",
        "blocks_21",
        "blocks_22",
        "blocks_23",
        "blocks_24",
        "blocks_25",
        "blocks_26",
        "blocks_27",
        "blocks_28",
        "blocks_29",
        "blocks_30",
        "blocks_31",
        "blocks_32",
        "blocks_33",
        "blocks_34",
        "blocks_35",
        "blocks_36",
        "blocks_37",
        "blocks_38",
        "blocks_39",
        "blocks_40",
        "blocks_41",
        "blocks_42",
        "blocks_43",
        "blocks_44",
        "blocks_45",
        "blocks_46",
        "blocks_47",
        "final_layer",
    )
    CATEGORY = "model/merging/model specific"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {
        "model1": "MODEL",
        "model2": "MODEL",
        "pos_frequencies": "FLOAT",
        "t_embedder": "FLOAT",
        "t5_y_embedder": "FLOAT",
        "t5_yproj": "FLOAT",
        "blocks_0": "FLOAT",
        "blocks_1": "FLOAT",
        "blocks_2": "FLOAT",
        "blocks_3": "FLOAT",
        "blocks_4": "FLOAT",
        "blocks_5": "FLOAT",
        "blocks_6": "FLOAT",
        "blocks_7": "FLOAT",
        "blocks_8": "FLOAT",
        "blocks_9": "FLOAT",
        "blocks_10": "FLOAT",
        "blocks_11": "FLOAT",
        "blocks_12": "FLOAT",
        "blocks_13": "FLOAT",
        "blocks_14": "FLOAT",
        "blocks_15": "FLOAT",
        "blocks_16": "FLOAT",
        "blocks_17": "FLOAT",
        "blocks_18": "FLOAT",
        "blocks_19": "FLOAT",
        "blocks_20": "FLOAT",
        "blocks_21": "FLOAT",
        "blocks_22": "FLOAT",
        "blocks_23": "FLOAT",
        "blocks_24": "FLOAT",
        "blocks_25": "FLOAT",
        "blocks_26": "FLOAT",
        "blocks_27": "FLOAT",
        "blocks_28": "FLOAT",
        "blocks_29": "FLOAT",
        "blocks_30": "FLOAT",
        "blocks_31": "FLOAT",
        "blocks_32": "FLOAT",
        "blocks_33": "FLOAT",
        "blocks_34": "FLOAT",
        "blocks_35": "FLOAT",
        "blocks_36": "FLOAT",
        "blocks_37": "FLOAT",
        "blocks_38": "FLOAT",
        "blocks_39": "FLOAT",
        "blocks_40": "FLOAT",
        "blocks_41": "FLOAT",
        "blocks_42": "FLOAT",
        "blocks_43": "FLOAT",
        "blocks_44": "FLOAT",
        "blocks_45": "FLOAT",
        "blocks_46": "FLOAT",
        "blocks_47": "FLOAT",
        "final_layer": "FLOAT",
    }
    INPUT_NAMES = {
        "pos_frequencies": "pos_frequencies.",
        "t_embedder": "t_embedder.",
        "t5_y_embedder": "t5_y_embedder.",
        "t5_yproj": "t5_yproj.",
        "blocks_0": "blocks.0.",
        "blocks_1": "blocks.1.",
        "blocks_2": "blocks.2.",
        "blocks_3": "blocks.3.",
        "blocks_4": "blocks.4.",
        "blocks_5": "blocks.5.",
        "blocks_6": "blocks.6.",
        "blocks_7": "blocks.7.",
        "blocks_8": "blocks.8.",
        "blocks_9": "blocks.9.",
        "blocks_10": "blocks.10.",
        "blocks_11": "blocks.11.",
        "blocks_12": "blocks.12.",
        "blocks_13": "blocks.13.",
        "blocks_14": "blocks.14.",
        "blocks_15": "blocks.15.",
        "blocks_16": "blocks.16.",
        "blocks_17": "blocks.17.",
        "blocks_18": "blocks.18.",
        "blocks_19": "blocks.19.",
        "blocks_20": "blocks.20.",
        "blocks_21": "blocks.21.",
        "blocks_22": "blocks.22.",
        "blocks_23": "blocks.23.",
        "blocks_24": "blocks.24.",
        "blocks_25": "blocks.25.",
        "blocks_26": "blocks.26.",
        "blocks_27": "blocks.27.",
        "blocks_28": "blocks.28.",
        "blocks_29": "blocks.29.",
        "blocks_30": "blocks.30.",
        "blocks_31": "blocks.31.",
        "blocks_32": "blocks.32.",
        "blocks_33": "blocks.33.",
        "blocks_34": "blocks.34.",
        "blocks_35": "blocks.35.",
        "blocks_36": "blocks.36.",
        "blocks_37": "blocks.37.",
        "blocks_38": "blocks.38.",
        "blocks_39": "blocks.39.",
        "blocks_40": "blocks.40.",
        "blocks_41": "blocks.41.",
        "blocks_42": "blocks.42.",
        "blocks_43": "blocks.43.",
        "blocks_44": "blocks.44.",
        "blocks_45": "blocks.45.",
        "blocks_46": "blocks.46.",
        "blocks_47": "blocks.47.",
        "final_layer": "final_layer.",
    }

    def __init__(
        self,
        node_id: str,
        model1: tuple[str, int],
        model2: tuple[str, int],
        pos_frequencies: float | tuple[str, int] = 1.0,
        t_embedder: float | tuple[str, int] = 1.0,
        t5_y_embedder: float | tuple[str, int] = 1.0,
        t5_yproj: float | tuple[str, int] = 1.0,
        blocks_0: float | tuple[str, int] = 1.0,
        blocks_1: float | tuple[str, int] = 1.0,
        blocks_2: float | tuple[str, int] = 1.0,
        blocks_3: float | tuple[str, int] = 1.0,
        blocks_4: float | tuple[str, int] = 1.0,
        blocks_5: float | tuple[str, int] = 1.0,
        blocks_6: float | tuple[str, int] = 1.0,
        blocks_7: float | tuple[str, int] = 1.0,
        blocks_8: float | tuple[str, int] = 1.0,
        blocks_9: float | tuple[str, int] = 1.0,
        blocks_10: float | tuple[str, int] = 1.0,
        blocks_11: float | tuple[str, int] = 1.0,
        blocks_12: float | tuple[str, int] = 1.0,
        blocks_13: float | tuple[str, int] = 1.0,
        blocks_14: float | tuple[str, int] = 1.0,
        blocks_15: float | tuple[str, int] = 1.0,
        blocks_16: float | tuple[str, int] = 1.0,
        blocks_17: float | tuple[str, int] = 1.0,
        blocks_18: float | tuple[str, int] = 1.0,
        blocks_19: float | tuple[str, int] = 1.0,
        blocks_20: float | tuple[str, int] = 1.0,
        blocks_21: float | tuple[str, int] = 1.0,
        blocks_22: float | tuple[str, int] = 1.0,
        blocks_23: float | tuple[str, int] = 1.0,
        blocks_24: float | tuple[str, int] = 1.0,
        blocks_25: float | tuple[str, int] = 1.0,
        blocks_26: float | tuple[str, int] = 1.0,
        blocks_27: float | tuple[str, int] = 1.0,
        blocks_28: float | tuple[str, int] = 1.0,
        blocks_29: float | tuple[str, int] = 1.0,
        blocks_30: float | tuple[str, int] = 1.0,
        blocks_31: float | tuple[str, int] = 1.0,
        blocks_32: float | tuple[str, int] = 1.0,
        blocks_33: float | tuple[str, int] = 1.0,
        blocks_34: float | tuple[str, int] = 1.0,
        blocks_35: float | tuple[str, int] = 1.0,
        blocks_36: float | tuple[str, int] = 1.0,
        blocks_37: float | tuple[str, int] = 1.0,
        blocks_38: float | tuple[str, int] = 1.0,
        blocks_39: float | tuple[str, int] = 1.0,
        blocks_40: float | tuple[str, int] = 1.0,
        blocks_41: float | tuple[str, int] = 1.0,
        blocks_42: float | tuple[str, int] = 1.0,
        blocks_43: float | tuple[str, int] = 1.0,
        blocks_44: float | tuple[str, int] = 1.0,
        blocks_45: float | tuple[str, int] = 1.0,
        blocks_46: float | tuple[str, int] = 1.0,
        blocks_47: float | tuple[str, int] = 1.0,
        final_layer: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model1 = model1
        self.model2 = model2
        self.pos_frequencies = pos_frequencies
        self.t_embedder = t_embedder
        self.t5_y_embedder = t5_y_embedder
        self.t5_yproj = t5_yproj
        self.blocks_0 = blocks_0
        self.blocks_1 = blocks_1
        self.blocks_2 = blocks_2
        self.blocks_3 = blocks_3
        self.blocks_4 = blocks_4
        self.blocks_5 = blocks_5
        self.blocks_6 = blocks_6
        self.blocks_7 = blocks_7
        self.blocks_8 = blocks_8
        self.blocks_9 = blocks_9
        self.blocks_10 = blocks_10
        self.blocks_11 = blocks_11
        self.blocks_12 = blocks_12
        self.blocks_13 = blocks_13
        self.blocks_14 = blocks_14
        self.blocks_15 = blocks_15
        self.blocks_16 = blocks_16
        self.blocks_17 = blocks_17
        self.blocks_18 = blocks_18
        self.blocks_19 = blocks_19
        self.blocks_20 = blocks_20
        self.blocks_21 = blocks_21
        self.blocks_22 = blocks_22
        self.blocks_23 = blocks_23
        self.blocks_24 = blocks_24
        self.blocks_25 = blocks_25
        self.blocks_26 = blocks_26
        self.blocks_27 = blocks_27
        self.blocks_28 = blocks_28
        self.blocks_29 = blocks_29
        self.blocks_30 = blocks_30
        self.blocks_31 = blocks_31
        self.blocks_32 = blocks_32
        self.blocks_33 = blocks_33
        self.blocks_34 = blocks_34
        self.blocks_35 = blocks_35
        self.blocks_36 = blocks_36
        self.blocks_37 = blocks_37
        self.blocks_38 = blocks_38
        self.blocks_39 = blocks_39
        self.blocks_40 = blocks_40
        self.blocks_41 = blocks_41
        self.blocks_42 = blocks_42
        self.blocks_43 = blocks_43
        self.blocks_44 = blocks_44
        self.blocks_45 = blocks_45
        self.blocks_46 = blocks_46
        self.blocks_47 = blocks_47
        self.final_layer = final_layer
        self.__validate()

    def __validate(self) -> None:
        validate_range("pos_frequencies", self.pos_frequencies, 0.0, 1.0)
        validate_range("t_embedder", self.t_embedder, 0.0, 1.0)
        validate_range("t5_y_embedder", self.t5_y_embedder, 0.0, 1.0)
        validate_range("t5_yproj", self.t5_yproj, 0.0, 1.0)
        validate_range("blocks_0", self.blocks_0, 0.0, 1.0)
        validate_range("blocks_1", self.blocks_1, 0.0, 1.0)
        validate_range("blocks_2", self.blocks_2, 0.0, 1.0)
        validate_range("blocks_3", self.blocks_3, 0.0, 1.0)
        validate_range("blocks_4", self.blocks_4, 0.0, 1.0)
        validate_range("blocks_5", self.blocks_5, 0.0, 1.0)
        validate_range("blocks_6", self.blocks_6, 0.0, 1.0)
        validate_range("blocks_7", self.blocks_7, 0.0, 1.0)
        validate_range("blocks_8", self.blocks_8, 0.0, 1.0)
        validate_range("blocks_9", self.blocks_9, 0.0, 1.0)
        validate_range("blocks_10", self.blocks_10, 0.0, 1.0)
        validate_range("blocks_11", self.blocks_11, 0.0, 1.0)
        validate_range("blocks_12", self.blocks_12, 0.0, 1.0)
        validate_range("blocks_13", self.blocks_13, 0.0, 1.0)
        validate_range("blocks_14", self.blocks_14, 0.0, 1.0)
        validate_range("blocks_15", self.blocks_15, 0.0, 1.0)
        validate_range("blocks_16", self.blocks_16, 0.0, 1.0)
        validate_range("blocks_17", self.blocks_17, 0.0, 1.0)
        validate_range("blocks_18", self.blocks_18, 0.0, 1.0)
        validate_range("blocks_19", self.blocks_19, 0.0, 1.0)
        validate_range("blocks_20", self.blocks_20, 0.0, 1.0)
        validate_range("blocks_21", self.blocks_21, 0.0, 1.0)
        validate_range("blocks_22", self.blocks_22, 0.0, 1.0)
        validate_range("blocks_23", self.blocks_23, 0.0, 1.0)
        validate_range("blocks_24", self.blocks_24, 0.0, 1.0)
        validate_range("blocks_25", self.blocks_25, 0.0, 1.0)
        validate_range("blocks_26", self.blocks_26, 0.0, 1.0)
        validate_range("blocks_27", self.blocks_27, 0.0, 1.0)
        validate_range("blocks_28", self.blocks_28, 0.0, 1.0)
        validate_range("blocks_29", self.blocks_29, 0.0, 1.0)
        validate_range("blocks_30", self.blocks_30, 0.0, 1.0)
        validate_range("blocks_31", self.blocks_31, 0.0, 1.0)
        validate_range("blocks_32", self.blocks_32, 0.0, 1.0)
        validate_range("blocks_33", self.blocks_33, 0.0, 1.0)
        validate_range("blocks_34", self.blocks_34, 0.0, 1.0)
        validate_range("blocks_35", self.blocks_35, 0.0, 1.0)
        validate_range("blocks_36", self.blocks_36, 0.0, 1.0)
        validate_range("blocks_37", self.blocks_37, 0.0, 1.0)
        validate_range("blocks_38", self.blocks_38, 0.0, 1.0)
        validate_range("blocks_39", self.blocks_39, 0.0, 1.0)
        validate_range("blocks_40", self.blocks_40, 0.0, 1.0)
        validate_range("blocks_41", self.blocks_41, 0.0, 1.0)
        validate_range("blocks_42", self.blocks_42, 0.0, 1.0)
        validate_range("blocks_43", self.blocks_43, 0.0, 1.0)
        validate_range("blocks_44", self.blocks_44, 0.0, 1.0)
        validate_range("blocks_45", self.blocks_45, 0.0, 1.0)
        validate_range("blocks_46", self.blocks_46, 0.0, 1.0)
        validate_range("blocks_47", self.blocks_47, 0.0, 1.0)
        validate_range("final_layer", self.final_layer, 0.0, 1.0)


class ModelMergeLTXV(BaseNode):
    __slots__ = (
        "model1",
        "model2",
        "patchify_proj",
        "adaln_single",
        "caption_projection",
        "transformer_blocks_0",
        "transformer_blocks_1",
        "transformer_blocks_2",
        "transformer_blocks_3",
        "transformer_blocks_4",
        "transformer_blocks_5",
        "transformer_blocks_6",
        "transformer_blocks_7",
        "transformer_blocks_8",
        "transformer_blocks_9",
        "transformer_blocks_10",
        "transformer_blocks_11",
        "transformer_blocks_12",
        "transformer_blocks_13",
        "transformer_blocks_14",
        "transformer_blocks_15",
        "transformer_blocks_16",
        "transformer_blocks_17",
        "transformer_blocks_18",
        "transformer_blocks_19",
        "transformer_blocks_20",
        "transformer_blocks_21",
        "transformer_blocks_22",
        "transformer_blocks_23",
        "transformer_blocks_24",
        "transformer_blocks_25",
        "transformer_blocks_26",
        "transformer_blocks_27",
        "scale_shift_table",
        "proj_out",
    )
    CATEGORY = "model/merging/model specific"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {
        "model1": "MODEL",
        "model2": "MODEL",
        "patchify_proj": "FLOAT",
        "adaln_single": "FLOAT",
        "caption_projection": "FLOAT",
        "transformer_blocks_0": "FLOAT",
        "transformer_blocks_1": "FLOAT",
        "transformer_blocks_2": "FLOAT",
        "transformer_blocks_3": "FLOAT",
        "transformer_blocks_4": "FLOAT",
        "transformer_blocks_5": "FLOAT",
        "transformer_blocks_6": "FLOAT",
        "transformer_blocks_7": "FLOAT",
        "transformer_blocks_8": "FLOAT",
        "transformer_blocks_9": "FLOAT",
        "transformer_blocks_10": "FLOAT",
        "transformer_blocks_11": "FLOAT",
        "transformer_blocks_12": "FLOAT",
        "transformer_blocks_13": "FLOAT",
        "transformer_blocks_14": "FLOAT",
        "transformer_blocks_15": "FLOAT",
        "transformer_blocks_16": "FLOAT",
        "transformer_blocks_17": "FLOAT",
        "transformer_blocks_18": "FLOAT",
        "transformer_blocks_19": "FLOAT",
        "transformer_blocks_20": "FLOAT",
        "transformer_blocks_21": "FLOAT",
        "transformer_blocks_22": "FLOAT",
        "transformer_blocks_23": "FLOAT",
        "transformer_blocks_24": "FLOAT",
        "transformer_blocks_25": "FLOAT",
        "transformer_blocks_26": "FLOAT",
        "transformer_blocks_27": "FLOAT",
        "scale_shift_table": "FLOAT",
        "proj_out": "FLOAT",
    }
    INPUT_NAMES = {
        "patchify_proj": "patchify_proj.",
        "adaln_single": "adaln_single.",
        "caption_projection": "caption_projection.",
        "transformer_blocks_0": "transformer_blocks.0.",
        "transformer_blocks_1": "transformer_blocks.1.",
        "transformer_blocks_2": "transformer_blocks.2.",
        "transformer_blocks_3": "transformer_blocks.3.",
        "transformer_blocks_4": "transformer_blocks.4.",
        "transformer_blocks_5": "transformer_blocks.5.",
        "transformer_blocks_6": "transformer_blocks.6.",
        "transformer_blocks_7": "transformer_blocks.7.",
        "transformer_blocks_8": "transformer_blocks.8.",
        "transformer_blocks_9": "transformer_blocks.9.",
        "transformer_blocks_10": "transformer_blocks.10.",
        "transformer_blocks_11": "transformer_blocks.11.",
        "transformer_blocks_12": "transformer_blocks.12.",
        "transformer_blocks_13": "transformer_blocks.13.",
        "transformer_blocks_14": "transformer_blocks.14.",
        "transformer_blocks_15": "transformer_blocks.15.",
        "transformer_blocks_16": "transformer_blocks.16.",
        "transformer_blocks_17": "transformer_blocks.17.",
        "transformer_blocks_18": "transformer_blocks.18.",
        "transformer_blocks_19": "transformer_blocks.19.",
        "transformer_blocks_20": "transformer_blocks.20.",
        "transformer_blocks_21": "transformer_blocks.21.",
        "transformer_blocks_22": "transformer_blocks.22.",
        "transformer_blocks_23": "transformer_blocks.23.",
        "transformer_blocks_24": "transformer_blocks.24.",
        "transformer_blocks_25": "transformer_blocks.25.",
        "transformer_blocks_26": "transformer_blocks.26.",
        "transformer_blocks_27": "transformer_blocks.27.",
        "proj_out": "proj_out.",
    }

    def __init__(
        self,
        node_id: str,
        model1: tuple[str, int],
        model2: tuple[str, int],
        patchify_proj: float | tuple[str, int] = 1.0,
        adaln_single: float | tuple[str, int] = 1.0,
        caption_projection: float | tuple[str, int] = 1.0,
        transformer_blocks_0: float | tuple[str, int] = 1.0,
        transformer_blocks_1: float | tuple[str, int] = 1.0,
        transformer_blocks_2: float | tuple[str, int] = 1.0,
        transformer_blocks_3: float | tuple[str, int] = 1.0,
        transformer_blocks_4: float | tuple[str, int] = 1.0,
        transformer_blocks_5: float | tuple[str, int] = 1.0,
        transformer_blocks_6: float | tuple[str, int] = 1.0,
        transformer_blocks_7: float | tuple[str, int] = 1.0,
        transformer_blocks_8: float | tuple[str, int] = 1.0,
        transformer_blocks_9: float | tuple[str, int] = 1.0,
        transformer_blocks_10: float | tuple[str, int] = 1.0,
        transformer_blocks_11: float | tuple[str, int] = 1.0,
        transformer_blocks_12: float | tuple[str, int] = 1.0,
        transformer_blocks_13: float | tuple[str, int] = 1.0,
        transformer_blocks_14: float | tuple[str, int] = 1.0,
        transformer_blocks_15: float | tuple[str, int] = 1.0,
        transformer_blocks_16: float | tuple[str, int] = 1.0,
        transformer_blocks_17: float | tuple[str, int] = 1.0,
        transformer_blocks_18: float | tuple[str, int] = 1.0,
        transformer_blocks_19: float | tuple[str, int] = 1.0,
        transformer_blocks_20: float | tuple[str, int] = 1.0,
        transformer_blocks_21: float | tuple[str, int] = 1.0,
        transformer_blocks_22: float | tuple[str, int] = 1.0,
        transformer_blocks_23: float | tuple[str, int] = 1.0,
        transformer_blocks_24: float | tuple[str, int] = 1.0,
        transformer_blocks_25: float | tuple[str, int] = 1.0,
        transformer_blocks_26: float | tuple[str, int] = 1.0,
        transformer_blocks_27: float | tuple[str, int] = 1.0,
        scale_shift_table: float | tuple[str, int] = 1.0,
        proj_out: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model1 = model1
        self.model2 = model2
        self.patchify_proj = patchify_proj
        self.adaln_single = adaln_single
        self.caption_projection = caption_projection
        self.transformer_blocks_0 = transformer_blocks_0
        self.transformer_blocks_1 = transformer_blocks_1
        self.transformer_blocks_2 = transformer_blocks_2
        self.transformer_blocks_3 = transformer_blocks_3
        self.transformer_blocks_4 = transformer_blocks_4
        self.transformer_blocks_5 = transformer_blocks_5
        self.transformer_blocks_6 = transformer_blocks_6
        self.transformer_blocks_7 = transformer_blocks_7
        self.transformer_blocks_8 = transformer_blocks_8
        self.transformer_blocks_9 = transformer_blocks_9
        self.transformer_blocks_10 = transformer_blocks_10
        self.transformer_blocks_11 = transformer_blocks_11
        self.transformer_blocks_12 = transformer_blocks_12
        self.transformer_blocks_13 = transformer_blocks_13
        self.transformer_blocks_14 = transformer_blocks_14
        self.transformer_blocks_15 = transformer_blocks_15
        self.transformer_blocks_16 = transformer_blocks_16
        self.transformer_blocks_17 = transformer_blocks_17
        self.transformer_blocks_18 = transformer_blocks_18
        self.transformer_blocks_19 = transformer_blocks_19
        self.transformer_blocks_20 = transformer_blocks_20
        self.transformer_blocks_21 = transformer_blocks_21
        self.transformer_blocks_22 = transformer_blocks_22
        self.transformer_blocks_23 = transformer_blocks_23
        self.transformer_blocks_24 = transformer_blocks_24
        self.transformer_blocks_25 = transformer_blocks_25
        self.transformer_blocks_26 = transformer_blocks_26
        self.transformer_blocks_27 = transformer_blocks_27
        self.scale_shift_table = scale_shift_table
        self.proj_out = proj_out
        self.__validate()

    def __validate(self) -> None:
        validate_range("patchify_proj", self.patchify_proj, 0.0, 1.0)
        validate_range("adaln_single", self.adaln_single, 0.0, 1.0)
        validate_range("caption_projection", self.caption_projection, 0.0, 1.0)
        validate_range("transformer_blocks_0", self.transformer_blocks_0, 0.0, 1.0)
        validate_range("transformer_blocks_1", self.transformer_blocks_1, 0.0, 1.0)
        validate_range("transformer_blocks_2", self.transformer_blocks_2, 0.0, 1.0)
        validate_range("transformer_blocks_3", self.transformer_blocks_3, 0.0, 1.0)
        validate_range("transformer_blocks_4", self.transformer_blocks_4, 0.0, 1.0)
        validate_range("transformer_blocks_5", self.transformer_blocks_5, 0.0, 1.0)
        validate_range("transformer_blocks_6", self.transformer_blocks_6, 0.0, 1.0)
        validate_range("transformer_blocks_7", self.transformer_blocks_7, 0.0, 1.0)
        validate_range("transformer_blocks_8", self.transformer_blocks_8, 0.0, 1.0)
        validate_range("transformer_blocks_9", self.transformer_blocks_9, 0.0, 1.0)
        validate_range("transformer_blocks_10", self.transformer_blocks_10, 0.0, 1.0)
        validate_range("transformer_blocks_11", self.transformer_blocks_11, 0.0, 1.0)
        validate_range("transformer_blocks_12", self.transformer_blocks_12, 0.0, 1.0)
        validate_range("transformer_blocks_13", self.transformer_blocks_13, 0.0, 1.0)
        validate_range("transformer_blocks_14", self.transformer_blocks_14, 0.0, 1.0)
        validate_range("transformer_blocks_15", self.transformer_blocks_15, 0.0, 1.0)
        validate_range("transformer_blocks_16", self.transformer_blocks_16, 0.0, 1.0)
        validate_range("transformer_blocks_17", self.transformer_blocks_17, 0.0, 1.0)
        validate_range("transformer_blocks_18", self.transformer_blocks_18, 0.0, 1.0)
        validate_range("transformer_blocks_19", self.transformer_blocks_19, 0.0, 1.0)
        validate_range("transformer_blocks_20", self.transformer_blocks_20, 0.0, 1.0)
        validate_range("transformer_blocks_21", self.transformer_blocks_21, 0.0, 1.0)
        validate_range("transformer_blocks_22", self.transformer_blocks_22, 0.0, 1.0)
        validate_range("transformer_blocks_23", self.transformer_blocks_23, 0.0, 1.0)
        validate_range("transformer_blocks_24", self.transformer_blocks_24, 0.0, 1.0)
        validate_range("transformer_blocks_25", self.transformer_blocks_25, 0.0, 1.0)
        validate_range("transformer_blocks_26", self.transformer_blocks_26, 0.0, 1.0)
        validate_range("transformer_blocks_27", self.transformer_blocks_27, 0.0, 1.0)
        validate_range("scale_shift_table", self.scale_shift_table, 0.0, 1.0)
        validate_range("proj_out", self.proj_out, 0.0, 1.0)


class ModelMergeCosmos7B(BaseNode):
    __slots__ = (
        "model1",
        "model2",
        "pos_embedder",
        "extra_pos_embedder",
        "x_embedder",
        "t_embedder",
        "affline_norm",
        "blocks_block0",
        "blocks_block1",
        "blocks_block2",
        "blocks_block3",
        "blocks_block4",
        "blocks_block5",
        "blocks_block6",
        "blocks_block7",
        "blocks_block8",
        "blocks_block9",
        "blocks_block10",
        "blocks_block11",
        "blocks_block12",
        "blocks_block13",
        "blocks_block14",
        "blocks_block15",
        "blocks_block16",
        "blocks_block17",
        "blocks_block18",
        "blocks_block19",
        "blocks_block20",
        "blocks_block21",
        "blocks_block22",
        "blocks_block23",
        "blocks_block24",
        "blocks_block25",
        "blocks_block26",
        "blocks_block27",
        "final_layer",
    )
    CATEGORY = "model/merging/model specific"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {
        "model1": "MODEL",
        "model2": "MODEL",
        "pos_embedder": "FLOAT",
        "extra_pos_embedder": "FLOAT",
        "x_embedder": "FLOAT",
        "t_embedder": "FLOAT",
        "affline_norm": "FLOAT",
        "blocks_block0": "FLOAT",
        "blocks_block1": "FLOAT",
        "blocks_block2": "FLOAT",
        "blocks_block3": "FLOAT",
        "blocks_block4": "FLOAT",
        "blocks_block5": "FLOAT",
        "blocks_block6": "FLOAT",
        "blocks_block7": "FLOAT",
        "blocks_block8": "FLOAT",
        "blocks_block9": "FLOAT",
        "blocks_block10": "FLOAT",
        "blocks_block11": "FLOAT",
        "blocks_block12": "FLOAT",
        "blocks_block13": "FLOAT",
        "blocks_block14": "FLOAT",
        "blocks_block15": "FLOAT",
        "blocks_block16": "FLOAT",
        "blocks_block17": "FLOAT",
        "blocks_block18": "FLOAT",
        "blocks_block19": "FLOAT",
        "blocks_block20": "FLOAT",
        "blocks_block21": "FLOAT",
        "blocks_block22": "FLOAT",
        "blocks_block23": "FLOAT",
        "blocks_block24": "FLOAT",
        "blocks_block25": "FLOAT",
        "blocks_block26": "FLOAT",
        "blocks_block27": "FLOAT",
        "final_layer": "FLOAT",
    }
    INPUT_NAMES = {
        "pos_embedder": "pos_embedder.",
        "extra_pos_embedder": "extra_pos_embedder.",
        "x_embedder": "x_embedder.",
        "t_embedder": "t_embedder.",
        "affline_norm": "affline_norm.",
        "blocks_block0": "blocks.block0.",
        "blocks_block1": "blocks.block1.",
        "blocks_block2": "blocks.block2.",
        "blocks_block3": "blocks.block3.",
        "blocks_block4": "blocks.block4.",
        "blocks_block5": "blocks.block5.",
        "blocks_block6": "blocks.block6.",
        "blocks_block7": "blocks.block7.",
        "blocks_block8": "blocks.block8.",
        "blocks_block9": "blocks.block9.",
        "blocks_block10": "blocks.block10.",
        "blocks_block11": "blocks.block11.",
        "blocks_block12": "blocks.block12.",
        "blocks_block13": "blocks.block13.",
        "blocks_block14": "blocks.block14.",
        "blocks_block15": "blocks.block15.",
        "blocks_block16": "blocks.block16.",
        "blocks_block17": "blocks.block17.",
        "blocks_block18": "blocks.block18.",
        "blocks_block19": "blocks.block19.",
        "blocks_block20": "blocks.block20.",
        "blocks_block21": "blocks.block21.",
        "blocks_block22": "blocks.block22.",
        "blocks_block23": "blocks.block23.",
        "blocks_block24": "blocks.block24.",
        "blocks_block25": "blocks.block25.",
        "blocks_block26": "blocks.block26.",
        "blocks_block27": "blocks.block27.",
        "final_layer": "final_layer.",
    }

    def __init__(
        self,
        node_id: str,
        model1: tuple[str, int],
        model2: tuple[str, int],
        pos_embedder: float | tuple[str, int] = 1.0,
        extra_pos_embedder: float | tuple[str, int] = 1.0,
        x_embedder: float | tuple[str, int] = 1.0,
        t_embedder: float | tuple[str, int] = 1.0,
        affline_norm: float | tuple[str, int] = 1.0,
        blocks_block0: float | tuple[str, int] = 1.0,
        blocks_block1: float | tuple[str, int] = 1.0,
        blocks_block2: float | tuple[str, int] = 1.0,
        blocks_block3: float | tuple[str, int] = 1.0,
        blocks_block4: float | tuple[str, int] = 1.0,
        blocks_block5: float | tuple[str, int] = 1.0,
        blocks_block6: float | tuple[str, int] = 1.0,
        blocks_block7: float | tuple[str, int] = 1.0,
        blocks_block8: float | tuple[str, int] = 1.0,
        blocks_block9: float | tuple[str, int] = 1.0,
        blocks_block10: float | tuple[str, int] = 1.0,
        blocks_block11: float | tuple[str, int] = 1.0,
        blocks_block12: float | tuple[str, int] = 1.0,
        blocks_block13: float | tuple[str, int] = 1.0,
        blocks_block14: float | tuple[str, int] = 1.0,
        blocks_block15: float | tuple[str, int] = 1.0,
        blocks_block16: float | tuple[str, int] = 1.0,
        blocks_block17: float | tuple[str, int] = 1.0,
        blocks_block18: float | tuple[str, int] = 1.0,
        blocks_block19: float | tuple[str, int] = 1.0,
        blocks_block20: float | tuple[str, int] = 1.0,
        blocks_block21: float | tuple[str, int] = 1.0,
        blocks_block22: float | tuple[str, int] = 1.0,
        blocks_block23: float | tuple[str, int] = 1.0,
        blocks_block24: float | tuple[str, int] = 1.0,
        blocks_block25: float | tuple[str, int] = 1.0,
        blocks_block26: float | tuple[str, int] = 1.0,
        blocks_block27: float | tuple[str, int] = 1.0,
        final_layer: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model1 = model1
        self.model2 = model2
        self.pos_embedder = pos_embedder
        self.extra_pos_embedder = extra_pos_embedder
        self.x_embedder = x_embedder
        self.t_embedder = t_embedder
        self.affline_norm = affline_norm
        self.blocks_block0 = blocks_block0
        self.blocks_block1 = blocks_block1
        self.blocks_block2 = blocks_block2
        self.blocks_block3 = blocks_block3
        self.blocks_block4 = blocks_block4
        self.blocks_block5 = blocks_block5
        self.blocks_block6 = blocks_block6
        self.blocks_block7 = blocks_block7
        self.blocks_block8 = blocks_block8
        self.blocks_block9 = blocks_block9
        self.blocks_block10 = blocks_block10
        self.blocks_block11 = blocks_block11
        self.blocks_block12 = blocks_block12
        self.blocks_block13 = blocks_block13
        self.blocks_block14 = blocks_block14
        self.blocks_block15 = blocks_block15
        self.blocks_block16 = blocks_block16
        self.blocks_block17 = blocks_block17
        self.blocks_block18 = blocks_block18
        self.blocks_block19 = blocks_block19
        self.blocks_block20 = blocks_block20
        self.blocks_block21 = blocks_block21
        self.blocks_block22 = blocks_block22
        self.blocks_block23 = blocks_block23
        self.blocks_block24 = blocks_block24
        self.blocks_block25 = blocks_block25
        self.blocks_block26 = blocks_block26
        self.blocks_block27 = blocks_block27
        self.final_layer = final_layer
        self.__validate()

    def __validate(self) -> None:
        validate_range("pos_embedder", self.pos_embedder, 0.0, 1.0)
        validate_range("extra_pos_embedder", self.extra_pos_embedder, 0.0, 1.0)
        validate_range("x_embedder", self.x_embedder, 0.0, 1.0)
        validate_range("t_embedder", self.t_embedder, 0.0, 1.0)
        validate_range("affline_norm", self.affline_norm, 0.0, 1.0)
        validate_range("blocks_block0", self.blocks_block0, 0.0, 1.0)
        validate_range("blocks_block1", self.blocks_block1, 0.0, 1.0)
        validate_range("blocks_block2", self.blocks_block2, 0.0, 1.0)
        validate_range("blocks_block3", self.blocks_block3, 0.0, 1.0)
        validate_range("blocks_block4", self.blocks_block4, 0.0, 1.0)
        validate_range("blocks_block5", self.blocks_block5, 0.0, 1.0)
        validate_range("blocks_block6", self.blocks_block6, 0.0, 1.0)
        validate_range("blocks_block7", self.blocks_block7, 0.0, 1.0)
        validate_range("blocks_block8", self.blocks_block8, 0.0, 1.0)
        validate_range("blocks_block9", self.blocks_block9, 0.0, 1.0)
        validate_range("blocks_block10", self.blocks_block10, 0.0, 1.0)
        validate_range("blocks_block11", self.blocks_block11, 0.0, 1.0)
        validate_range("blocks_block12", self.blocks_block12, 0.0, 1.0)
        validate_range("blocks_block13", self.blocks_block13, 0.0, 1.0)
        validate_range("blocks_block14", self.blocks_block14, 0.0, 1.0)
        validate_range("blocks_block15", self.blocks_block15, 0.0, 1.0)
        validate_range("blocks_block16", self.blocks_block16, 0.0, 1.0)
        validate_range("blocks_block17", self.blocks_block17, 0.0, 1.0)
        validate_range("blocks_block18", self.blocks_block18, 0.0, 1.0)
        validate_range("blocks_block19", self.blocks_block19, 0.0, 1.0)
        validate_range("blocks_block20", self.blocks_block20, 0.0, 1.0)
        validate_range("blocks_block21", self.blocks_block21, 0.0, 1.0)
        validate_range("blocks_block22", self.blocks_block22, 0.0, 1.0)
        validate_range("blocks_block23", self.blocks_block23, 0.0, 1.0)
        validate_range("blocks_block24", self.blocks_block24, 0.0, 1.0)
        validate_range("blocks_block25", self.blocks_block25, 0.0, 1.0)
        validate_range("blocks_block26", self.blocks_block26, 0.0, 1.0)
        validate_range("blocks_block27", self.blocks_block27, 0.0, 1.0)
        validate_range("final_layer", self.final_layer, 0.0, 1.0)


class ModelMergeCosmos14B(BaseNode):
    __slots__ = (
        "model1",
        "model2",
        "pos_embedder",
        "extra_pos_embedder",
        "x_embedder",
        "t_embedder",
        "affline_norm",
        "blocks_block0",
        "blocks_block1",
        "blocks_block2",
        "blocks_block3",
        "blocks_block4",
        "blocks_block5",
        "blocks_block6",
        "blocks_block7",
        "blocks_block8",
        "blocks_block9",
        "blocks_block10",
        "blocks_block11",
        "blocks_block12",
        "blocks_block13",
        "blocks_block14",
        "blocks_block15",
        "blocks_block16",
        "blocks_block17",
        "blocks_block18",
        "blocks_block19",
        "blocks_block20",
        "blocks_block21",
        "blocks_block22",
        "blocks_block23",
        "blocks_block24",
        "blocks_block25",
        "blocks_block26",
        "blocks_block27",
        "blocks_block28",
        "blocks_block29",
        "blocks_block30",
        "blocks_block31",
        "blocks_block32",
        "blocks_block33",
        "blocks_block34",
        "blocks_block35",
        "final_layer",
    )
    CATEGORY = "model/merging/model specific"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {
        "model1": "MODEL",
        "model2": "MODEL",
        "pos_embedder": "FLOAT",
        "extra_pos_embedder": "FLOAT",
        "x_embedder": "FLOAT",
        "t_embedder": "FLOAT",
        "affline_norm": "FLOAT",
        "blocks_block0": "FLOAT",
        "blocks_block1": "FLOAT",
        "blocks_block2": "FLOAT",
        "blocks_block3": "FLOAT",
        "blocks_block4": "FLOAT",
        "blocks_block5": "FLOAT",
        "blocks_block6": "FLOAT",
        "blocks_block7": "FLOAT",
        "blocks_block8": "FLOAT",
        "blocks_block9": "FLOAT",
        "blocks_block10": "FLOAT",
        "blocks_block11": "FLOAT",
        "blocks_block12": "FLOAT",
        "blocks_block13": "FLOAT",
        "blocks_block14": "FLOAT",
        "blocks_block15": "FLOAT",
        "blocks_block16": "FLOAT",
        "blocks_block17": "FLOAT",
        "blocks_block18": "FLOAT",
        "blocks_block19": "FLOAT",
        "blocks_block20": "FLOAT",
        "blocks_block21": "FLOAT",
        "blocks_block22": "FLOAT",
        "blocks_block23": "FLOAT",
        "blocks_block24": "FLOAT",
        "blocks_block25": "FLOAT",
        "blocks_block26": "FLOAT",
        "blocks_block27": "FLOAT",
        "blocks_block28": "FLOAT",
        "blocks_block29": "FLOAT",
        "blocks_block30": "FLOAT",
        "blocks_block31": "FLOAT",
        "blocks_block32": "FLOAT",
        "blocks_block33": "FLOAT",
        "blocks_block34": "FLOAT",
        "blocks_block35": "FLOAT",
        "final_layer": "FLOAT",
    }
    INPUT_NAMES = {
        "pos_embedder": "pos_embedder.",
        "extra_pos_embedder": "extra_pos_embedder.",
        "x_embedder": "x_embedder.",
        "t_embedder": "t_embedder.",
        "affline_norm": "affline_norm.",
        "blocks_block0": "blocks.block0.",
        "blocks_block1": "blocks.block1.",
        "blocks_block2": "blocks.block2.",
        "blocks_block3": "blocks.block3.",
        "blocks_block4": "blocks.block4.",
        "blocks_block5": "blocks.block5.",
        "blocks_block6": "blocks.block6.",
        "blocks_block7": "blocks.block7.",
        "blocks_block8": "blocks.block8.",
        "blocks_block9": "blocks.block9.",
        "blocks_block10": "blocks.block10.",
        "blocks_block11": "blocks.block11.",
        "blocks_block12": "blocks.block12.",
        "blocks_block13": "blocks.block13.",
        "blocks_block14": "blocks.block14.",
        "blocks_block15": "blocks.block15.",
        "blocks_block16": "blocks.block16.",
        "blocks_block17": "blocks.block17.",
        "blocks_block18": "blocks.block18.",
        "blocks_block19": "blocks.block19.",
        "blocks_block20": "blocks.block20.",
        "blocks_block21": "blocks.block21.",
        "blocks_block22": "blocks.block22.",
        "blocks_block23": "blocks.block23.",
        "blocks_block24": "blocks.block24.",
        "blocks_block25": "blocks.block25.",
        "blocks_block26": "blocks.block26.",
        "blocks_block27": "blocks.block27.",
        "blocks_block28": "blocks.block28.",
        "blocks_block29": "blocks.block29.",
        "blocks_block30": "blocks.block30.",
        "blocks_block31": "blocks.block31.",
        "blocks_block32": "blocks.block32.",
        "blocks_block33": "blocks.block33.",
        "blocks_block34": "blocks.block34.",
        "blocks_block35": "blocks.block35.",
        "final_layer": "final_layer.",
    }

    def __init__(
        self,
        node_id: str,
        model1: tuple[str, int],
        model2: tuple[str, int],
        pos_embedder: float | tuple[str, int] = 1.0,
        extra_pos_embedder: float | tuple[str, int] = 1.0,
        x_embedder: float | tuple[str, int] = 1.0,
        t_embedder: float | tuple[str, int] = 1.0,
        affline_norm: float | tuple[str, int] = 1.0,
        blocks_block0: float | tuple[str, int] = 1.0,
        blocks_block1: float | tuple[str, int] = 1.0,
        blocks_block2: float | tuple[str, int] = 1.0,
        blocks_block3: float | tuple[str, int] = 1.0,
        blocks_block4: float | tuple[str, int] = 1.0,
        blocks_block5: float | tuple[str, int] = 1.0,
        blocks_block6: float | tuple[str, int] = 1.0,
        blocks_block7: float | tuple[str, int] = 1.0,
        blocks_block8: float | tuple[str, int] = 1.0,
        blocks_block9: float | tuple[str, int] = 1.0,
        blocks_block10: float | tuple[str, int] = 1.0,
        blocks_block11: float | tuple[str, int] = 1.0,
        blocks_block12: float | tuple[str, int] = 1.0,
        blocks_block13: float | tuple[str, int] = 1.0,
        blocks_block14: float | tuple[str, int] = 1.0,
        blocks_block15: float | tuple[str, int] = 1.0,
        blocks_block16: float | tuple[str, int] = 1.0,
        blocks_block17: float | tuple[str, int] = 1.0,
        blocks_block18: float | tuple[str, int] = 1.0,
        blocks_block19: float | tuple[str, int] = 1.0,
        blocks_block20: float | tuple[str, int] = 1.0,
        blocks_block21: float | tuple[str, int] = 1.0,
        blocks_block22: float | tuple[str, int] = 1.0,
        blocks_block23: float | tuple[str, int] = 1.0,
        blocks_block24: float | tuple[str, int] = 1.0,
        blocks_block25: float | tuple[str, int] = 1.0,
        blocks_block26: float | tuple[str, int] = 1.0,
        blocks_block27: float | tuple[str, int] = 1.0,
        blocks_block28: float | tuple[str, int] = 1.0,
        blocks_block29: float | tuple[str, int] = 1.0,
        blocks_block30: float | tuple[str, int] = 1.0,
        blocks_block31: float | tuple[str, int] = 1.0,
        blocks_block32: float | tuple[str, int] = 1.0,
        blocks_block33: float | tuple[str, int] = 1.0,
        blocks_block34: float | tuple[str, int] = 1.0,
        blocks_block35: float | tuple[str, int] = 1.0,
        final_layer: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model1 = model1
        self.model2 = model2
        self.pos_embedder = pos_embedder
        self.extra_pos_embedder = extra_pos_embedder
        self.x_embedder = x_embedder
        self.t_embedder = t_embedder
        self.affline_norm = affline_norm
        self.blocks_block0 = blocks_block0
        self.blocks_block1 = blocks_block1
        self.blocks_block2 = blocks_block2
        self.blocks_block3 = blocks_block3
        self.blocks_block4 = blocks_block4
        self.blocks_block5 = blocks_block5
        self.blocks_block6 = blocks_block6
        self.blocks_block7 = blocks_block7
        self.blocks_block8 = blocks_block8
        self.blocks_block9 = blocks_block9
        self.blocks_block10 = blocks_block10
        self.blocks_block11 = blocks_block11
        self.blocks_block12 = blocks_block12
        self.blocks_block13 = blocks_block13
        self.blocks_block14 = blocks_block14
        self.blocks_block15 = blocks_block15
        self.blocks_block16 = blocks_block16
        self.blocks_block17 = blocks_block17
        self.blocks_block18 = blocks_block18
        self.blocks_block19 = blocks_block19
        self.blocks_block20 = blocks_block20
        self.blocks_block21 = blocks_block21
        self.blocks_block22 = blocks_block22
        self.blocks_block23 = blocks_block23
        self.blocks_block24 = blocks_block24
        self.blocks_block25 = blocks_block25
        self.blocks_block26 = blocks_block26
        self.blocks_block27 = blocks_block27
        self.blocks_block28 = blocks_block28
        self.blocks_block29 = blocks_block29
        self.blocks_block30 = blocks_block30
        self.blocks_block31 = blocks_block31
        self.blocks_block32 = blocks_block32
        self.blocks_block33 = blocks_block33
        self.blocks_block34 = blocks_block34
        self.blocks_block35 = blocks_block35
        self.final_layer = final_layer
        self.__validate()

    def __validate(self) -> None:
        validate_range("pos_embedder", self.pos_embedder, 0.0, 1.0)
        validate_range("extra_pos_embedder", self.extra_pos_embedder, 0.0, 1.0)
        validate_range("x_embedder", self.x_embedder, 0.0, 1.0)
        validate_range("t_embedder", self.t_embedder, 0.0, 1.0)
        validate_range("affline_norm", self.affline_norm, 0.0, 1.0)
        validate_range("blocks_block0", self.blocks_block0, 0.0, 1.0)
        validate_range("blocks_block1", self.blocks_block1, 0.0, 1.0)
        validate_range("blocks_block2", self.blocks_block2, 0.0, 1.0)
        validate_range("blocks_block3", self.blocks_block3, 0.0, 1.0)
        validate_range("blocks_block4", self.blocks_block4, 0.0, 1.0)
        validate_range("blocks_block5", self.blocks_block5, 0.0, 1.0)
        validate_range("blocks_block6", self.blocks_block6, 0.0, 1.0)
        validate_range("blocks_block7", self.blocks_block7, 0.0, 1.0)
        validate_range("blocks_block8", self.blocks_block8, 0.0, 1.0)
        validate_range("blocks_block9", self.blocks_block9, 0.0, 1.0)
        validate_range("blocks_block10", self.blocks_block10, 0.0, 1.0)
        validate_range("blocks_block11", self.blocks_block11, 0.0, 1.0)
        validate_range("blocks_block12", self.blocks_block12, 0.0, 1.0)
        validate_range("blocks_block13", self.blocks_block13, 0.0, 1.0)
        validate_range("blocks_block14", self.blocks_block14, 0.0, 1.0)
        validate_range("blocks_block15", self.blocks_block15, 0.0, 1.0)
        validate_range("blocks_block16", self.blocks_block16, 0.0, 1.0)
        validate_range("blocks_block17", self.blocks_block17, 0.0, 1.0)
        validate_range("blocks_block18", self.blocks_block18, 0.0, 1.0)
        validate_range("blocks_block19", self.blocks_block19, 0.0, 1.0)
        validate_range("blocks_block20", self.blocks_block20, 0.0, 1.0)
        validate_range("blocks_block21", self.blocks_block21, 0.0, 1.0)
        validate_range("blocks_block22", self.blocks_block22, 0.0, 1.0)
        validate_range("blocks_block23", self.blocks_block23, 0.0, 1.0)
        validate_range("blocks_block24", self.blocks_block24, 0.0, 1.0)
        validate_range("blocks_block25", self.blocks_block25, 0.0, 1.0)
        validate_range("blocks_block26", self.blocks_block26, 0.0, 1.0)
        validate_range("blocks_block27", self.blocks_block27, 0.0, 1.0)
        validate_range("blocks_block28", self.blocks_block28, 0.0, 1.0)
        validate_range("blocks_block29", self.blocks_block29, 0.0, 1.0)
        validate_range("blocks_block30", self.blocks_block30, 0.0, 1.0)
        validate_range("blocks_block31", self.blocks_block31, 0.0, 1.0)
        validate_range("blocks_block32", self.blocks_block32, 0.0, 1.0)
        validate_range("blocks_block33", self.blocks_block33, 0.0, 1.0)
        validate_range("blocks_block34", self.blocks_block34, 0.0, 1.0)
        validate_range("blocks_block35", self.blocks_block35, 0.0, 1.0)
        validate_range("final_layer", self.final_layer, 0.0, 1.0)


class ModelMergeWAN2_1(BaseNode):
    __slots__ = (
        "model1",
        "model2",
        "patch_embedding",
        "time_embedding",
        "time_projection",
        "text_embedding",
        "img_emb",
        "blocks_0",
        "blocks_1",
        "blocks_2",
        "blocks_3",
        "blocks_4",
        "blocks_5",
        "blocks_6",
        "blocks_7",
        "blocks_8",
        "blocks_9",
        "blocks_10",
        "blocks_11",
        "blocks_12",
        "blocks_13",
        "blocks_14",
        "blocks_15",
        "blocks_16",
        "blocks_17",
        "blocks_18",
        "blocks_19",
        "blocks_20",
        "blocks_21",
        "blocks_22",
        "blocks_23",
        "blocks_24",
        "blocks_25",
        "blocks_26",
        "blocks_27",
        "blocks_28",
        "blocks_29",
        "blocks_30",
        "blocks_31",
        "blocks_32",
        "blocks_33",
        "blocks_34",
        "blocks_35",
        "blocks_36",
        "blocks_37",
        "blocks_38",
        "blocks_39",
        "head",
    )
    CATEGORY = "model/merging/model specific"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {
        "model1": "MODEL",
        "model2": "MODEL",
        "patch_embedding": "FLOAT",
        "time_embedding": "FLOAT",
        "time_projection": "FLOAT",
        "text_embedding": "FLOAT",
        "img_emb": "FLOAT",
        "blocks_0": "FLOAT",
        "blocks_1": "FLOAT",
        "blocks_2": "FLOAT",
        "blocks_3": "FLOAT",
        "blocks_4": "FLOAT",
        "blocks_5": "FLOAT",
        "blocks_6": "FLOAT",
        "blocks_7": "FLOAT",
        "blocks_8": "FLOAT",
        "blocks_9": "FLOAT",
        "blocks_10": "FLOAT",
        "blocks_11": "FLOAT",
        "blocks_12": "FLOAT",
        "blocks_13": "FLOAT",
        "blocks_14": "FLOAT",
        "blocks_15": "FLOAT",
        "blocks_16": "FLOAT",
        "blocks_17": "FLOAT",
        "blocks_18": "FLOAT",
        "blocks_19": "FLOAT",
        "blocks_20": "FLOAT",
        "blocks_21": "FLOAT",
        "blocks_22": "FLOAT",
        "blocks_23": "FLOAT",
        "blocks_24": "FLOAT",
        "blocks_25": "FLOAT",
        "blocks_26": "FLOAT",
        "blocks_27": "FLOAT",
        "blocks_28": "FLOAT",
        "blocks_29": "FLOAT",
        "blocks_30": "FLOAT",
        "blocks_31": "FLOAT",
        "blocks_32": "FLOAT",
        "blocks_33": "FLOAT",
        "blocks_34": "FLOAT",
        "blocks_35": "FLOAT",
        "blocks_36": "FLOAT",
        "blocks_37": "FLOAT",
        "blocks_38": "FLOAT",
        "blocks_39": "FLOAT",
        "head": "FLOAT",
    }
    INPUT_NAMES = {
        "patch_embedding": "patch_embedding.",
        "time_embedding": "time_embedding.",
        "time_projection": "time_projection.",
        "text_embedding": "text_embedding.",
        "img_emb": "img_emb.",
        "blocks_0": "blocks.0.",
        "blocks_1": "blocks.1.",
        "blocks_2": "blocks.2.",
        "blocks_3": "blocks.3.",
        "blocks_4": "blocks.4.",
        "blocks_5": "blocks.5.",
        "blocks_6": "blocks.6.",
        "blocks_7": "blocks.7.",
        "blocks_8": "blocks.8.",
        "blocks_9": "blocks.9.",
        "blocks_10": "blocks.10.",
        "blocks_11": "blocks.11.",
        "blocks_12": "blocks.12.",
        "blocks_13": "blocks.13.",
        "blocks_14": "blocks.14.",
        "blocks_15": "blocks.15.",
        "blocks_16": "blocks.16.",
        "blocks_17": "blocks.17.",
        "blocks_18": "blocks.18.",
        "blocks_19": "blocks.19.",
        "blocks_20": "blocks.20.",
        "blocks_21": "blocks.21.",
        "blocks_22": "blocks.22.",
        "blocks_23": "blocks.23.",
        "blocks_24": "blocks.24.",
        "blocks_25": "blocks.25.",
        "blocks_26": "blocks.26.",
        "blocks_27": "blocks.27.",
        "blocks_28": "blocks.28.",
        "blocks_29": "blocks.29.",
        "blocks_30": "blocks.30.",
        "blocks_31": "blocks.31.",
        "blocks_32": "blocks.32.",
        "blocks_33": "blocks.33.",
        "blocks_34": "blocks.34.",
        "blocks_35": "blocks.35.",
        "blocks_36": "blocks.36.",
        "blocks_37": "blocks.37.",
        "blocks_38": "blocks.38.",
        "blocks_39": "blocks.39.",
        "head": "head.",
    }

    def __init__(
        self,
        node_id: str,
        model1: tuple[str, int],
        model2: tuple[str, int],
        patch_embedding: float | tuple[str, int] = 1.0,
        time_embedding: float | tuple[str, int] = 1.0,
        time_projection: float | tuple[str, int] = 1.0,
        text_embedding: float | tuple[str, int] = 1.0,
        img_emb: float | tuple[str, int] = 1.0,
        blocks_0: float | tuple[str, int] = 1.0,
        blocks_1: float | tuple[str, int] = 1.0,
        blocks_2: float | tuple[str, int] = 1.0,
        blocks_3: float | tuple[str, int] = 1.0,
        blocks_4: float | tuple[str, int] = 1.0,
        blocks_5: float | tuple[str, int] = 1.0,
        blocks_6: float | tuple[str, int] = 1.0,
        blocks_7: float | tuple[str, int] = 1.0,
        blocks_8: float | tuple[str, int] = 1.0,
        blocks_9: float | tuple[str, int] = 1.0,
        blocks_10: float | tuple[str, int] = 1.0,
        blocks_11: float | tuple[str, int] = 1.0,
        blocks_12: float | tuple[str, int] = 1.0,
        blocks_13: float | tuple[str, int] = 1.0,
        blocks_14: float | tuple[str, int] = 1.0,
        blocks_15: float | tuple[str, int] = 1.0,
        blocks_16: float | tuple[str, int] = 1.0,
        blocks_17: float | tuple[str, int] = 1.0,
        blocks_18: float | tuple[str, int] = 1.0,
        blocks_19: float | tuple[str, int] = 1.0,
        blocks_20: float | tuple[str, int] = 1.0,
        blocks_21: float | tuple[str, int] = 1.0,
        blocks_22: float | tuple[str, int] = 1.0,
        blocks_23: float | tuple[str, int] = 1.0,
        blocks_24: float | tuple[str, int] = 1.0,
        blocks_25: float | tuple[str, int] = 1.0,
        blocks_26: float | tuple[str, int] = 1.0,
        blocks_27: float | tuple[str, int] = 1.0,
        blocks_28: float | tuple[str, int] = 1.0,
        blocks_29: float | tuple[str, int] = 1.0,
        blocks_30: float | tuple[str, int] = 1.0,
        blocks_31: float | tuple[str, int] = 1.0,
        blocks_32: float | tuple[str, int] = 1.0,
        blocks_33: float | tuple[str, int] = 1.0,
        blocks_34: float | tuple[str, int] = 1.0,
        blocks_35: float | tuple[str, int] = 1.0,
        blocks_36: float | tuple[str, int] = 1.0,
        blocks_37: float | tuple[str, int] = 1.0,
        blocks_38: float | tuple[str, int] = 1.0,
        blocks_39: float | tuple[str, int] = 1.0,
        head: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model1 = model1
        self.model2 = model2
        self.patch_embedding = patch_embedding
        self.time_embedding = time_embedding
        self.time_projection = time_projection
        self.text_embedding = text_embedding
        self.img_emb = img_emb
        self.blocks_0 = blocks_0
        self.blocks_1 = blocks_1
        self.blocks_2 = blocks_2
        self.blocks_3 = blocks_3
        self.blocks_4 = blocks_4
        self.blocks_5 = blocks_5
        self.blocks_6 = blocks_6
        self.blocks_7 = blocks_7
        self.blocks_8 = blocks_8
        self.blocks_9 = blocks_9
        self.blocks_10 = blocks_10
        self.blocks_11 = blocks_11
        self.blocks_12 = blocks_12
        self.blocks_13 = blocks_13
        self.blocks_14 = blocks_14
        self.blocks_15 = blocks_15
        self.blocks_16 = blocks_16
        self.blocks_17 = blocks_17
        self.blocks_18 = blocks_18
        self.blocks_19 = blocks_19
        self.blocks_20 = blocks_20
        self.blocks_21 = blocks_21
        self.blocks_22 = blocks_22
        self.blocks_23 = blocks_23
        self.blocks_24 = blocks_24
        self.blocks_25 = blocks_25
        self.blocks_26 = blocks_26
        self.blocks_27 = blocks_27
        self.blocks_28 = blocks_28
        self.blocks_29 = blocks_29
        self.blocks_30 = blocks_30
        self.blocks_31 = blocks_31
        self.blocks_32 = blocks_32
        self.blocks_33 = blocks_33
        self.blocks_34 = blocks_34
        self.blocks_35 = blocks_35
        self.blocks_36 = blocks_36
        self.blocks_37 = blocks_37
        self.blocks_38 = blocks_38
        self.blocks_39 = blocks_39
        self.head = head
        self.__validate()

    def __validate(self) -> None:
        validate_range("patch_embedding", self.patch_embedding, 0.0, 1.0)
        validate_range("time_embedding", self.time_embedding, 0.0, 1.0)
        validate_range("time_projection", self.time_projection, 0.0, 1.0)
        validate_range("text_embedding", self.text_embedding, 0.0, 1.0)
        validate_range("img_emb", self.img_emb, 0.0, 1.0)
        validate_range("blocks_0", self.blocks_0, 0.0, 1.0)
        validate_range("blocks_1", self.blocks_1, 0.0, 1.0)
        validate_range("blocks_2", self.blocks_2, 0.0, 1.0)
        validate_range("blocks_3", self.blocks_3, 0.0, 1.0)
        validate_range("blocks_4", self.blocks_4, 0.0, 1.0)
        validate_range("blocks_5", self.blocks_5, 0.0, 1.0)
        validate_range("blocks_6", self.blocks_6, 0.0, 1.0)
        validate_range("blocks_7", self.blocks_7, 0.0, 1.0)
        validate_range("blocks_8", self.blocks_8, 0.0, 1.0)
        validate_range("blocks_9", self.blocks_9, 0.0, 1.0)
        validate_range("blocks_10", self.blocks_10, 0.0, 1.0)
        validate_range("blocks_11", self.blocks_11, 0.0, 1.0)
        validate_range("blocks_12", self.blocks_12, 0.0, 1.0)
        validate_range("blocks_13", self.blocks_13, 0.0, 1.0)
        validate_range("blocks_14", self.blocks_14, 0.0, 1.0)
        validate_range("blocks_15", self.blocks_15, 0.0, 1.0)
        validate_range("blocks_16", self.blocks_16, 0.0, 1.0)
        validate_range("blocks_17", self.blocks_17, 0.0, 1.0)
        validate_range("blocks_18", self.blocks_18, 0.0, 1.0)
        validate_range("blocks_19", self.blocks_19, 0.0, 1.0)
        validate_range("blocks_20", self.blocks_20, 0.0, 1.0)
        validate_range("blocks_21", self.blocks_21, 0.0, 1.0)
        validate_range("blocks_22", self.blocks_22, 0.0, 1.0)
        validate_range("blocks_23", self.blocks_23, 0.0, 1.0)
        validate_range("blocks_24", self.blocks_24, 0.0, 1.0)
        validate_range("blocks_25", self.blocks_25, 0.0, 1.0)
        validate_range("blocks_26", self.blocks_26, 0.0, 1.0)
        validate_range("blocks_27", self.blocks_27, 0.0, 1.0)
        validate_range("blocks_28", self.blocks_28, 0.0, 1.0)
        validate_range("blocks_29", self.blocks_29, 0.0, 1.0)
        validate_range("blocks_30", self.blocks_30, 0.0, 1.0)
        validate_range("blocks_31", self.blocks_31, 0.0, 1.0)
        validate_range("blocks_32", self.blocks_32, 0.0, 1.0)
        validate_range("blocks_33", self.blocks_33, 0.0, 1.0)
        validate_range("blocks_34", self.blocks_34, 0.0, 1.0)
        validate_range("blocks_35", self.blocks_35, 0.0, 1.0)
        validate_range("blocks_36", self.blocks_36, 0.0, 1.0)
        validate_range("blocks_37", self.blocks_37, 0.0, 1.0)
        validate_range("blocks_38", self.blocks_38, 0.0, 1.0)
        validate_range("blocks_39", self.blocks_39, 0.0, 1.0)
        validate_range("head", self.head, 0.0, 1.0)


class ModelMergeCosmosPredict2_2B(BaseNode):
    __slots__ = (
        "model1",
        "model2",
        "pos_embedder",
        "x_embedder",
        "t_embedder",
        "t_embedding_norm",
        "blocks_0",
        "blocks_1",
        "blocks_2",
        "blocks_3",
        "blocks_4",
        "blocks_5",
        "blocks_6",
        "blocks_7",
        "blocks_8",
        "blocks_9",
        "blocks_10",
        "blocks_11",
        "blocks_12",
        "blocks_13",
        "blocks_14",
        "blocks_15",
        "blocks_16",
        "blocks_17",
        "blocks_18",
        "blocks_19",
        "blocks_20",
        "blocks_21",
        "blocks_22",
        "blocks_23",
        "blocks_24",
        "blocks_25",
        "blocks_26",
        "blocks_27",
        "final_layer",
    )
    CATEGORY = "model/merging/model specific"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {
        "model1": "MODEL",
        "model2": "MODEL",
        "pos_embedder": "FLOAT",
        "x_embedder": "FLOAT",
        "t_embedder": "FLOAT",
        "t_embedding_norm": "FLOAT",
        "blocks_0": "FLOAT",
        "blocks_1": "FLOAT",
        "blocks_2": "FLOAT",
        "blocks_3": "FLOAT",
        "blocks_4": "FLOAT",
        "blocks_5": "FLOAT",
        "blocks_6": "FLOAT",
        "blocks_7": "FLOAT",
        "blocks_8": "FLOAT",
        "blocks_9": "FLOAT",
        "blocks_10": "FLOAT",
        "blocks_11": "FLOAT",
        "blocks_12": "FLOAT",
        "blocks_13": "FLOAT",
        "blocks_14": "FLOAT",
        "blocks_15": "FLOAT",
        "blocks_16": "FLOAT",
        "blocks_17": "FLOAT",
        "blocks_18": "FLOAT",
        "blocks_19": "FLOAT",
        "blocks_20": "FLOAT",
        "blocks_21": "FLOAT",
        "blocks_22": "FLOAT",
        "blocks_23": "FLOAT",
        "blocks_24": "FLOAT",
        "blocks_25": "FLOAT",
        "blocks_26": "FLOAT",
        "blocks_27": "FLOAT",
        "final_layer": "FLOAT",
    }
    INPUT_NAMES = {
        "pos_embedder": "pos_embedder.",
        "x_embedder": "x_embedder.",
        "t_embedder": "t_embedder.",
        "t_embedding_norm": "t_embedding_norm.",
        "blocks_0": "blocks.0.",
        "blocks_1": "blocks.1.",
        "blocks_2": "blocks.2.",
        "blocks_3": "blocks.3.",
        "blocks_4": "blocks.4.",
        "blocks_5": "blocks.5.",
        "blocks_6": "blocks.6.",
        "blocks_7": "blocks.7.",
        "blocks_8": "blocks.8.",
        "blocks_9": "blocks.9.",
        "blocks_10": "blocks.10.",
        "blocks_11": "blocks.11.",
        "blocks_12": "blocks.12.",
        "blocks_13": "blocks.13.",
        "blocks_14": "blocks.14.",
        "blocks_15": "blocks.15.",
        "blocks_16": "blocks.16.",
        "blocks_17": "blocks.17.",
        "blocks_18": "blocks.18.",
        "blocks_19": "blocks.19.",
        "blocks_20": "blocks.20.",
        "blocks_21": "blocks.21.",
        "blocks_22": "blocks.22.",
        "blocks_23": "blocks.23.",
        "blocks_24": "blocks.24.",
        "blocks_25": "blocks.25.",
        "blocks_26": "blocks.26.",
        "blocks_27": "blocks.27.",
        "final_layer": "final_layer.",
    }

    def __init__(
        self,
        node_id: str,
        model1: tuple[str, int],
        model2: tuple[str, int],
        pos_embedder: float | tuple[str, int] = 1.0,
        x_embedder: float | tuple[str, int] = 1.0,
        t_embedder: float | tuple[str, int] = 1.0,
        t_embedding_norm: float | tuple[str, int] = 1.0,
        blocks_0: float | tuple[str, int] = 1.0,
        blocks_1: float | tuple[str, int] = 1.0,
        blocks_2: float | tuple[str, int] = 1.0,
        blocks_3: float | tuple[str, int] = 1.0,
        blocks_4: float | tuple[str, int] = 1.0,
        blocks_5: float | tuple[str, int] = 1.0,
        blocks_6: float | tuple[str, int] = 1.0,
        blocks_7: float | tuple[str, int] = 1.0,
        blocks_8: float | tuple[str, int] = 1.0,
        blocks_9: float | tuple[str, int] = 1.0,
        blocks_10: float | tuple[str, int] = 1.0,
        blocks_11: float | tuple[str, int] = 1.0,
        blocks_12: float | tuple[str, int] = 1.0,
        blocks_13: float | tuple[str, int] = 1.0,
        blocks_14: float | tuple[str, int] = 1.0,
        blocks_15: float | tuple[str, int] = 1.0,
        blocks_16: float | tuple[str, int] = 1.0,
        blocks_17: float | tuple[str, int] = 1.0,
        blocks_18: float | tuple[str, int] = 1.0,
        blocks_19: float | tuple[str, int] = 1.0,
        blocks_20: float | tuple[str, int] = 1.0,
        blocks_21: float | tuple[str, int] = 1.0,
        blocks_22: float | tuple[str, int] = 1.0,
        blocks_23: float | tuple[str, int] = 1.0,
        blocks_24: float | tuple[str, int] = 1.0,
        blocks_25: float | tuple[str, int] = 1.0,
        blocks_26: float | tuple[str, int] = 1.0,
        blocks_27: float | tuple[str, int] = 1.0,
        final_layer: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model1 = model1
        self.model2 = model2
        self.pos_embedder = pos_embedder
        self.x_embedder = x_embedder
        self.t_embedder = t_embedder
        self.t_embedding_norm = t_embedding_norm
        self.blocks_0 = blocks_0
        self.blocks_1 = blocks_1
        self.blocks_2 = blocks_2
        self.blocks_3 = blocks_3
        self.blocks_4 = blocks_4
        self.blocks_5 = blocks_5
        self.blocks_6 = blocks_6
        self.blocks_7 = blocks_7
        self.blocks_8 = blocks_8
        self.blocks_9 = blocks_9
        self.blocks_10 = blocks_10
        self.blocks_11 = blocks_11
        self.blocks_12 = blocks_12
        self.blocks_13 = blocks_13
        self.blocks_14 = blocks_14
        self.blocks_15 = blocks_15
        self.blocks_16 = blocks_16
        self.blocks_17 = blocks_17
        self.blocks_18 = blocks_18
        self.blocks_19 = blocks_19
        self.blocks_20 = blocks_20
        self.blocks_21 = blocks_21
        self.blocks_22 = blocks_22
        self.blocks_23 = blocks_23
        self.blocks_24 = blocks_24
        self.blocks_25 = blocks_25
        self.blocks_26 = blocks_26
        self.blocks_27 = blocks_27
        self.final_layer = final_layer
        self.__validate()

    def __validate(self) -> None:
        validate_range("pos_embedder", self.pos_embedder, 0.0, 1.0)
        validate_range("x_embedder", self.x_embedder, 0.0, 1.0)
        validate_range("t_embedder", self.t_embedder, 0.0, 1.0)
        validate_range("t_embedding_norm", self.t_embedding_norm, 0.0, 1.0)
        validate_range("blocks_0", self.blocks_0, 0.0, 1.0)
        validate_range("blocks_1", self.blocks_1, 0.0, 1.0)
        validate_range("blocks_2", self.blocks_2, 0.0, 1.0)
        validate_range("blocks_3", self.blocks_3, 0.0, 1.0)
        validate_range("blocks_4", self.blocks_4, 0.0, 1.0)
        validate_range("blocks_5", self.blocks_5, 0.0, 1.0)
        validate_range("blocks_6", self.blocks_6, 0.0, 1.0)
        validate_range("blocks_7", self.blocks_7, 0.0, 1.0)
        validate_range("blocks_8", self.blocks_8, 0.0, 1.0)
        validate_range("blocks_9", self.blocks_9, 0.0, 1.0)
        validate_range("blocks_10", self.blocks_10, 0.0, 1.0)
        validate_range("blocks_11", self.blocks_11, 0.0, 1.0)
        validate_range("blocks_12", self.blocks_12, 0.0, 1.0)
        validate_range("blocks_13", self.blocks_13, 0.0, 1.0)
        validate_range("blocks_14", self.blocks_14, 0.0, 1.0)
        validate_range("blocks_15", self.blocks_15, 0.0, 1.0)
        validate_range("blocks_16", self.blocks_16, 0.0, 1.0)
        validate_range("blocks_17", self.blocks_17, 0.0, 1.0)
        validate_range("blocks_18", self.blocks_18, 0.0, 1.0)
        validate_range("blocks_19", self.blocks_19, 0.0, 1.0)
        validate_range("blocks_20", self.blocks_20, 0.0, 1.0)
        validate_range("blocks_21", self.blocks_21, 0.0, 1.0)
        validate_range("blocks_22", self.blocks_22, 0.0, 1.0)
        validate_range("blocks_23", self.blocks_23, 0.0, 1.0)
        validate_range("blocks_24", self.blocks_24, 0.0, 1.0)
        validate_range("blocks_25", self.blocks_25, 0.0, 1.0)
        validate_range("blocks_26", self.blocks_26, 0.0, 1.0)
        validate_range("blocks_27", self.blocks_27, 0.0, 1.0)
        validate_range("final_layer", self.final_layer, 0.0, 1.0)


class ModelMergeCosmosPredict2_14B(BaseNode):
    __slots__ = (
        "model1",
        "model2",
        "pos_embedder",
        "x_embedder",
        "t_embedder",
        "t_embedding_norm",
        "blocks_0",
        "blocks_1",
        "blocks_2",
        "blocks_3",
        "blocks_4",
        "blocks_5",
        "blocks_6",
        "blocks_7",
        "blocks_8",
        "blocks_9",
        "blocks_10",
        "blocks_11",
        "blocks_12",
        "blocks_13",
        "blocks_14",
        "blocks_15",
        "blocks_16",
        "blocks_17",
        "blocks_18",
        "blocks_19",
        "blocks_20",
        "blocks_21",
        "blocks_22",
        "blocks_23",
        "blocks_24",
        "blocks_25",
        "blocks_26",
        "blocks_27",
        "blocks_28",
        "blocks_29",
        "blocks_30",
        "blocks_31",
        "blocks_32",
        "blocks_33",
        "blocks_34",
        "blocks_35",
        "final_layer",
    )
    CATEGORY = "model/merging/model specific"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {
        "model1": "MODEL",
        "model2": "MODEL",
        "pos_embedder": "FLOAT",
        "x_embedder": "FLOAT",
        "t_embedder": "FLOAT",
        "t_embedding_norm": "FLOAT",
        "blocks_0": "FLOAT",
        "blocks_1": "FLOAT",
        "blocks_2": "FLOAT",
        "blocks_3": "FLOAT",
        "blocks_4": "FLOAT",
        "blocks_5": "FLOAT",
        "blocks_6": "FLOAT",
        "blocks_7": "FLOAT",
        "blocks_8": "FLOAT",
        "blocks_9": "FLOAT",
        "blocks_10": "FLOAT",
        "blocks_11": "FLOAT",
        "blocks_12": "FLOAT",
        "blocks_13": "FLOAT",
        "blocks_14": "FLOAT",
        "blocks_15": "FLOAT",
        "blocks_16": "FLOAT",
        "blocks_17": "FLOAT",
        "blocks_18": "FLOAT",
        "blocks_19": "FLOAT",
        "blocks_20": "FLOAT",
        "blocks_21": "FLOAT",
        "blocks_22": "FLOAT",
        "blocks_23": "FLOAT",
        "blocks_24": "FLOAT",
        "blocks_25": "FLOAT",
        "blocks_26": "FLOAT",
        "blocks_27": "FLOAT",
        "blocks_28": "FLOAT",
        "blocks_29": "FLOAT",
        "blocks_30": "FLOAT",
        "blocks_31": "FLOAT",
        "blocks_32": "FLOAT",
        "blocks_33": "FLOAT",
        "blocks_34": "FLOAT",
        "blocks_35": "FLOAT",
        "final_layer": "FLOAT",
    }
    INPUT_NAMES = {
        "pos_embedder": "pos_embedder.",
        "x_embedder": "x_embedder.",
        "t_embedder": "t_embedder.",
        "t_embedding_norm": "t_embedding_norm.",
        "blocks_0": "blocks.0.",
        "blocks_1": "blocks.1.",
        "blocks_2": "blocks.2.",
        "blocks_3": "blocks.3.",
        "blocks_4": "blocks.4.",
        "blocks_5": "blocks.5.",
        "blocks_6": "blocks.6.",
        "blocks_7": "blocks.7.",
        "blocks_8": "blocks.8.",
        "blocks_9": "blocks.9.",
        "blocks_10": "blocks.10.",
        "blocks_11": "blocks.11.",
        "blocks_12": "blocks.12.",
        "blocks_13": "blocks.13.",
        "blocks_14": "blocks.14.",
        "blocks_15": "blocks.15.",
        "blocks_16": "blocks.16.",
        "blocks_17": "blocks.17.",
        "blocks_18": "blocks.18.",
        "blocks_19": "blocks.19.",
        "blocks_20": "blocks.20.",
        "blocks_21": "blocks.21.",
        "blocks_22": "blocks.22.",
        "blocks_23": "blocks.23.",
        "blocks_24": "blocks.24.",
        "blocks_25": "blocks.25.",
        "blocks_26": "blocks.26.",
        "blocks_27": "blocks.27.",
        "blocks_28": "blocks.28.",
        "blocks_29": "blocks.29.",
        "blocks_30": "blocks.30.",
        "blocks_31": "blocks.31.",
        "blocks_32": "blocks.32.",
        "blocks_33": "blocks.33.",
        "blocks_34": "blocks.34.",
        "blocks_35": "blocks.35.",
        "final_layer": "final_layer.",
    }

    def __init__(
        self,
        node_id: str,
        model1: tuple[str, int],
        model2: tuple[str, int],
        pos_embedder: float | tuple[str, int] = 1.0,
        x_embedder: float | tuple[str, int] = 1.0,
        t_embedder: float | tuple[str, int] = 1.0,
        t_embedding_norm: float | tuple[str, int] = 1.0,
        blocks_0: float | tuple[str, int] = 1.0,
        blocks_1: float | tuple[str, int] = 1.0,
        blocks_2: float | tuple[str, int] = 1.0,
        blocks_3: float | tuple[str, int] = 1.0,
        blocks_4: float | tuple[str, int] = 1.0,
        blocks_5: float | tuple[str, int] = 1.0,
        blocks_6: float | tuple[str, int] = 1.0,
        blocks_7: float | tuple[str, int] = 1.0,
        blocks_8: float | tuple[str, int] = 1.0,
        blocks_9: float | tuple[str, int] = 1.0,
        blocks_10: float | tuple[str, int] = 1.0,
        blocks_11: float | tuple[str, int] = 1.0,
        blocks_12: float | tuple[str, int] = 1.0,
        blocks_13: float | tuple[str, int] = 1.0,
        blocks_14: float | tuple[str, int] = 1.0,
        blocks_15: float | tuple[str, int] = 1.0,
        blocks_16: float | tuple[str, int] = 1.0,
        blocks_17: float | tuple[str, int] = 1.0,
        blocks_18: float | tuple[str, int] = 1.0,
        blocks_19: float | tuple[str, int] = 1.0,
        blocks_20: float | tuple[str, int] = 1.0,
        blocks_21: float | tuple[str, int] = 1.0,
        blocks_22: float | tuple[str, int] = 1.0,
        blocks_23: float | tuple[str, int] = 1.0,
        blocks_24: float | tuple[str, int] = 1.0,
        blocks_25: float | tuple[str, int] = 1.0,
        blocks_26: float | tuple[str, int] = 1.0,
        blocks_27: float | tuple[str, int] = 1.0,
        blocks_28: float | tuple[str, int] = 1.0,
        blocks_29: float | tuple[str, int] = 1.0,
        blocks_30: float | tuple[str, int] = 1.0,
        blocks_31: float | tuple[str, int] = 1.0,
        blocks_32: float | tuple[str, int] = 1.0,
        blocks_33: float | tuple[str, int] = 1.0,
        blocks_34: float | tuple[str, int] = 1.0,
        blocks_35: float | tuple[str, int] = 1.0,
        final_layer: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model1 = model1
        self.model2 = model2
        self.pos_embedder = pos_embedder
        self.x_embedder = x_embedder
        self.t_embedder = t_embedder
        self.t_embedding_norm = t_embedding_norm
        self.blocks_0 = blocks_0
        self.blocks_1 = blocks_1
        self.blocks_2 = blocks_2
        self.blocks_3 = blocks_3
        self.blocks_4 = blocks_4
        self.blocks_5 = blocks_5
        self.blocks_6 = blocks_6
        self.blocks_7 = blocks_7
        self.blocks_8 = blocks_8
        self.blocks_9 = blocks_9
        self.blocks_10 = blocks_10
        self.blocks_11 = blocks_11
        self.blocks_12 = blocks_12
        self.blocks_13 = blocks_13
        self.blocks_14 = blocks_14
        self.blocks_15 = blocks_15
        self.blocks_16 = blocks_16
        self.blocks_17 = blocks_17
        self.blocks_18 = blocks_18
        self.blocks_19 = blocks_19
        self.blocks_20 = blocks_20
        self.blocks_21 = blocks_21
        self.blocks_22 = blocks_22
        self.blocks_23 = blocks_23
        self.blocks_24 = blocks_24
        self.blocks_25 = blocks_25
        self.blocks_26 = blocks_26
        self.blocks_27 = blocks_27
        self.blocks_28 = blocks_28
        self.blocks_29 = blocks_29
        self.blocks_30 = blocks_30
        self.blocks_31 = blocks_31
        self.blocks_32 = blocks_32
        self.blocks_33 = blocks_33
        self.blocks_34 = blocks_34
        self.blocks_35 = blocks_35
        self.final_layer = final_layer
        self.__validate()

    def __validate(self) -> None:
        validate_range("pos_embedder", self.pos_embedder, 0.0, 1.0)
        validate_range("x_embedder", self.x_embedder, 0.0, 1.0)
        validate_range("t_embedder", self.t_embedder, 0.0, 1.0)
        validate_range("t_embedding_norm", self.t_embedding_norm, 0.0, 1.0)
        validate_range("blocks_0", self.blocks_0, 0.0, 1.0)
        validate_range("blocks_1", self.blocks_1, 0.0, 1.0)
        validate_range("blocks_2", self.blocks_2, 0.0, 1.0)
        validate_range("blocks_3", self.blocks_3, 0.0, 1.0)
        validate_range("blocks_4", self.blocks_4, 0.0, 1.0)
        validate_range("blocks_5", self.blocks_5, 0.0, 1.0)
        validate_range("blocks_6", self.blocks_6, 0.0, 1.0)
        validate_range("blocks_7", self.blocks_7, 0.0, 1.0)
        validate_range("blocks_8", self.blocks_8, 0.0, 1.0)
        validate_range("blocks_9", self.blocks_9, 0.0, 1.0)
        validate_range("blocks_10", self.blocks_10, 0.0, 1.0)
        validate_range("blocks_11", self.blocks_11, 0.0, 1.0)
        validate_range("blocks_12", self.blocks_12, 0.0, 1.0)
        validate_range("blocks_13", self.blocks_13, 0.0, 1.0)
        validate_range("blocks_14", self.blocks_14, 0.0, 1.0)
        validate_range("blocks_15", self.blocks_15, 0.0, 1.0)
        validate_range("blocks_16", self.blocks_16, 0.0, 1.0)
        validate_range("blocks_17", self.blocks_17, 0.0, 1.0)
        validate_range("blocks_18", self.blocks_18, 0.0, 1.0)
        validate_range("blocks_19", self.blocks_19, 0.0, 1.0)
        validate_range("blocks_20", self.blocks_20, 0.0, 1.0)
        validate_range("blocks_21", self.blocks_21, 0.0, 1.0)
        validate_range("blocks_22", self.blocks_22, 0.0, 1.0)
        validate_range("blocks_23", self.blocks_23, 0.0, 1.0)
        validate_range("blocks_24", self.blocks_24, 0.0, 1.0)
        validate_range("blocks_25", self.blocks_25, 0.0, 1.0)
        validate_range("blocks_26", self.blocks_26, 0.0, 1.0)
        validate_range("blocks_27", self.blocks_27, 0.0, 1.0)
        validate_range("blocks_28", self.blocks_28, 0.0, 1.0)
        validate_range("blocks_29", self.blocks_29, 0.0, 1.0)
        validate_range("blocks_30", self.blocks_30, 0.0, 1.0)
        validate_range("blocks_31", self.blocks_31, 0.0, 1.0)
        validate_range("blocks_32", self.blocks_32, 0.0, 1.0)
        validate_range("blocks_33", self.blocks_33, 0.0, 1.0)
        validate_range("blocks_34", self.blocks_34, 0.0, 1.0)
        validate_range("blocks_35", self.blocks_35, 0.0, 1.0)
        validate_range("final_layer", self.final_layer, 0.0, 1.0)


class ModelMergeQwenImage(BaseNode):
    __slots__ = (
        "model1",
        "model2",
        "pos_embeds",
        "img_in",
        "txt_norm",
        "txt_in",
        "time_text_embed",
        "transformer_blocks_0",
        "transformer_blocks_1",
        "transformer_blocks_2",
        "transformer_blocks_3",
        "transformer_blocks_4",
        "transformer_blocks_5",
        "transformer_blocks_6",
        "transformer_blocks_7",
        "transformer_blocks_8",
        "transformer_blocks_9",
        "transformer_blocks_10",
        "transformer_blocks_11",
        "transformer_blocks_12",
        "transformer_blocks_13",
        "transformer_blocks_14",
        "transformer_blocks_15",
        "transformer_blocks_16",
        "transformer_blocks_17",
        "transformer_blocks_18",
        "transformer_blocks_19",
        "transformer_blocks_20",
        "transformer_blocks_21",
        "transformer_blocks_22",
        "transformer_blocks_23",
        "transformer_blocks_24",
        "transformer_blocks_25",
        "transformer_blocks_26",
        "transformer_blocks_27",
        "transformer_blocks_28",
        "transformer_blocks_29",
        "transformer_blocks_30",
        "transformer_blocks_31",
        "transformer_blocks_32",
        "transformer_blocks_33",
        "transformer_blocks_34",
        "transformer_blocks_35",
        "transformer_blocks_36",
        "transformer_blocks_37",
        "transformer_blocks_38",
        "transformer_blocks_39",
        "transformer_blocks_40",
        "transformer_blocks_41",
        "transformer_blocks_42",
        "transformer_blocks_43",
        "transformer_blocks_44",
        "transformer_blocks_45",
        "transformer_blocks_46",
        "transformer_blocks_47",
        "transformer_blocks_48",
        "transformer_blocks_49",
        "transformer_blocks_50",
        "transformer_blocks_51",
        "transformer_blocks_52",
        "transformer_blocks_53",
        "transformer_blocks_54",
        "transformer_blocks_55",
        "transformer_blocks_56",
        "transformer_blocks_57",
        "transformer_blocks_58",
        "transformer_blocks_59",
        "proj_out",
    )
    CATEGORY = "model/merging/model specific"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {
        "model1": "MODEL",
        "model2": "MODEL",
        "pos_embeds": "FLOAT",
        "img_in": "FLOAT",
        "txt_norm": "FLOAT",
        "txt_in": "FLOAT",
        "time_text_embed": "FLOAT",
        "transformer_blocks_0": "FLOAT",
        "transformer_blocks_1": "FLOAT",
        "transformer_blocks_2": "FLOAT",
        "transformer_blocks_3": "FLOAT",
        "transformer_blocks_4": "FLOAT",
        "transformer_blocks_5": "FLOAT",
        "transformer_blocks_6": "FLOAT",
        "transformer_blocks_7": "FLOAT",
        "transformer_blocks_8": "FLOAT",
        "transformer_blocks_9": "FLOAT",
        "transformer_blocks_10": "FLOAT",
        "transformer_blocks_11": "FLOAT",
        "transformer_blocks_12": "FLOAT",
        "transformer_blocks_13": "FLOAT",
        "transformer_blocks_14": "FLOAT",
        "transformer_blocks_15": "FLOAT",
        "transformer_blocks_16": "FLOAT",
        "transformer_blocks_17": "FLOAT",
        "transformer_blocks_18": "FLOAT",
        "transformer_blocks_19": "FLOAT",
        "transformer_blocks_20": "FLOAT",
        "transformer_blocks_21": "FLOAT",
        "transformer_blocks_22": "FLOAT",
        "transformer_blocks_23": "FLOAT",
        "transformer_blocks_24": "FLOAT",
        "transformer_blocks_25": "FLOAT",
        "transformer_blocks_26": "FLOAT",
        "transformer_blocks_27": "FLOAT",
        "transformer_blocks_28": "FLOAT",
        "transformer_blocks_29": "FLOAT",
        "transformer_blocks_30": "FLOAT",
        "transformer_blocks_31": "FLOAT",
        "transformer_blocks_32": "FLOAT",
        "transformer_blocks_33": "FLOAT",
        "transformer_blocks_34": "FLOAT",
        "transformer_blocks_35": "FLOAT",
        "transformer_blocks_36": "FLOAT",
        "transformer_blocks_37": "FLOAT",
        "transformer_blocks_38": "FLOAT",
        "transformer_blocks_39": "FLOAT",
        "transformer_blocks_40": "FLOAT",
        "transformer_blocks_41": "FLOAT",
        "transformer_blocks_42": "FLOAT",
        "transformer_blocks_43": "FLOAT",
        "transformer_blocks_44": "FLOAT",
        "transformer_blocks_45": "FLOAT",
        "transformer_blocks_46": "FLOAT",
        "transformer_blocks_47": "FLOAT",
        "transformer_blocks_48": "FLOAT",
        "transformer_blocks_49": "FLOAT",
        "transformer_blocks_50": "FLOAT",
        "transformer_blocks_51": "FLOAT",
        "transformer_blocks_52": "FLOAT",
        "transformer_blocks_53": "FLOAT",
        "transformer_blocks_54": "FLOAT",
        "transformer_blocks_55": "FLOAT",
        "transformer_blocks_56": "FLOAT",
        "transformer_blocks_57": "FLOAT",
        "transformer_blocks_58": "FLOAT",
        "transformer_blocks_59": "FLOAT",
        "proj_out": "FLOAT",
    }
    INPUT_NAMES = {
        "pos_embeds": "pos_embeds.",
        "img_in": "img_in.",
        "txt_norm": "txt_norm.",
        "txt_in": "txt_in.",
        "time_text_embed": "time_text_embed.",
        "transformer_blocks_0": "transformer_blocks.0.",
        "transformer_blocks_1": "transformer_blocks.1.",
        "transformer_blocks_2": "transformer_blocks.2.",
        "transformer_blocks_3": "transformer_blocks.3.",
        "transformer_blocks_4": "transformer_blocks.4.",
        "transformer_blocks_5": "transformer_blocks.5.",
        "transformer_blocks_6": "transformer_blocks.6.",
        "transformer_blocks_7": "transformer_blocks.7.",
        "transformer_blocks_8": "transformer_blocks.8.",
        "transformer_blocks_9": "transformer_blocks.9.",
        "transformer_blocks_10": "transformer_blocks.10.",
        "transformer_blocks_11": "transformer_blocks.11.",
        "transformer_blocks_12": "transformer_blocks.12.",
        "transformer_blocks_13": "transformer_blocks.13.",
        "transformer_blocks_14": "transformer_blocks.14.",
        "transformer_blocks_15": "transformer_blocks.15.",
        "transformer_blocks_16": "transformer_blocks.16.",
        "transformer_blocks_17": "transformer_blocks.17.",
        "transformer_blocks_18": "transformer_blocks.18.",
        "transformer_blocks_19": "transformer_blocks.19.",
        "transformer_blocks_20": "transformer_blocks.20.",
        "transformer_blocks_21": "transformer_blocks.21.",
        "transformer_blocks_22": "transformer_blocks.22.",
        "transformer_blocks_23": "transformer_blocks.23.",
        "transformer_blocks_24": "transformer_blocks.24.",
        "transformer_blocks_25": "transformer_blocks.25.",
        "transformer_blocks_26": "transformer_blocks.26.",
        "transformer_blocks_27": "transformer_blocks.27.",
        "transformer_blocks_28": "transformer_blocks.28.",
        "transformer_blocks_29": "transformer_blocks.29.",
        "transformer_blocks_30": "transformer_blocks.30.",
        "transformer_blocks_31": "transformer_blocks.31.",
        "transformer_blocks_32": "transformer_blocks.32.",
        "transformer_blocks_33": "transformer_blocks.33.",
        "transformer_blocks_34": "transformer_blocks.34.",
        "transformer_blocks_35": "transformer_blocks.35.",
        "transformer_blocks_36": "transformer_blocks.36.",
        "transformer_blocks_37": "transformer_blocks.37.",
        "transformer_blocks_38": "transformer_blocks.38.",
        "transformer_blocks_39": "transformer_blocks.39.",
        "transformer_blocks_40": "transformer_blocks.40.",
        "transformer_blocks_41": "transformer_blocks.41.",
        "transformer_blocks_42": "transformer_blocks.42.",
        "transformer_blocks_43": "transformer_blocks.43.",
        "transformer_blocks_44": "transformer_blocks.44.",
        "transformer_blocks_45": "transformer_blocks.45.",
        "transformer_blocks_46": "transformer_blocks.46.",
        "transformer_blocks_47": "transformer_blocks.47.",
        "transformer_blocks_48": "transformer_blocks.48.",
        "transformer_blocks_49": "transformer_blocks.49.",
        "transformer_blocks_50": "transformer_blocks.50.",
        "transformer_blocks_51": "transformer_blocks.51.",
        "transformer_blocks_52": "transformer_blocks.52.",
        "transformer_blocks_53": "transformer_blocks.53.",
        "transformer_blocks_54": "transformer_blocks.54.",
        "transformer_blocks_55": "transformer_blocks.55.",
        "transformer_blocks_56": "transformer_blocks.56.",
        "transformer_blocks_57": "transformer_blocks.57.",
        "transformer_blocks_58": "transformer_blocks.58.",
        "transformer_blocks_59": "transformer_blocks.59.",
        "proj_out": "proj_out.",
    }

    def __init__(
        self,
        node_id: str,
        model1: tuple[str, int],
        model2: tuple[str, int],
        pos_embeds: float | tuple[str, int] = 1.0,
        img_in: float | tuple[str, int] = 1.0,
        txt_norm: float | tuple[str, int] = 1.0,
        txt_in: float | tuple[str, int] = 1.0,
        time_text_embed: float | tuple[str, int] = 1.0,
        transformer_blocks_0: float | tuple[str, int] = 1.0,
        transformer_blocks_1: float | tuple[str, int] = 1.0,
        transformer_blocks_2: float | tuple[str, int] = 1.0,
        transformer_blocks_3: float | tuple[str, int] = 1.0,
        transformer_blocks_4: float | tuple[str, int] = 1.0,
        transformer_blocks_5: float | tuple[str, int] = 1.0,
        transformer_blocks_6: float | tuple[str, int] = 1.0,
        transformer_blocks_7: float | tuple[str, int] = 1.0,
        transformer_blocks_8: float | tuple[str, int] = 1.0,
        transformer_blocks_9: float | tuple[str, int] = 1.0,
        transformer_blocks_10: float | tuple[str, int] = 1.0,
        transformer_blocks_11: float | tuple[str, int] = 1.0,
        transformer_blocks_12: float | tuple[str, int] = 1.0,
        transformer_blocks_13: float | tuple[str, int] = 1.0,
        transformer_blocks_14: float | tuple[str, int] = 1.0,
        transformer_blocks_15: float | tuple[str, int] = 1.0,
        transformer_blocks_16: float | tuple[str, int] = 1.0,
        transformer_blocks_17: float | tuple[str, int] = 1.0,
        transformer_blocks_18: float | tuple[str, int] = 1.0,
        transformer_blocks_19: float | tuple[str, int] = 1.0,
        transformer_blocks_20: float | tuple[str, int] = 1.0,
        transformer_blocks_21: float | tuple[str, int] = 1.0,
        transformer_blocks_22: float | tuple[str, int] = 1.0,
        transformer_blocks_23: float | tuple[str, int] = 1.0,
        transformer_blocks_24: float | tuple[str, int] = 1.0,
        transformer_blocks_25: float | tuple[str, int] = 1.0,
        transformer_blocks_26: float | tuple[str, int] = 1.0,
        transformer_blocks_27: float | tuple[str, int] = 1.0,
        transformer_blocks_28: float | tuple[str, int] = 1.0,
        transformer_blocks_29: float | tuple[str, int] = 1.0,
        transformer_blocks_30: float | tuple[str, int] = 1.0,
        transformer_blocks_31: float | tuple[str, int] = 1.0,
        transformer_blocks_32: float | tuple[str, int] = 1.0,
        transformer_blocks_33: float | tuple[str, int] = 1.0,
        transformer_blocks_34: float | tuple[str, int] = 1.0,
        transformer_blocks_35: float | tuple[str, int] = 1.0,
        transformer_blocks_36: float | tuple[str, int] = 1.0,
        transformer_blocks_37: float | tuple[str, int] = 1.0,
        transformer_blocks_38: float | tuple[str, int] = 1.0,
        transformer_blocks_39: float | tuple[str, int] = 1.0,
        transformer_blocks_40: float | tuple[str, int] = 1.0,
        transformer_blocks_41: float | tuple[str, int] = 1.0,
        transformer_blocks_42: float | tuple[str, int] = 1.0,
        transformer_blocks_43: float | tuple[str, int] = 1.0,
        transformer_blocks_44: float | tuple[str, int] = 1.0,
        transformer_blocks_45: float | tuple[str, int] = 1.0,
        transformer_blocks_46: float | tuple[str, int] = 1.0,
        transformer_blocks_47: float | tuple[str, int] = 1.0,
        transformer_blocks_48: float | tuple[str, int] = 1.0,
        transformer_blocks_49: float | tuple[str, int] = 1.0,
        transformer_blocks_50: float | tuple[str, int] = 1.0,
        transformer_blocks_51: float | tuple[str, int] = 1.0,
        transformer_blocks_52: float | tuple[str, int] = 1.0,
        transformer_blocks_53: float | tuple[str, int] = 1.0,
        transformer_blocks_54: float | tuple[str, int] = 1.0,
        transformer_blocks_55: float | tuple[str, int] = 1.0,
        transformer_blocks_56: float | tuple[str, int] = 1.0,
        transformer_blocks_57: float | tuple[str, int] = 1.0,
        transformer_blocks_58: float | tuple[str, int] = 1.0,
        transformer_blocks_59: float | tuple[str, int] = 1.0,
        proj_out: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model1 = model1
        self.model2 = model2
        self.pos_embeds = pos_embeds
        self.img_in = img_in
        self.txt_norm = txt_norm
        self.txt_in = txt_in
        self.time_text_embed = time_text_embed
        self.transformer_blocks_0 = transformer_blocks_0
        self.transformer_blocks_1 = transformer_blocks_1
        self.transformer_blocks_2 = transformer_blocks_2
        self.transformer_blocks_3 = transformer_blocks_3
        self.transformer_blocks_4 = transformer_blocks_4
        self.transformer_blocks_5 = transformer_blocks_5
        self.transformer_blocks_6 = transformer_blocks_6
        self.transformer_blocks_7 = transformer_blocks_7
        self.transformer_blocks_8 = transformer_blocks_8
        self.transformer_blocks_9 = transformer_blocks_9
        self.transformer_blocks_10 = transformer_blocks_10
        self.transformer_blocks_11 = transformer_blocks_11
        self.transformer_blocks_12 = transformer_blocks_12
        self.transformer_blocks_13 = transformer_blocks_13
        self.transformer_blocks_14 = transformer_blocks_14
        self.transformer_blocks_15 = transformer_blocks_15
        self.transformer_blocks_16 = transformer_blocks_16
        self.transformer_blocks_17 = transformer_blocks_17
        self.transformer_blocks_18 = transformer_blocks_18
        self.transformer_blocks_19 = transformer_blocks_19
        self.transformer_blocks_20 = transformer_blocks_20
        self.transformer_blocks_21 = transformer_blocks_21
        self.transformer_blocks_22 = transformer_blocks_22
        self.transformer_blocks_23 = transformer_blocks_23
        self.transformer_blocks_24 = transformer_blocks_24
        self.transformer_blocks_25 = transformer_blocks_25
        self.transformer_blocks_26 = transformer_blocks_26
        self.transformer_blocks_27 = transformer_blocks_27
        self.transformer_blocks_28 = transformer_blocks_28
        self.transformer_blocks_29 = transformer_blocks_29
        self.transformer_blocks_30 = transformer_blocks_30
        self.transformer_blocks_31 = transformer_blocks_31
        self.transformer_blocks_32 = transformer_blocks_32
        self.transformer_blocks_33 = transformer_blocks_33
        self.transformer_blocks_34 = transformer_blocks_34
        self.transformer_blocks_35 = transformer_blocks_35
        self.transformer_blocks_36 = transformer_blocks_36
        self.transformer_blocks_37 = transformer_blocks_37
        self.transformer_blocks_38 = transformer_blocks_38
        self.transformer_blocks_39 = transformer_blocks_39
        self.transformer_blocks_40 = transformer_blocks_40
        self.transformer_blocks_41 = transformer_blocks_41
        self.transformer_blocks_42 = transformer_blocks_42
        self.transformer_blocks_43 = transformer_blocks_43
        self.transformer_blocks_44 = transformer_blocks_44
        self.transformer_blocks_45 = transformer_blocks_45
        self.transformer_blocks_46 = transformer_blocks_46
        self.transformer_blocks_47 = transformer_blocks_47
        self.transformer_blocks_48 = transformer_blocks_48
        self.transformer_blocks_49 = transformer_blocks_49
        self.transformer_blocks_50 = transformer_blocks_50
        self.transformer_blocks_51 = transformer_blocks_51
        self.transformer_blocks_52 = transformer_blocks_52
        self.transformer_blocks_53 = transformer_blocks_53
        self.transformer_blocks_54 = transformer_blocks_54
        self.transformer_blocks_55 = transformer_blocks_55
        self.transformer_blocks_56 = transformer_blocks_56
        self.transformer_blocks_57 = transformer_blocks_57
        self.transformer_blocks_58 = transformer_blocks_58
        self.transformer_blocks_59 = transformer_blocks_59
        self.proj_out = proj_out
        self.__validate()

    def __validate(self) -> None:
        validate_range("pos_embeds", self.pos_embeds, 0.0, 1.0)
        validate_range("img_in", self.img_in, 0.0, 1.0)
        validate_range("txt_norm", self.txt_norm, 0.0, 1.0)
        validate_range("txt_in", self.txt_in, 0.0, 1.0)
        validate_range("time_text_embed", self.time_text_embed, 0.0, 1.0)
        validate_range("transformer_blocks_0", self.transformer_blocks_0, 0.0, 1.0)
        validate_range("transformer_blocks_1", self.transformer_blocks_1, 0.0, 1.0)
        validate_range("transformer_blocks_2", self.transformer_blocks_2, 0.0, 1.0)
        validate_range("transformer_blocks_3", self.transformer_blocks_3, 0.0, 1.0)
        validate_range("transformer_blocks_4", self.transformer_blocks_4, 0.0, 1.0)
        validate_range("transformer_blocks_5", self.transformer_blocks_5, 0.0, 1.0)
        validate_range("transformer_blocks_6", self.transformer_blocks_6, 0.0, 1.0)
        validate_range("transformer_blocks_7", self.transformer_blocks_7, 0.0, 1.0)
        validate_range("transformer_blocks_8", self.transformer_blocks_8, 0.0, 1.0)
        validate_range("transformer_blocks_9", self.transformer_blocks_9, 0.0, 1.0)
        validate_range("transformer_blocks_10", self.transformer_blocks_10, 0.0, 1.0)
        validate_range("transformer_blocks_11", self.transformer_blocks_11, 0.0, 1.0)
        validate_range("transformer_blocks_12", self.transformer_blocks_12, 0.0, 1.0)
        validate_range("transformer_blocks_13", self.transformer_blocks_13, 0.0, 1.0)
        validate_range("transformer_blocks_14", self.transformer_blocks_14, 0.0, 1.0)
        validate_range("transformer_blocks_15", self.transformer_blocks_15, 0.0, 1.0)
        validate_range("transformer_blocks_16", self.transformer_blocks_16, 0.0, 1.0)
        validate_range("transformer_blocks_17", self.transformer_blocks_17, 0.0, 1.0)
        validate_range("transformer_blocks_18", self.transformer_blocks_18, 0.0, 1.0)
        validate_range("transformer_blocks_19", self.transformer_blocks_19, 0.0, 1.0)
        validate_range("transformer_blocks_20", self.transformer_blocks_20, 0.0, 1.0)
        validate_range("transformer_blocks_21", self.transformer_blocks_21, 0.0, 1.0)
        validate_range("transformer_blocks_22", self.transformer_blocks_22, 0.0, 1.0)
        validate_range("transformer_blocks_23", self.transformer_blocks_23, 0.0, 1.0)
        validate_range("transformer_blocks_24", self.transformer_blocks_24, 0.0, 1.0)
        validate_range("transformer_blocks_25", self.transformer_blocks_25, 0.0, 1.0)
        validate_range("transformer_blocks_26", self.transformer_blocks_26, 0.0, 1.0)
        validate_range("transformer_blocks_27", self.transformer_blocks_27, 0.0, 1.0)
        validate_range("transformer_blocks_28", self.transformer_blocks_28, 0.0, 1.0)
        validate_range("transformer_blocks_29", self.transformer_blocks_29, 0.0, 1.0)
        validate_range("transformer_blocks_30", self.transformer_blocks_30, 0.0, 1.0)
        validate_range("transformer_blocks_31", self.transformer_blocks_31, 0.0, 1.0)
        validate_range("transformer_blocks_32", self.transformer_blocks_32, 0.0, 1.0)
        validate_range("transformer_blocks_33", self.transformer_blocks_33, 0.0, 1.0)
        validate_range("transformer_blocks_34", self.transformer_blocks_34, 0.0, 1.0)
        validate_range("transformer_blocks_35", self.transformer_blocks_35, 0.0, 1.0)
        validate_range("transformer_blocks_36", self.transformer_blocks_36, 0.0, 1.0)
        validate_range("transformer_blocks_37", self.transformer_blocks_37, 0.0, 1.0)
        validate_range("transformer_blocks_38", self.transformer_blocks_38, 0.0, 1.0)
        validate_range("transformer_blocks_39", self.transformer_blocks_39, 0.0, 1.0)
        validate_range("transformer_blocks_40", self.transformer_blocks_40, 0.0, 1.0)
        validate_range("transformer_blocks_41", self.transformer_blocks_41, 0.0, 1.0)
        validate_range("transformer_blocks_42", self.transformer_blocks_42, 0.0, 1.0)
        validate_range("transformer_blocks_43", self.transformer_blocks_43, 0.0, 1.0)
        validate_range("transformer_blocks_44", self.transformer_blocks_44, 0.0, 1.0)
        validate_range("transformer_blocks_45", self.transformer_blocks_45, 0.0, 1.0)
        validate_range("transformer_blocks_46", self.transformer_blocks_46, 0.0, 1.0)
        validate_range("transformer_blocks_47", self.transformer_blocks_47, 0.0, 1.0)
        validate_range("transformer_blocks_48", self.transformer_blocks_48, 0.0, 1.0)
        validate_range("transformer_blocks_49", self.transformer_blocks_49, 0.0, 1.0)
        validate_range("transformer_blocks_50", self.transformer_blocks_50, 0.0, 1.0)
        validate_range("transformer_blocks_51", self.transformer_blocks_51, 0.0, 1.0)
        validate_range("transformer_blocks_52", self.transformer_blocks_52, 0.0, 1.0)
        validate_range("transformer_blocks_53", self.transformer_blocks_53, 0.0, 1.0)
        validate_range("transformer_blocks_54", self.transformer_blocks_54, 0.0, 1.0)
        validate_range("transformer_blocks_55", self.transformer_blocks_55, 0.0, 1.0)
        validate_range("transformer_blocks_56", self.transformer_blocks_56, 0.0, 1.0)
        validate_range("transformer_blocks_57", self.transformer_blocks_57, 0.0, 1.0)
        validate_range("transformer_blocks_58", self.transformer_blocks_58, 0.0, 1.0)
        validate_range("transformer_blocks_59", self.transformer_blocks_59, 0.0, 1.0)
        validate_range("proj_out", self.proj_out, 0.0, 1.0)


class ModelMergeKrea2(BaseNode):
    __slots__ = (
        "model1",
        "model2",
        "first",
        "tmlp",
        "txtmlp",
        "tproj",
        "txtfusion_layerwise_blocks_0",
        "txtfusion_layerwise_blocks_1",
        "txtfusion_projector",
        "txtfusion_refiner_blocks_0",
        "txtfusion_refiner_blocks_1",
        "blocks_0",
        "blocks_1",
        "blocks_2",
        "blocks_3",
        "blocks_4",
        "blocks_5",
        "blocks_6",
        "blocks_7",
        "blocks_8",
        "blocks_9",
        "blocks_10",
        "blocks_11",
        "blocks_12",
        "blocks_13",
        "blocks_14",
        "blocks_15",
        "blocks_16",
        "blocks_17",
        "blocks_18",
        "blocks_19",
        "blocks_20",
        "blocks_21",
        "blocks_22",
        "blocks_23",
        "blocks_24",
        "blocks_25",
        "blocks_26",
        "blocks_27",
        "last",
    )
    CATEGORY = "model/merging/model specific"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {
        "model1": "MODEL",
        "model2": "MODEL",
        "first": "FLOAT",
        "tmlp": "FLOAT",
        "txtmlp": "FLOAT",
        "tproj": "FLOAT",
        "txtfusion_layerwise_blocks_0": "FLOAT",
        "txtfusion_layerwise_blocks_1": "FLOAT",
        "txtfusion_projector": "FLOAT",
        "txtfusion_refiner_blocks_0": "FLOAT",
        "txtfusion_refiner_blocks_1": "FLOAT",
        "blocks_0": "FLOAT",
        "blocks_1": "FLOAT",
        "blocks_2": "FLOAT",
        "blocks_3": "FLOAT",
        "blocks_4": "FLOAT",
        "blocks_5": "FLOAT",
        "blocks_6": "FLOAT",
        "blocks_7": "FLOAT",
        "blocks_8": "FLOAT",
        "blocks_9": "FLOAT",
        "blocks_10": "FLOAT",
        "blocks_11": "FLOAT",
        "blocks_12": "FLOAT",
        "blocks_13": "FLOAT",
        "blocks_14": "FLOAT",
        "blocks_15": "FLOAT",
        "blocks_16": "FLOAT",
        "blocks_17": "FLOAT",
        "blocks_18": "FLOAT",
        "blocks_19": "FLOAT",
        "blocks_20": "FLOAT",
        "blocks_21": "FLOAT",
        "blocks_22": "FLOAT",
        "blocks_23": "FLOAT",
        "blocks_24": "FLOAT",
        "blocks_25": "FLOAT",
        "blocks_26": "FLOAT",
        "blocks_27": "FLOAT",
        "last": "FLOAT",
    }
    INPUT_NAMES = {
        "first": "first.",
        "tmlp": "tmlp.",
        "txtmlp": "txtmlp.",
        "tproj": "tproj.",
        "txtfusion_layerwise_blocks_0": "txtfusion.layerwise_blocks.0.",
        "txtfusion_layerwise_blocks_1": "txtfusion.layerwise_blocks.1.",
        "txtfusion_projector": "txtfusion.projector.",
        "txtfusion_refiner_blocks_0": "txtfusion.refiner_blocks.0.",
        "txtfusion_refiner_blocks_1": "txtfusion.refiner_blocks.1.",
        "blocks_0": "blocks.0.",
        "blocks_1": "blocks.1.",
        "blocks_2": "blocks.2.",
        "blocks_3": "blocks.3.",
        "blocks_4": "blocks.4.",
        "blocks_5": "blocks.5.",
        "blocks_6": "blocks.6.",
        "blocks_7": "blocks.7.",
        "blocks_8": "blocks.8.",
        "blocks_9": "blocks.9.",
        "blocks_10": "blocks.10.",
        "blocks_11": "blocks.11.",
        "blocks_12": "blocks.12.",
        "blocks_13": "blocks.13.",
        "blocks_14": "blocks.14.",
        "blocks_15": "blocks.15.",
        "blocks_16": "blocks.16.",
        "blocks_17": "blocks.17.",
        "blocks_18": "blocks.18.",
        "blocks_19": "blocks.19.",
        "blocks_20": "blocks.20.",
        "blocks_21": "blocks.21.",
        "blocks_22": "blocks.22.",
        "blocks_23": "blocks.23.",
        "blocks_24": "blocks.24.",
        "blocks_25": "blocks.25.",
        "blocks_26": "blocks.26.",
        "blocks_27": "blocks.27.",
        "last": "last.",
    }

    def __init__(
        self,
        node_id: str,
        model1: tuple[str, int],
        model2: tuple[str, int],
        first: float | tuple[str, int] = 1.0,
        tmlp: float | tuple[str, int] = 1.0,
        txtmlp: float | tuple[str, int] = 1.0,
        tproj: float | tuple[str, int] = 1.0,
        txtfusion_layerwise_blocks_0: float | tuple[str, int] = 1.0,
        txtfusion_layerwise_blocks_1: float | tuple[str, int] = 1.0,
        txtfusion_projector: float | tuple[str, int] = 1.0,
        txtfusion_refiner_blocks_0: float | tuple[str, int] = 1.0,
        txtfusion_refiner_blocks_1: float | tuple[str, int] = 1.0,
        blocks_0: float | tuple[str, int] = 1.0,
        blocks_1: float | tuple[str, int] = 1.0,
        blocks_2: float | tuple[str, int] = 1.0,
        blocks_3: float | tuple[str, int] = 1.0,
        blocks_4: float | tuple[str, int] = 1.0,
        blocks_5: float | tuple[str, int] = 1.0,
        blocks_6: float | tuple[str, int] = 1.0,
        blocks_7: float | tuple[str, int] = 1.0,
        blocks_8: float | tuple[str, int] = 1.0,
        blocks_9: float | tuple[str, int] = 1.0,
        blocks_10: float | tuple[str, int] = 1.0,
        blocks_11: float | tuple[str, int] = 1.0,
        blocks_12: float | tuple[str, int] = 1.0,
        blocks_13: float | tuple[str, int] = 1.0,
        blocks_14: float | tuple[str, int] = 1.0,
        blocks_15: float | tuple[str, int] = 1.0,
        blocks_16: float | tuple[str, int] = 1.0,
        blocks_17: float | tuple[str, int] = 1.0,
        blocks_18: float | tuple[str, int] = 1.0,
        blocks_19: float | tuple[str, int] = 1.0,
        blocks_20: float | tuple[str, int] = 1.0,
        blocks_21: float | tuple[str, int] = 1.0,
        blocks_22: float | tuple[str, int] = 1.0,
        blocks_23: float | tuple[str, int] = 1.0,
        blocks_24: float | tuple[str, int] = 1.0,
        blocks_25: float | tuple[str, int] = 1.0,
        blocks_26: float | tuple[str, int] = 1.0,
        blocks_27: float | tuple[str, int] = 1.0,
        last: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model1 = model1
        self.model2 = model2
        self.first = first
        self.tmlp = tmlp
        self.txtmlp = txtmlp
        self.tproj = tproj
        self.txtfusion_layerwise_blocks_0 = txtfusion_layerwise_blocks_0
        self.txtfusion_layerwise_blocks_1 = txtfusion_layerwise_blocks_1
        self.txtfusion_projector = txtfusion_projector
        self.txtfusion_refiner_blocks_0 = txtfusion_refiner_blocks_0
        self.txtfusion_refiner_blocks_1 = txtfusion_refiner_blocks_1
        self.blocks_0 = blocks_0
        self.blocks_1 = blocks_1
        self.blocks_2 = blocks_2
        self.blocks_3 = blocks_3
        self.blocks_4 = blocks_4
        self.blocks_5 = blocks_5
        self.blocks_6 = blocks_6
        self.blocks_7 = blocks_7
        self.blocks_8 = blocks_8
        self.blocks_9 = blocks_9
        self.blocks_10 = blocks_10
        self.blocks_11 = blocks_11
        self.blocks_12 = blocks_12
        self.blocks_13 = blocks_13
        self.blocks_14 = blocks_14
        self.blocks_15 = blocks_15
        self.blocks_16 = blocks_16
        self.blocks_17 = blocks_17
        self.blocks_18 = blocks_18
        self.blocks_19 = blocks_19
        self.blocks_20 = blocks_20
        self.blocks_21 = blocks_21
        self.blocks_22 = blocks_22
        self.blocks_23 = blocks_23
        self.blocks_24 = blocks_24
        self.blocks_25 = blocks_25
        self.blocks_26 = blocks_26
        self.blocks_27 = blocks_27
        self.last = last
        self.__validate()

    def __validate(self) -> None:
        validate_range("first", self.first, 0.0, 1.0)
        validate_range("tmlp", self.tmlp, 0.0, 1.0)
        validate_range("txtmlp", self.txtmlp, 0.0, 1.0)
        validate_range("tproj", self.tproj, 0.0, 1.0)
        validate_range(
            "txtfusion_layerwise_blocks_0", self.txtfusion_layerwise_blocks_0, 0.0, 1.0
        )
        validate_range(
            "txtfusion_layerwise_blocks_1", self.txtfusion_layerwise_blocks_1, 0.0, 1.0
        )
        validate_range("txtfusion_projector", self.txtfusion_projector, 0.0, 1.0)
        validate_range(
            "txtfusion_refiner_blocks_0", self.txtfusion_refiner_blocks_0, 0.0, 1.0
        )
        validate_range(
            "txtfusion_refiner_blocks_1", self.txtfusion_refiner_blocks_1, 0.0, 1.0
        )
        validate_range("blocks_0", self.blocks_0, 0.0, 1.0)
        validate_range("blocks_1", self.blocks_1, 0.0, 1.0)
        validate_range("blocks_2", self.blocks_2, 0.0, 1.0)
        validate_range("blocks_3", self.blocks_3, 0.0, 1.0)
        validate_range("blocks_4", self.blocks_4, 0.0, 1.0)
        validate_range("blocks_5", self.blocks_5, 0.0, 1.0)
        validate_range("blocks_6", self.blocks_6, 0.0, 1.0)
        validate_range("blocks_7", self.blocks_7, 0.0, 1.0)
        validate_range("blocks_8", self.blocks_8, 0.0, 1.0)
        validate_range("blocks_9", self.blocks_9, 0.0, 1.0)
        validate_range("blocks_10", self.blocks_10, 0.0, 1.0)
        validate_range("blocks_11", self.blocks_11, 0.0, 1.0)
        validate_range("blocks_12", self.blocks_12, 0.0, 1.0)
        validate_range("blocks_13", self.blocks_13, 0.0, 1.0)
        validate_range("blocks_14", self.blocks_14, 0.0, 1.0)
        validate_range("blocks_15", self.blocks_15, 0.0, 1.0)
        validate_range("blocks_16", self.blocks_16, 0.0, 1.0)
        validate_range("blocks_17", self.blocks_17, 0.0, 1.0)
        validate_range("blocks_18", self.blocks_18, 0.0, 1.0)
        validate_range("blocks_19", self.blocks_19, 0.0, 1.0)
        validate_range("blocks_20", self.blocks_20, 0.0, 1.0)
        validate_range("blocks_21", self.blocks_21, 0.0, 1.0)
        validate_range("blocks_22", self.blocks_22, 0.0, 1.0)
        validate_range("blocks_23", self.blocks_23, 0.0, 1.0)
        validate_range("blocks_24", self.blocks_24, 0.0, 1.0)
        validate_range("blocks_25", self.blocks_25, 0.0, 1.0)
        validate_range("blocks_26", self.blocks_26, 0.0, 1.0)
        validate_range("blocks_27", self.blocks_27, 0.0, 1.0)
        validate_range("last", self.last, 0.0, 1.0)
