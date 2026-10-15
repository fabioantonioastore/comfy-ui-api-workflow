from ..base import BaseNode
from ..base import NodeOutput
from ...enums import OutputTypes


class CheckpointLoaderSimple(BaseNode):
    __slots__ = ("ckpt_name", "output")
    CATEGORY = "models/loaders"

    def __init__(self, node_id: str, ckpt_name: str) -> None:
        super().__init__(node_id=node_id)
        self.ckpt_name = ckpt_name
        self.output = NodeOutput(
            node_id=self.node_id,
            outputs={OutputTypes.MODEL: 0, OutputTypes.CLIP: 1, OutputTypes.VAE: 2},
        )
