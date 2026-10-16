from ...base import BaseNode


class Hunyuan3Dv2Conditioning(BaseNode):
    __slots__ = ("clip_vision_output",)
    CATEGORY = "model/conditioning/hunyuan 3d"
    OUTPUTS = {"positive": 0, "negative": 1}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    INPUT_SOCKET_TYPES = {"clip_vision_output": "CLIP_VISION_OUTPUT"}

    def __init__(self, node_id: str, clip_vision_output: tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.clip_vision_output = clip_vision_output


class Hunyuan3Dv2ConditioningMultiView(BaseNode):
    __slots__ = ("front", "left", "back", "right")
    CATEGORY = "model/conditioning/hunyuan 3d"
    OUTPUTS = {"positive": 0, "negative": 1}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    INPUT_SOCKET_TYPES = {
        "front": "CLIP_VISION_OUTPUT",
        "left": "CLIP_VISION_OUTPUT",
        "back": "CLIP_VISION_OUTPUT",
        "right": "CLIP_VISION_OUTPUT",
    }

    def __init__(
        self,
        node_id: str,
        front: tuple[str, int] | None = None,
        left: tuple[str, int] | None = None,
        back: tuple[str, int] | None = None,
        right: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        if front is not None:
            self.front = front
        if left is not None:
            self.left = left
        if back is not None:
            self.back = back
        if right is not None:
            self.right = right
