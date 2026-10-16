from ...base import BaseNode


class ByteDanceCreateVideoAsset(BaseNode):
    __slots__ = ("video", "group_id")
    CATEGORY = "partner/video/ByteDance"
    OUTPUTS = {"asset_id": 0, "group_id": 1}
    RETURN_TYPES = ("STRING", "STRING")
    INPUT_SOCKET_TYPES = {"video": "VIDEO", "group_id": "STRING"}

    def __init__(
        self, node_id: str, video: tuple[str, int], group_id: str | tuple[str, int] = ""
    ) -> None:
        super().__init__(node_id=node_id)
        self.video = video
        self.group_id = group_id
