from ...base import BaseNode


class ByteDanceCreateImageAsset(BaseNode):
    __slots__ = ("image", "group_id")
    CATEGORY = "partner/image/ByteDance"
    OUTPUTS = {"asset_id": 0, "group_id": 1}
    RETURN_TYPES = ("STRING", "STRING")
    INPUT_SOCKET_TYPES = {"image": "IMAGE", "group_id": "STRING"}

    def __init__(
        self, node_id: str, image: tuple[str, int], group_id: str | tuple[str, int] = ""
    ) -> None:
        super().__init__(node_id=node_id)
        self.image = image
        self.group_id = group_id
