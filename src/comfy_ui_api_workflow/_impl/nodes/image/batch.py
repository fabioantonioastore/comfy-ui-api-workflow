from ..base import BaseNode
from ..validation import validate_range


class RebatchImages(BaseNode):
    __slots__ = ("images", "batch_size")
    CATEGORY = "image/batch"
    OUTPUTS = {"IMAGE": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_SOCKET_TYPES = {"images": "IMAGE", "batch_size": "INT"}

    def __init__(
        self,
        node_id: str,
        images: tuple[str, int],
        batch_size: int | tuple[str, int] = 1,
    ) -> None:
        super().__init__(node_id=node_id)
        self.images = images
        self.batch_size = batch_size
        self.__validate()

    def __validate(self) -> None:
        validate_range("batch_size", self.batch_size, 1, 4096)


class RepeatImageBatch(BaseNode):
    __slots__ = ("image", "amount")
    CATEGORY = "image/batch"
    OUTPUTS = {"IMAGE": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_SOCKET_TYPES = {"image": "IMAGE", "amount": "INT"}

    def __init__(
        self, node_id: str, image: tuple[str, int], amount: int | tuple[str, int] = 1
    ) -> None:
        super().__init__(node_id=node_id)
        self.image = image
        self.amount = amount
        self.__validate()

    def __validate(self) -> None:
        validate_range("amount", self.amount, 1, 4096)


class ImageFromBatch(BaseNode):
    __slots__ = ("image", "batch_index", "length")
    CATEGORY = "image/batch"
    OUTPUTS = {"IMAGE": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_SOCKET_TYPES = {"image": "IMAGE", "batch_index": "INT", "length": "INT"}

    def __init__(
        self,
        node_id: str,
        image: tuple[str, int],
        batch_index: int | tuple[str, int] = 0,
        length: int | tuple[str, int] = 1,
    ) -> None:
        super().__init__(node_id=node_id)
        self.image = image
        self.batch_index = batch_index
        self.length = length
        self.__validate()

    def __validate(self) -> None:
        validate_range("batch_index", self.batch_index, -16_384, 16_384)
        validate_range("length", self.length, 1, 4096)


class SplitImageToTileList(BaseNode):
    __slots__ = ("image", "tile_width", "tile_height", "overlap")
    CATEGORY = "image/batch"
    OUTPUTS = {"IMAGE": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_SOCKET_TYPES = {
        "image": "IMAGE",
        "tile_width": "INT",
        "tile_height": "INT",
        "overlap": "INT",
    }

    def __init__(
        self,
        node_id: str,
        image: tuple[str, int],
        tile_width: int | tuple[str, int] = 1024,
        tile_height: int | tuple[str, int] = 1024,
        overlap: int | tuple[str, int] = 128,
    ) -> None:
        super().__init__(node_id=node_id)
        self.image = image
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.overlap = overlap
        self.__validate()

    def __validate(self) -> None:
        validate_range("tile_width", self.tile_width, 64, 16_384)
        validate_range("tile_height", self.tile_height, 64, 16_384)
        validate_range("overlap", self.overlap, 0, 4096)


class ImageMergeTileList(BaseNode):
    __slots__ = ("image_list", "final_width", "final_height", "overlap")
    CATEGORY = "image/batch"
    OUTPUTS = {"IMAGE": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_SOCKET_TYPES = {
        "image_list": "IMAGE",
        "final_width": "INT",
        "final_height": "INT",
        "overlap": "INT",
    }

    def __init__(
        self,
        node_id: str,
        image_list: tuple[str, int],
        final_width: int | tuple[str, int] = 1024,
        final_height: int | tuple[str, int] = 1024,
        overlap: int | tuple[str, int] = 128,
    ) -> None:
        super().__init__(node_id=node_id)
        self.image_list = image_list
        self.final_width = final_width
        self.final_height = final_height
        self.overlap = overlap
        self.__validate()

    def __validate(self) -> None:
        validate_range("final_width", self.final_width, 64, 32_768)
        validate_range("final_height", self.final_height, 64, 32_768)
        validate_range("overlap", self.overlap, 0, 4096)


class ShuffleDataset(BaseNode):
    __slots__ = ("images", "seed")
    CATEGORY = "image/batch"
    OUTPUTS = {"images": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_SOCKET_TYPES = {"images": "IMAGE", "seed": "INT"}

    def __init__(
        self, node_id: str, images: tuple[str, int], seed: int | tuple[str, int] = 0
    ) -> None:
        super().__init__(node_id=node_id)
        self.images = images
        self.seed = seed
        self.__validate()

    def __validate(self) -> None:
        validate_range("seed", self.seed, 0, 18_446_744_073_709_551_615)


class ShuffleImageTextDataset(BaseNode):
    __slots__ = ("images", "texts", "seed")
    CATEGORY = "image/batch"
    OUTPUTS = {"images": 0, "texts": 1}
    RETURN_TYPES = ("IMAGE", "STRING")
    INPUT_SOCKET_TYPES = {"images": "IMAGE", "texts": "STRING", "seed": "INT"}

    def __init__(
        self,
        node_id: str,
        images: tuple[str, int],
        texts: tuple[str, int],
        seed: int | tuple[str, int] = 0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.images = images
        self.texts = texts
        self.seed = seed
        self.__validate()

    def __validate(self) -> None:
        validate_range("seed", self.seed, 0, 18_446_744_073_709_551_615)


class ImageDeduplication(BaseNode):
    __slots__ = ("images", "similarity_threshold")
    CATEGORY = "image/batch"
    OUTPUTS = {"images": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_SOCKET_TYPES = {"images": "IMAGE", "similarity_threshold": "FLOAT"}

    def __init__(
        self,
        node_id: str,
        images: tuple[str, int],
        similarity_threshold: float | tuple[str, int] = 0.95,
    ) -> None:
        super().__init__(node_id=node_id)
        self.images = images
        self.similarity_threshold = similarity_threshold
        self.__validate()

    def __validate(self) -> None:
        validate_range("similarity_threshold", self.similarity_threshold, 0.0, 1.0)


class ImageGrid(BaseNode):
    __slots__ = ("images", "columns", "cell_width", "cell_height", "padding")
    CATEGORY = "image/batch"
    OUTPUTS = {"images": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_SOCKET_TYPES = {
        "images": "IMAGE",
        "columns": "INT",
        "cell_width": "INT",
        "cell_height": "INT",
        "padding": "INT",
    }

    def __init__(
        self,
        node_id: str,
        images: tuple[str, int],
        columns: int | tuple[str, int] = 4,
        cell_width: int | tuple[str, int] = 256,
        cell_height: int | tuple[str, int] = 256,
        padding: int | tuple[str, int] = 4,
    ) -> None:
        super().__init__(node_id=node_id)
        self.images = images
        self.columns = columns
        self.cell_width = cell_width
        self.cell_height = cell_height
        self.padding = padding
        self.__validate()

    def __validate(self) -> None:
        validate_range("columns", self.columns, 1, 20)
        validate_range("cell_width", self.cell_width, 32, 2048)
        validate_range("cell_height", self.cell_height, 32, 2048)
        validate_range("padding", self.padding, 0, 50)
