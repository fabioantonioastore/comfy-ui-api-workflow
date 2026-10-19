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


class BatchImagesNode(BaseNode):
    __slots__ = (
        "images_image0",
        "images_image1",
        "images_image2",
        "images_image3",
        "images_image4",
        "images_image5",
        "images_image6",
        "images_image7",
        "images_image8",
        "images_image9",
        "images_image10",
        "images_image11",
        "images_image12",
        "images_image13",
        "images_image14",
        "images_image15",
        "images_image16",
        "images_image17",
        "images_image18",
        "images_image19",
        "images_image20",
        "images_image21",
        "images_image22",
        "images_image23",
        "images_image24",
        "images_image25",
        "images_image26",
        "images_image27",
        "images_image28",
        "images_image29",
        "images_image30",
        "images_image31",
        "images_image32",
        "images_image33",
        "images_image34",
        "images_image35",
        "images_image36",
        "images_image37",
        "images_image38",
        "images_image39",
        "images_image40",
        "images_image41",
        "images_image42",
        "images_image43",
        "images_image44",
        "images_image45",
        "images_image46",
        "images_image47",
        "images_image48",
        "images_image49",
    )
    CATEGORY = "image/batch"
    OUTPUTS = {"IMAGE": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_NAMES = {
        "images_image0": "images.image0",
        "images_image1": "images.image1",
        "images_image2": "images.image2",
        "images_image3": "images.image3",
        "images_image4": "images.image4",
        "images_image5": "images.image5",
        "images_image6": "images.image6",
        "images_image7": "images.image7",
        "images_image8": "images.image8",
        "images_image9": "images.image9",
        "images_image10": "images.image10",
        "images_image11": "images.image11",
        "images_image12": "images.image12",
        "images_image13": "images.image13",
        "images_image14": "images.image14",
        "images_image15": "images.image15",
        "images_image16": "images.image16",
        "images_image17": "images.image17",
        "images_image18": "images.image18",
        "images_image19": "images.image19",
        "images_image20": "images.image20",
        "images_image21": "images.image21",
        "images_image22": "images.image22",
        "images_image23": "images.image23",
        "images_image24": "images.image24",
        "images_image25": "images.image25",
        "images_image26": "images.image26",
        "images_image27": "images.image27",
        "images_image28": "images.image28",
        "images_image29": "images.image29",
        "images_image30": "images.image30",
        "images_image31": "images.image31",
        "images_image32": "images.image32",
        "images_image33": "images.image33",
        "images_image34": "images.image34",
        "images_image35": "images.image35",
        "images_image36": "images.image36",
        "images_image37": "images.image37",
        "images_image38": "images.image38",
        "images_image39": "images.image39",
        "images_image40": "images.image40",
        "images_image41": "images.image41",
        "images_image42": "images.image42",
        "images_image43": "images.image43",
        "images_image44": "images.image44",
        "images_image45": "images.image45",
        "images_image46": "images.image46",
        "images_image47": "images.image47",
        "images_image48": "images.image48",
        "images_image49": "images.image49",
    }
    INPUT_SOCKET_TYPES = {
        "images_image0": "IMAGE",
        "images_image1": "IMAGE",
        "images_image2": "IMAGE",
        "images_image3": "IMAGE",
        "images_image4": "IMAGE",
        "images_image5": "IMAGE",
        "images_image6": "IMAGE",
        "images_image7": "IMAGE",
        "images_image8": "IMAGE",
        "images_image9": "IMAGE",
        "images_image10": "IMAGE",
        "images_image11": "IMAGE",
        "images_image12": "IMAGE",
        "images_image13": "IMAGE",
        "images_image14": "IMAGE",
        "images_image15": "IMAGE",
        "images_image16": "IMAGE",
        "images_image17": "IMAGE",
        "images_image18": "IMAGE",
        "images_image19": "IMAGE",
        "images_image20": "IMAGE",
        "images_image21": "IMAGE",
        "images_image22": "IMAGE",
        "images_image23": "IMAGE",
        "images_image24": "IMAGE",
        "images_image25": "IMAGE",
        "images_image26": "IMAGE",
        "images_image27": "IMAGE",
        "images_image28": "IMAGE",
        "images_image29": "IMAGE",
        "images_image30": "IMAGE",
        "images_image31": "IMAGE",
        "images_image32": "IMAGE",
        "images_image33": "IMAGE",
        "images_image34": "IMAGE",
        "images_image35": "IMAGE",
        "images_image36": "IMAGE",
        "images_image37": "IMAGE",
        "images_image38": "IMAGE",
        "images_image39": "IMAGE",
        "images_image40": "IMAGE",
        "images_image41": "IMAGE",
        "images_image42": "IMAGE",
        "images_image43": "IMAGE",
        "images_image44": "IMAGE",
        "images_image45": "IMAGE",
        "images_image46": "IMAGE",
        "images_image47": "IMAGE",
        "images_image48": "IMAGE",
        "images_image49": "IMAGE",
    }

    def __init__(
        self,
        node_id: str,
        images_image0: tuple[str, int],
        images_image1: tuple[str, int] | None = None,
        images_image2: tuple[str, int] | None = None,
        images_image3: tuple[str, int] | None = None,
        images_image4: tuple[str, int] | None = None,
        images_image5: tuple[str, int] | None = None,
        images_image6: tuple[str, int] | None = None,
        images_image7: tuple[str, int] | None = None,
        images_image8: tuple[str, int] | None = None,
        images_image9: tuple[str, int] | None = None,
        images_image10: tuple[str, int] | None = None,
        images_image11: tuple[str, int] | None = None,
        images_image12: tuple[str, int] | None = None,
        images_image13: tuple[str, int] | None = None,
        images_image14: tuple[str, int] | None = None,
        images_image15: tuple[str, int] | None = None,
        images_image16: tuple[str, int] | None = None,
        images_image17: tuple[str, int] | None = None,
        images_image18: tuple[str, int] | None = None,
        images_image19: tuple[str, int] | None = None,
        images_image20: tuple[str, int] | None = None,
        images_image21: tuple[str, int] | None = None,
        images_image22: tuple[str, int] | None = None,
        images_image23: tuple[str, int] | None = None,
        images_image24: tuple[str, int] | None = None,
        images_image25: tuple[str, int] | None = None,
        images_image26: tuple[str, int] | None = None,
        images_image27: tuple[str, int] | None = None,
        images_image28: tuple[str, int] | None = None,
        images_image29: tuple[str, int] | None = None,
        images_image30: tuple[str, int] | None = None,
        images_image31: tuple[str, int] | None = None,
        images_image32: tuple[str, int] | None = None,
        images_image33: tuple[str, int] | None = None,
        images_image34: tuple[str, int] | None = None,
        images_image35: tuple[str, int] | None = None,
        images_image36: tuple[str, int] | None = None,
        images_image37: tuple[str, int] | None = None,
        images_image38: tuple[str, int] | None = None,
        images_image39: tuple[str, int] | None = None,
        images_image40: tuple[str, int] | None = None,
        images_image41: tuple[str, int] | None = None,
        images_image42: tuple[str, int] | None = None,
        images_image43: tuple[str, int] | None = None,
        images_image44: tuple[str, int] | None = None,
        images_image45: tuple[str, int] | None = None,
        images_image46: tuple[str, int] | None = None,
        images_image47: tuple[str, int] | None = None,
        images_image48: tuple[str, int] | None = None,
        images_image49: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.images_image0 = images_image0
        if images_image1 is not None:
            self.images_image1 = images_image1
        if images_image2 is not None:
            self.images_image2 = images_image2
        if images_image3 is not None:
            self.images_image3 = images_image3
        if images_image4 is not None:
            self.images_image4 = images_image4
        if images_image5 is not None:
            self.images_image5 = images_image5
        if images_image6 is not None:
            self.images_image6 = images_image6
        if images_image7 is not None:
            self.images_image7 = images_image7
        if images_image8 is not None:
            self.images_image8 = images_image8
        if images_image9 is not None:
            self.images_image9 = images_image9
        if images_image10 is not None:
            self.images_image10 = images_image10
        if images_image11 is not None:
            self.images_image11 = images_image11
        if images_image12 is not None:
            self.images_image12 = images_image12
        if images_image13 is not None:
            self.images_image13 = images_image13
        if images_image14 is not None:
            self.images_image14 = images_image14
        if images_image15 is not None:
            self.images_image15 = images_image15
        if images_image16 is not None:
            self.images_image16 = images_image16
        if images_image17 is not None:
            self.images_image17 = images_image17
        if images_image18 is not None:
            self.images_image18 = images_image18
        if images_image19 is not None:
            self.images_image19 = images_image19
        if images_image20 is not None:
            self.images_image20 = images_image20
        if images_image21 is not None:
            self.images_image21 = images_image21
        if images_image22 is not None:
            self.images_image22 = images_image22
        if images_image23 is not None:
            self.images_image23 = images_image23
        if images_image24 is not None:
            self.images_image24 = images_image24
        if images_image25 is not None:
            self.images_image25 = images_image25
        if images_image26 is not None:
            self.images_image26 = images_image26
        if images_image27 is not None:
            self.images_image27 = images_image27
        if images_image28 is not None:
            self.images_image28 = images_image28
        if images_image29 is not None:
            self.images_image29 = images_image29
        if images_image30 is not None:
            self.images_image30 = images_image30
        if images_image31 is not None:
            self.images_image31 = images_image31
        if images_image32 is not None:
            self.images_image32 = images_image32
        if images_image33 is not None:
            self.images_image33 = images_image33
        if images_image34 is not None:
            self.images_image34 = images_image34
        if images_image35 is not None:
            self.images_image35 = images_image35
        if images_image36 is not None:
            self.images_image36 = images_image36
        if images_image37 is not None:
            self.images_image37 = images_image37
        if images_image38 is not None:
            self.images_image38 = images_image38
        if images_image39 is not None:
            self.images_image39 = images_image39
        if images_image40 is not None:
            self.images_image40 = images_image40
        if images_image41 is not None:
            self.images_image41 = images_image41
        if images_image42 is not None:
            self.images_image42 = images_image42
        if images_image43 is not None:
            self.images_image43 = images_image43
        if images_image44 is not None:
            self.images_image44 = images_image44
        if images_image45 is not None:
            self.images_image45 = images_image45
        if images_image46 is not None:
            self.images_image46 = images_image46
        if images_image47 is not None:
            self.images_image47 = images_image47
        if images_image48 is not None:
            self.images_image48 = images_image48
        if images_image49 is not None:
            self.images_image49 = images_image49
