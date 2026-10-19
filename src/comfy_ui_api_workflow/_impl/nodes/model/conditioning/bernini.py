from ...base import BaseNode
from ...validation import validate_range


class BerniniConditioning(BaseNode):
    __slots__ = (
        "positive",
        "negative",
        "vae",
        "width",
        "height",
        "length",
        "batch_size",
        "source_video",
        "reference_video",
        "reference_images_reference_image_0",
        "reference_images_reference_image_1",
        "reference_images_reference_image_2",
        "reference_images_reference_image_3",
        "reference_images_reference_image_4",
        "reference_images_reference_image_5",
        "reference_images_reference_image_6",
        "reference_images_reference_image_7",
        "ref_max_size",
    )
    CATEGORY = "model/conditioning/bernini"
    OUTPUTS = {"positive": 0, "negative": 1, "latent": 2}
    RETURN_TYPES = (
        "CONDITIONING",
        "CONDITIONING",
        "LATENT",
    )
    INPUT_NAMES = {
        "reference_images_reference_image_0": "reference_images.reference_image_0",
        "reference_images_reference_image_1": "reference_images.reference_image_1",
        "reference_images_reference_image_2": "reference_images.reference_image_2",
        "reference_images_reference_image_3": "reference_images.reference_image_3",
        "reference_images_reference_image_4": "reference_images.reference_image_4",
        "reference_images_reference_image_5": "reference_images.reference_image_5",
        "reference_images_reference_image_6": "reference_images.reference_image_6",
        "reference_images_reference_image_7": "reference_images.reference_image_7",
    }
    INPUT_SOCKET_TYPES = {
        "positive": "CONDITIONING",
        "negative": "CONDITIONING",
        "vae": "VAE",
        "width": "INT",
        "height": "INT",
        "length": "INT",
        "batch_size": "INT",
        "source_video": "IMAGE",
        "reference_video": "IMAGE",
        "reference_images_reference_image_0": "IMAGE",
        "reference_images_reference_image_1": "IMAGE",
        "reference_images_reference_image_2": "IMAGE",
        "reference_images_reference_image_3": "IMAGE",
        "reference_images_reference_image_4": "IMAGE",
        "reference_images_reference_image_5": "IMAGE",
        "reference_images_reference_image_6": "IMAGE",
        "reference_images_reference_image_7": "IMAGE",
        "ref_max_size": "INT",
    }

    def __init__(
        self,
        node_id: str,
        positive: tuple[str, int],
        negative: tuple[str, int],
        vae: tuple[str, int],
        width: int | tuple[str, int] = 832,
        height: int | tuple[str, int] = 480,
        length: int | tuple[str, int] = 81,
        batch_size: int | tuple[str, int] = 1,
        source_video: tuple[str, int] | None = None,
        reference_video: tuple[str, int] | None = None,
        reference_images_reference_image_0: tuple[str, int] | None = None,
        reference_images_reference_image_1: tuple[str, int] | None = None,
        reference_images_reference_image_2: tuple[str, int] | None = None,
        reference_images_reference_image_3: tuple[str, int] | None = None,
        reference_images_reference_image_4: tuple[str, int] | None = None,
        reference_images_reference_image_5: tuple[str, int] | None = None,
        reference_images_reference_image_6: tuple[str, int] | None = None,
        reference_images_reference_image_7: tuple[str, int] | None = None,
        ref_max_size: int | tuple[str, int] = 848,
    ) -> None:
        super().__init__(node_id=node_id)
        self.positive = positive
        self.negative = negative
        self.vae = vae
        self.width = width
        self.height = height
        self.length = length
        self.batch_size = batch_size
        if source_video is not None:
            self.source_video = source_video
        if reference_video is not None:
            self.reference_video = reference_video
        if reference_images_reference_image_0 is not None:
            self.reference_images_reference_image_0 = reference_images_reference_image_0
        if reference_images_reference_image_1 is not None:
            self.reference_images_reference_image_1 = reference_images_reference_image_1
        if reference_images_reference_image_2 is not None:
            self.reference_images_reference_image_2 = reference_images_reference_image_2
        if reference_images_reference_image_3 is not None:
            self.reference_images_reference_image_3 = reference_images_reference_image_3
        if reference_images_reference_image_4 is not None:
            self.reference_images_reference_image_4 = reference_images_reference_image_4
        if reference_images_reference_image_5 is not None:
            self.reference_images_reference_image_5 = reference_images_reference_image_5
        if reference_images_reference_image_6 is not None:
            self.reference_images_reference_image_6 = reference_images_reference_image_6
        if reference_images_reference_image_7 is not None:
            self.reference_images_reference_image_7 = reference_images_reference_image_7
        self.ref_max_size = ref_max_size
        self.__validate()

    def __validate(self) -> None:
        validate_range("width", self.width, 16, 8192)
        validate_range("height", self.height, 16, 8192)
        validate_range("length", self.length, 1, 8192)
        validate_range("batch_size", self.batch_size, 1, 4096)
        validate_range("ref_max_size", self.ref_max_size, 16, 8192)
