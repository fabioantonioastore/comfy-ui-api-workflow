from ...base import BaseNode
from ...validation import validate_bool, validate_range


class ChromaRadianceOptions(BaseNode):
    __slots__ = (
        "model",
        "preserve_wrapper",
        "start_sigma",
        "end_sigma",
        "nerf_tile_size",
        "force_sequential_txt_ids",
    )
    CATEGORY = "model/patch/chroma radiance"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "preserve_wrapper": "BOOLEAN",
        "start_sigma": "FLOAT",
        "end_sigma": "FLOAT",
        "nerf_tile_size": "INT",
        "force_sequential_txt_ids": "BOOLEAN",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        preserve_wrapper: bool | tuple[str, int] = True,
        start_sigma: float | tuple[str, int] = 1.0,
        end_sigma: float | tuple[str, int] = 0.0,
        nerf_tile_size: int | tuple[str, int] = -1,
        force_sequential_txt_ids: bool | tuple[str, int] = False,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.preserve_wrapper = preserve_wrapper
        self.start_sigma = start_sigma
        self.end_sigma = end_sigma
        self.nerf_tile_size = nerf_tile_size
        self.force_sequential_txt_ids = force_sequential_txt_ids
        self.__validate()

    def __validate(self) -> None:
        validate_bool("preserve_wrapper", self.preserve_wrapper)
        validate_range("start_sigma", self.start_sigma, 0.0, 1.0)
        validate_range("end_sigma", self.end_sigma, 0.0, 1.0)
        validate_range("nerf_tile_size", self.nerf_tile_size, -1)
        validate_bool("force_sequential_txt_ids", self.force_sequential_txt_ids)
