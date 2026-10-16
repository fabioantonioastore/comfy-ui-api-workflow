from typing import Literal, get_args

from ....base import BaseNode
from ....validation import validate_bool, validate_choice, validate_range


class WanTrackToVideo(BaseNode):
    __slots__ = (
        "positive",
        "negative",
        "vae",
        "start_image",
        "tracks",
        "width",
        "height",
        "length",
        "batch_size",
        "temperature",
        "topk",
        "clip_vision_output",
    )
    CATEGORY = "model/conditioning/wan/move"
    OUTPUTS = {"positive": 0, "negative": 1, "latent": 2}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    INPUT_SOCKET_TYPES = {
        "positive": "CONDITIONING",
        "negative": "CONDITIONING",
        "vae": "VAE",
        "tracks": "STRING",
        "width": "INT",
        "height": "INT",
        "length": "INT",
        "batch_size": "INT",
        "temperature": "FLOAT",
        "topk": "INT",
        "start_image": "IMAGE",
        "clip_vision_output": "CLIP_VISION_OUTPUT",
    }

    def __init__(
        self,
        node_id: str,
        positive: tuple[str, int],
        negative: tuple[str, int],
        vae: tuple[str, int],
        start_image: tuple[str, int],
        tracks: str | tuple[str, int] = "[]",
        width: int | tuple[str, int] = 832,
        height: int | tuple[str, int] = 480,
        length: int | tuple[str, int] = 81,
        batch_size: int | tuple[str, int] = 1,
        temperature: float | tuple[str, int] = 220.0,
        topk: int | tuple[str, int] = 2,
        clip_vision_output: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.positive = positive
        self.negative = negative
        self.vae = vae
        self.start_image = start_image
        self.tracks = tracks
        self.width = width
        self.height = height
        self.length = length
        self.batch_size = batch_size
        self.temperature = temperature
        self.topk = topk
        if clip_vision_output is not None:
            self.clip_vision_output = clip_vision_output
        self.__validate()

    def __validate(self) -> None:
        validate_range("width", self.width, 16, 16_384)
        validate_range("height", self.height, 16, 16_384)
        validate_range("length", self.length, 1, 16_384)
        validate_range("batch_size", self.batch_size, 1, 4096)
        validate_range("temperature", self.temperature, 1.0, 1000.0)
        validate_range("topk", self.topk, 1, 10)


class WanMoveTrackToVideo(BaseNode):
    __slots__ = (
        "positive",
        "negative",
        "vae",
        "start_image",
        "strength",
        "width",
        "height",
        "length",
        "batch_size",
        "tracks",
        "clip_vision_output",
    )
    CATEGORY = "model/conditioning/wan/move"
    OUTPUTS = {"positive": 0, "negative": 1, "latent": 2}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    INPUT_SOCKET_TYPES = {
        "positive": "CONDITIONING",
        "negative": "CONDITIONING",
        "vae": "VAE",
        "strength": "FLOAT",
        "width": "INT",
        "height": "INT",
        "length": "INT",
        "batch_size": "INT",
        "start_image": "IMAGE",
        "tracks": "TRACKS",
        "clip_vision_output": "CLIP_VISION_OUTPUT",
    }

    def __init__(
        self,
        node_id: str,
        positive: tuple[str, int],
        negative: tuple[str, int],
        vae: tuple[str, int],
        start_image: tuple[str, int],
        strength: float | tuple[str, int] = 1.0,
        width: int | tuple[str, int] = 832,
        height: int | tuple[str, int] = 480,
        length: int | tuple[str, int] = 81,
        batch_size: int | tuple[str, int] = 1,
        tracks: tuple[str, int] | None = None,
        clip_vision_output: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.positive = positive
        self.negative = negative
        self.vae = vae
        self.start_image = start_image
        self.strength = strength
        self.width = width
        self.height = height
        self.length = length
        self.batch_size = batch_size
        if tracks is not None:
            self.tracks = tracks
        if clip_vision_output is not None:
            self.clip_vision_output = clip_vision_output
        self.__validate()

    def __validate(self) -> None:
        validate_range("strength", self.strength, 0.0, 100.0)
        validate_range("width", self.width, 16, 16_384)
        validate_range("height", self.height, 16, 16_384)
        validate_range("length", self.length, 1, 16_384)
        validate_range("batch_size", self.batch_size, 1, 4096)


class WanMoveTracksFromCoords(BaseNode):
    __slots__ = ("track_coords", "track_mask")
    CATEGORY = "model/conditioning/wan/move"
    OUTPUTS = {"TRACKS": 0, "track_length": 1}
    RETURN_TYPES = ("TRACKS", "INT")
    INPUT_SOCKET_TYPES = {"track_coords": "STRING", "track_mask": "MASK"}

    def __init__(
        self,
        node_id: str,
        track_coords: tuple[str, int] | None = None,
        track_mask: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        if track_coords is not None:
            self.track_coords = track_coords
        if track_mask is not None:
            self.track_mask = track_mask


class WanMoveConcatTrack(BaseNode):
    __slots__ = ("tracks_1", "tracks_2")
    CATEGORY = "model/conditioning/wan/move"
    OUTPUTS = {"TRACKS": 0}
    RETURN_TYPES = ("TRACKS",)
    INPUT_SOCKET_TYPES = {"tracks_1": "TRACKS", "tracks_2": "TRACKS"}

    def __init__(
        self,
        node_id: str,
        tracks_1: tuple[str, int],
        tracks_2: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.tracks_1 = tracks_1
        if tracks_2 is not None:
            self.tracks_2 = tracks_2


class WanMoveVisualizeTracks(BaseNode):
    __slots__ = (
        "images",
        "line_resolution",
        "circle_size",
        "opacity",
        "line_width",
        "tracks",
    )
    CATEGORY = "model/conditioning/wan/move"
    OUTPUTS = {"IMAGE": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_SOCKET_TYPES = {
        "images": "IMAGE",
        "line_resolution": "INT",
        "circle_size": "INT",
        "opacity": "FLOAT",
        "line_width": "INT",
        "tracks": "TRACKS",
    }

    def __init__(
        self,
        node_id: str,
        images: tuple[str, int],
        line_resolution: int | tuple[str, int] = 24,
        circle_size: int | tuple[str, int] = 12,
        opacity: float | tuple[str, int] = 0.75,
        line_width: int | tuple[str, int] = 16,
        tracks: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.images = images
        self.line_resolution = line_resolution
        self.circle_size = circle_size
        self.opacity = opacity
        self.line_width = line_width
        if tracks is not None:
            self.tracks = tracks
        self.__validate()

    def __validate(self) -> None:
        validate_range("line_resolution", self.line_resolution, 1, 1024)
        validate_range("circle_size", self.circle_size, 1, 128)
        validate_range("opacity", self.opacity, 0.0, 1.0)
        validate_range("line_width", self.line_width, 1, 128)


GenerateTracks_Interpolations = Literal[
    "linear", "ease_in", "ease_out", "ease_in_out", "constant"
]
GENERATETRACKS_INTERPOLATIONS = frozenset(get_args(GenerateTracks_Interpolations))


class GenerateTracks(BaseNode):
    __slots__ = (
        "interpolation",
        "width",
        "height",
        "start_x",
        "start_y",
        "end_x",
        "end_y",
        "num_frames",
        "num_tracks",
        "track_spread",
        "bezier",
        "mid_x",
        "mid_y",
        "track_mask",
    )
    CATEGORY = "model/conditioning/wan/move"
    OUTPUTS = {"TRACKS": 0, "track_length": 1}
    RETURN_TYPES = ("TRACKS", "INT")
    INPUT_SOCKET_TYPES = {
        "width": "INT",
        "height": "INT",
        "start_x": "FLOAT",
        "start_y": "FLOAT",
        "end_x": "FLOAT",
        "end_y": "FLOAT",
        "num_frames": "INT",
        "num_tracks": "INT",
        "track_spread": "FLOAT",
        "bezier": "BOOLEAN",
        "mid_x": "FLOAT",
        "mid_y": "FLOAT",
        "interpolation": "COMBO",
        "track_mask": "MASK",
    }

    def __init__(
        self,
        node_id: str,
        interpolation: GenerateTracks_Interpolations | tuple[str, int],
        width: int | tuple[str, int] = 832,
        height: int | tuple[str, int] = 480,
        start_x: float | tuple[str, int] = 0.0,
        start_y: float | tuple[str, int] = 0.0,
        end_x: float | tuple[str, int] = 1.0,
        end_y: float | tuple[str, int] = 1.0,
        num_frames: int | tuple[str, int] = 81,
        num_tracks: int | tuple[str, int] = 5,
        track_spread: float | tuple[str, int] = 0.025,
        bezier: bool | tuple[str, int] = False,
        mid_x: float | tuple[str, int] = 0.5,
        mid_y: float | tuple[str, int] = 0.5,
        track_mask: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.interpolation = interpolation
        self.width = width
        self.height = height
        self.start_x = start_x
        self.start_y = start_y
        self.end_x = end_x
        self.end_y = end_y
        self.num_frames = num_frames
        self.num_tracks = num_tracks
        self.track_spread = track_spread
        self.bezier = bezier
        self.mid_x = mid_x
        self.mid_y = mid_y
        if track_mask is not None:
            self.track_mask = track_mask
        self.__validate()

    def __validate(self) -> None:
        validate_choice(
            "interpolation", self.interpolation, GENERATETRACKS_INTERPOLATIONS
        )
        validate_range("width", self.width, 16, 4096)
        validate_range("height", self.height, 16, 4096)
        validate_range("start_x", self.start_x, 0.0, 1.0)
        validate_range("start_y", self.start_y, 0.0, 1.0)
        validate_range("end_x", self.end_x, 0.0, 1.0)
        validate_range("end_y", self.end_y, 0.0, 1.0)
        validate_range("num_frames", self.num_frames, 1, 1024)
        validate_range("num_tracks", self.num_tracks, 1, 100)
        validate_range("track_spread", self.track_spread, 0.0, 1.0)
        validate_bool("bezier", self.bezier)
        validate_range("mid_x", self.mid_x, 0.0, 1.0)
        validate_range("mid_y", self.mid_y, 0.0, 1.0)
