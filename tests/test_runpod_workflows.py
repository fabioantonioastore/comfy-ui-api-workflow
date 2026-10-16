import base64
import colorsys
import html
import json
import math
import os
import re
import shutil
import struct
import threading
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pytest

from comfy_ui_api_workflow._impl.errors.node import NodeHasDependentsError
from comfy_ui_api_workflow._impl.errors.workflow import InvalidWorkflowError
from comfy_ui_api_workflow._impl.nodes.advanced.guidance import CFGNorm
from comfy_ui_api_workflow._impl.nodes.audio import (
    AudioAdjustVolume,
    AudioConcat,
    AudioEqualizer3Band,
    AudioMerge,
    EmptyAudio,
    JoinAudioChannels,
    SplitAudioChannels,
    TrimAudioDuration,
)
from comfy_ui_api_workflow._impl.nodes.experimental import (
    LatentBlend,
    SelfAttentionGuidance,
)
from comfy_ui_api_workflow._impl.nodes.image import (
    EmptyImage,
    GetImageSize,
    LoadImage,
    LoadImageMask,
    SaveAnimatedPNG,
    SaveAnimatedWEBP,
    SaveImage,
)
from comfy_ui_api_workflow._impl.nodes.image.adjustments import (
    AdjustBrightness,
    AdjustContrast,
)
from comfy_ui_api_workflow._impl.nodes.image.batch import (
    ImageFromBatch,
    ImageGrid,
    ImageMergeTileList,
    RebatchImages,
    RepeatImageBatch,
    ShuffleDataset,
    SplitImageToTileList,
)
from comfy_ui_api_workflow._impl.nodes.image.color import (
    ImageColorSpace,
    ImageInvert,
    ImageRGBToYUV,
    ImageYUVToRGB,
    NormalizeImages,
)
from comfy_ui_api_workflow._impl.nodes.image.compositing import (
    ImageCompositeMasked,
    JoinImageWithAlpha,
    PorterDuffImageComposite,
    SplitImageWithAlpha,
)
from comfy_ui_api_workflow._impl.nodes.image.filters import (
    Canny,
    ImageAddNoise,
    ImageBlend,
    ImageBlur,
    ImageQuantize,
    ImageSharpen,
    Morphology,
)
from comfy_ui_api_workflow._impl.nodes.image.mask import (
    CropMask,
    FeatherMask,
    GrowMask,
    ImageColorToMask,
    ImageToMask,
    InvertMask,
    MaskComposite,
    MaskToImage,
    SolidMask,
    ThresholdMask,
)
from comfy_ui_api_workflow._impl.nodes.image.transform import (
    CenterCropImages,
    ImageCropV2,
    ImageFlip,
    ImagePadForOutpaint,
    ImageRotate,
    ImageStitch,
    RandomCropImages,
    ResizeAndPadImage,
)
from comfy_ui_api_workflow._impl.nodes.image.upscaling import (
    ImageScale,
    ImageScaleBy,
    ImageScaleToMaxDimension,
    ImageScaleToTotalPixels,
)
from comfy_ui_api_workflow._impl.nodes.model.conditioning import (
    CLIPSetLastLayer,
    CLIPTextEncode,
    InpaintModelConditioning,
)
from comfy_ui_api_workflow._impl.nodes.model.conditioning.stable_diffusion import (
    CLIPTextEncodeSDXL,
)
from comfy_ui_api_workflow._impl.nodes.model.conditioning.transform import (
    ConditioningAverage,
    ConditioningCombine,
    ConditioningConcat,
    ConditioningMultiply,
    ConditioningSetArea,
    ConditioningSetMask,
    ConditioningSetTimestepRange,
    ConditioningZeroOut,
)
from comfy_ui_api_workflow._impl.nodes.model.latent import (
    EmptyLatentImage,
    LatentComposite,
    LatentUpscale,
    LatentUpscaleBy,
    SetLatentNoiseMask,
    VAEDecode,
    VAEDecodeTiled,
    VAEEncode,
    VAEEncodeForInpaint,
    VAEEncodeTiled,
)
from comfy_ui_api_workflow._impl.nodes.model.latent.advanced import (
    LatentAdd,
    LatentBatchSeedBehavior,
    LatentInterpolate,
    LatentMultiply,
    LatentSubtract,
)
from comfy_ui_api_workflow._impl.nodes.model.latent.advanced.operations import (
    LatentApplyOperation,
    LatentOperationSharpen,
    LatentOperationTonemapReinhard,
)
from comfy_ui_api_workflow._impl.nodes.model.latent.batch import (
    LatentFromBatch,
    RepeatLatentBatch,
)
from comfy_ui_api_workflow._impl.nodes.model.latent.transform import (
    LatentCrop,
    LatentFlip,
    LatentRotate,
)
from comfy_ui_api_workflow._impl.nodes.model.loaders import (
    CheckpointLoaderSimple,
    DualCLIPLoader,
    LoraLoader,
    UNETLoader,
    UpscaleModelLoader,
    VAELoader,
)
from comfy_ui_api_workflow._impl.nodes.model.merging import (
    CLIPMergeSimple,
    ModelMergeSimple,
)
from comfy_ui_api_workflow._impl.nodes.model.patch import (
    ModelSamplingDiscrete,
    RescaleCFG,
)
from comfy_ui_api_workflow._impl.nodes.model.patch.unet import (
    EpsilonScaling,
    FreeU,
    FreeU_V2,
    HyperTile,
    PatchModelAddDownscale,
    PerturbedAttentionGuidance,
    TomePatchModel,
)
from comfy_ui_api_workflow._impl.nodes.model.sampling import KSampler, KSamplerAdvanced
from comfy_ui_api_workflow._impl.nodes.model.sampling.custom import (
    APG,
    SamplerCustom,
    SamplerCustomAdvanced,
)
from comfy_ui_api_workflow._impl.nodes.model.sampling.guiders import (
    BasicGuider,
    CFGGuider,
    CFGOverride,
    DualCFGGuider,
)
from comfy_ui_api_workflow._impl.nodes.model.sampling.noise import RandomNoise
from comfy_ui_api_workflow._impl.nodes.model.sampling.samplers import (
    KSamplerSelect,
    SamplerDPMPP_2M_SDE,
    SamplerEulerAncestral,
)
from comfy_ui_api_workflow._impl.nodes.model.sampling.schedulers import (
    BasicScheduler,
    ExponentialScheduler,
    KarrasScheduler,
)
from comfy_ui_api_workflow._impl.nodes.model.sampling.sigmas import SplitSigmas
from comfy_ui_api_workflow._impl.nodes.text import (
    CaseConverter,
    JsonExtractString,
    RegexExtract,
    RegexMatch,
    RegexReplace,
    StringCompare,
    StringConcatenate,
    StringContains,
    StringLength,
    StringReplace,
    StringSubstring,
    StringTrim,
    TextOverlay,
)
from comfy_ui_api_workflow._impl.nodes.utilities import (
    ColorToRGBInt,
    ComfyMathExpression,
    ComfyNumberConvert,
    PreviewAny,
    ResolutionSelector,
    SeedNode,
)
from comfy_ui_api_workflow._impl.nodes.utilities.logic import (
    ComfyNotNode,
    ComfySwitchNode,
)
from comfy_ui_api_workflow._impl.nodes.utilities.primitive import (
    PrimitiveBoolean,
    PrimitiveBoundingBox,
    PrimitiveFloat,
    PrimitiveInt,
    PrimitiveString,
    PrimitiveStringMultiline,
)
from comfy_ui_api_workflow._impl.nodes.video import (
    CreateVideo,
    GetVideoComponents,
    SaveVideo,
    SaveWEBM,
    VideoFrameSample,
    VideoSlice,
)
from comfy_ui_api_workflow._impl.nodes.video.transform import VideoTemporalCrop
from comfy_ui_api_workflow._impl.pipeline import Pipeline
from comfy_ui_api_workflow._impl.workflow import Workflow

# A hundred workflows built with the library. Offline, each one must survive a
# reload; with `hatch test -- --runpod` each one runs on the RunPod endpoint in
# .env and its outputs land in runpod_outputs/workflows, with an index.html.
GALLERY = Path(__file__).resolve().parent.parent / "runpod_outputs" / "workflows"
CHECKPOINT = "sd_xl_base_1.0.safetensors"
NEGATIVE = "blurry, low quality, deformed, watermark, text, nsfw, nude"
TIMEOUT = 1200
GPU_TIMEOUT = 1800
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MATROSKA = b"\x1a\x45\xdf\xa3"

RED, GREEN, BLUE = 0xFF0000, 0x00FF00, 0x0000FF
YELLOW, CYAN, MAGENTA = 0xFFFF00, 0x00FFFF, 0xFF00FF
WHITE, BLACK, GRAY, ORANGE = 0xFFFFFF, 0x000000, 0x808080, 0xFF8000
QUADRANTS = (RED, GREEN, BLUE, WHITE)
SIX = (RED, GREEN, BLUE, YELLOW, CYAN, MAGENTA)
ROTATIONS = ("90 degrees", "180 degrees", "270 degrees")
ROTATION_LABELS = {"90 degrees": "rot90", "180 degrees": "rot180", "270 degrees": "rot270"}
FLIPS = ("x-axis: vertically", "y-axis: horizontally")
FLIP_LABELS = {"x-axis: vertically": "flip_x", "y-axis: horizontally": "flip_y"}


def rgb(color: Any) -> tuple[int, int, int]:
    if isinstance(color, tuple):
        return color
    return (color >> 16) & 0xFF, (color >> 8) & 0xFF, color & 0xFF


def hues(count: int) -> list[int]:
    colors = []
    for index in range(count):
        r, g, b = colorsys.hsv_to_rgb(index / count, 1.0, 1.0)
        colors.append((round(r * 255) << 16) | (round(g * 255) << 8) | round(b * 255))
    return colors


def pixel_of(values: Any) -> tuple[int, ...]:
    # SaveImage clips 255 * value and truncates it to uint8.
    return tuple(int(255 * min(max(value, 0.0), 1.0)) for value in values)


def gray(value: float) -> tuple[int, ...]:
    return pixel_of((value, value, value))


# ---------------------------------------------------------------- media files


@dataclass(frozen=True)
class Picture:
    width: int
    height: int
    channels: int
    rows: tuple[bytes, ...]

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height

    def pixel(self, x: int, y: int) -> tuple[int, ...]:
        start = x * self.channels
        values = self.rows[y][start : start + self.channels]
        return tuple(values[:3]) if self.channels >= 3 else (values[0],) * 3

    def alpha(self, x: int, y: int) -> int:
        return self.rows[y][x * self.channels + 3]

    def pixels(self, box: tuple[int, int, int, int] | None = None):
        left, top, right, bottom = box or (0, 0, self.width, self.height)
        for y in range(top, bottom):
            for x in range(left, right):
                yield self.pixel(x, y)

    def mean(self, box: tuple[int, int, int, int] | None = None) -> tuple[float, ...]:
        totals, count = [0, 0, 0], 0
        for value in self.pixels(box):
            count += 1
            for channel in range(3):
                totals[channel] += value[channel]
        return tuple(total / count for total in totals)

    def mirrored(self) -> "Picture":
        step = self.channels
        rows = tuple(
            b"".join(row[x * step : (x + 1) * step] for x in reversed(range(self.width)))
            for row in self.rows
        )
        return Picture(self.width, self.height, self.channels, rows)


def png_chunks(data: bytes):
    assert data[:8] == PNG_SIGNATURE, "not a PNG file"
    position = 8
    while position < len(data):
        (length,) = struct.unpack(">I", data[position : position + 4])
        yield data[position + 4 : position + 8], data[position + 8 : position + 8 + length]
        position += 12 + length


def png_size(data: bytes) -> tuple[int, int]:
    assert data[:8] == PNG_SIGNATURE, "not a PNG file"
    return struct.unpack(">II", data[16:24])


def apng_frames(data: bytes) -> int:
    for kind, body in png_chunks(data):
        if kind == b"acTL":
            return struct.unpack(">I", body[:4])[0]
    return 1


def unfilter(kind: int, line: bytearray, previous: bytearray, step: int) -> None:
    if kind == 1:
        for i in range(step, len(line)):
            line[i] = (line[i] + line[i - step]) & 0xFF
    elif kind == 2:
        for i in range(len(line)):
            line[i] = (line[i] + previous[i]) & 0xFF
    elif kind == 3:
        for i in range(len(line)):
            left = line[i - step] if i >= step else 0
            line[i] = (line[i] + (left + previous[i]) // 2) & 0xFF
    elif kind == 4:
        for i in range(len(line)):
            a = line[i - step] if i >= step else 0
            b = previous[i]
            c = previous[i - step] if i >= step else 0
            p = a + b - c
            pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
            predictor = a if pa <= pb and pa <= pc else b if pb <= pc else c
            line[i] = (line[i] + predictor) & 0xFF


def decode_png(data: bytes) -> Picture:
    header, idat = b"", b""
    for kind, body in png_chunks(data):
        if kind == b"IHDR":
            header = body
        elif kind == b"IDAT":
            idat += body
    width, height, depth, color_type, _, _, interlace = struct.unpack(">IIBBBBB", header)
    assert depth == 8 and interlace == 0, f"unsupported PNG: depth {depth}, interlace {interlace}"
    channels = {0: 1, 2: 3, 4: 2, 6: 4}[color_type]
    stride = width * channels
    raw = zlib.decompress(idat)
    rows, previous = [], bytearray(stride)
    for y in range(height):
        offset = y * (stride + 1)
        line = bytearray(raw[offset + 1 : offset + 1 + stride])
        unfilter(raw[offset], line, previous, channels)
        rows.append(bytes(line))
        previous = line
    return Picture(width, height, channels, tuple(rows))


def encode_png(width: int, height: int, pixel: Callable, alpha: bool = False) -> bytes:
    def chunk(kind: bytes, body: bytes) -> bytes:
        crc = struct.pack(">I", zlib.crc32(kind + body))
        return struct.pack(">I", len(body)) + kind + body + crc

    rows = b"".join(
        b"\x00" + bytes(value for x in range(width) for value in pixel(x, y))
        for y in range(height)
    )
    header = struct.pack(">IIBBBBB", width, height, 8, 6 if alpha else 2, 0, 0, 0)
    return (
        PNG_SIGNATURE
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def webp_info(data: bytes) -> tuple[tuple[int, int] | None, int]:
    assert data[:4] == b"RIFF" and data[8:12] == b"WEBP", "not a WEBP file"
    size, frames, position = None, 0, 12
    while position + 8 <= len(data):
        kind = data[position : position + 4]
        (length,) = struct.unpack("<I", data[position + 4 : position + 8])
        body = data[position + 8 : position + 8 + length]
        if kind == b"VP8X":
            width = 1 + int.from_bytes(body[4:7], "little")
            height = 1 + int.from_bytes(body[7:10], "little")
            size = (width, height)
        elif kind == b"ANMF":
            frames += 1
        position += 8 + length + (length & 1)
    return size, max(frames, 1)


def assert_mp4(data: bytes, *codecs: bytes) -> None:
    assert data[4:8] == b"ftyp", "not an MP4 file"
    for codec in codecs:
        assert codec in data, f"{codec!r} is not in the MP4 file"


def assert_matroska(data: bytes, *codecs: bytes) -> None:
    assert data[:4] == MATROSKA, "not a Matroska/WebM file"
    for codec in codecs:
        assert codec in data, f"{codec!r} is not in the Matroska file"


class Outputs:
    """The files a job returned, found by the filename_prefix of each save node."""

    def __init__(self, files: list[tuple[str, bytes]]) -> None:
        self.files = sorted(files)

    def names(self) -> list[str]:
        return [name for name, _ in self.files]

    def of(self, label: str) -> list[bytes]:
        pattern = re.compile(rf"{re.escape(label)}_\d+_\.\w+")
        return [data for name, data in self.files if pattern.fullmatch(name)]

    def one(self, label: str) -> bytes:
        found = self.of(label)
        assert len(found) == 1, f"expected one {label!r} file, got {self.names()}"
        return found[0]

    def name(self, label: str) -> str:
        pattern = re.compile(rf"{re.escape(label)}_\d+_\.\w+")
        (name,) = [name for name, _ in self.files if pattern.fullmatch(name)]
        return name

    def picture(self, label: str) -> Picture:
        return decode_png(self.one(label))

    def pictures(self, label: str) -> list[Picture]:
        return [decode_png(data) for data in self.of(label)]


def assert_pixel(picture: Picture, x: int, y: int, expected: Any, tolerance: int = 2):
    actual, want = picture.pixel(x, y), rgb(expected)
    assert all(abs(a - b) <= tolerance for a, b in zip(actual, want)), (
        f"pixel ({x}, {y}) of the {picture.width}x{picture.height} image is "
        f"{actual}, expected {want}"
    )


def quarter_points(width: int, height: int) -> list[tuple[int, int]]:
    left, right, top, bottom = width // 4, 3 * width // 4, height // 4, 3 * height // 4
    return [(left, top), (right, top), (left, bottom), (right, bottom)]


def quadrant_color(x: int, y: int, width: int, height: int, colors=QUADRANTS) -> int:
    return colors[(2 if y >= height // 2 else 0) + (1 if x >= width // 2 else 0)]


def assert_quadrants(picture: Picture, colors=QUADRANTS, tolerance: int = 2) -> None:
    for (x, y), color in zip(quarter_points(*picture.size), colors):
        assert_pixel(picture, x, y, color, tolerance)


def assert_generated(data: bytes, size: tuple[int, int], minimum: int = 40_000):
    assert png_size(data) == size, f"the image is {png_size(data)}, expected {size}"
    assert len(data) > minimum, (
        f"the {size[0]}x{size[1]} image has only {len(data)} bytes, so it is nearly flat"
    )


def mean_difference(first: Picture, second: Picture) -> float:
    assert first.size == second.size, f"{first.size} != {second.size}"
    total = 0
    for a, b in zip(first.pixels(), second.pixels()):
        total += sum(abs(x - y) for x, y in zip(a, b))
    return total / (first.width * first.height * 3)


def nearest(value: tuple[int, ...], palette, tolerance: int = 3) -> int:
    for candidate in palette:
        if all(abs(a - b) <= tolerance for a, b in zip(value, rgb(candidate))):
            return candidate
    raise AssertionError(f"{value} is none of {[rgb(color) for color in palette]}")


def rotated_source(rotation: str, x: int, y: int, width: int, height: int):
    # (width, height) is the size before the rotation; ImageRotate turns clockwise.
    if rotation == "90 degrees":
        return y, height - 1 - x
    if rotation == "180 degrees":
        return width - 1 - x, height - 1 - y
    if rotation == "270 degrees":
        return width - 1 - y, x
    return x, y


def flipped_source(method: str, x: int, y: int, width: int, height: int):
    if method.startswith("x"):
        return x, height - 1 - y
    return width - 1 - x, y


def pattern(x: int, y: int) -> tuple[int, int, int]:
    return (x * 10) % 256, (y * 15) % 256, (x * y * 3 + 40) % 256


PATTERN_PNG = encode_png(24, 16, pattern)
RGBA_PNG = encode_png(32, 32, lambda x, y: (200, 100, 50, 255 if x < 16 else 64), alpha=True)
STRIPE_PNG = encode_png(24, 16, lambda x, y: (255 if x < 12 else 0, 0, 0))


# ---------------------------------------------------------------- the registry


@dataclass(frozen=True)
class Case:
    name: str
    description: str
    build: Callable[..., Any]
    uploads: dict[str, bytes] = field(default_factory=dict)
    fails_with: tuple[str, ...] = ()
    library_says: str | None = None
    gpu: bool = False

    @property
    def timeout(self) -> float:
        return GPU_TIMEOUT if self.gpu else TIMEOUT

    def payload(self) -> tuple[dict, Callable | None]:
        if self.library_says is not None:
            return self.build(), None
        workflow = Workflow()
        check = self.build(workflow)
        return sent(workflow), check


CASES: list[Case] = []


def case(description: str, *, uploads=None, fails_with: Any = (), gpu: bool = False):
    reasons = (fails_with,) if isinstance(fails_with, str) else tuple(fails_with)

    def register(build: Callable) -> Callable:
        CASES.append(Case(build.__name__, description, build, uploads or {}, reasons, None, gpu))
        return build

    return register


def rejected(description: str, *, library_says: str, comfy_says: str):
    def register(build: Callable) -> Callable:
        CASES.append(Case(build.__name__, description, build, {}, (comfy_says,), library_says))
        return build

    return register


def sent(workflow: Workflow) -> dict:
    return json.loads(json.dumps(workflow.get_comfy_ui_workflow_format()))


# ---------------------------------------------------------------- building blocks


def solid(w: Workflow, width: Any, height: Any, color: Any, batch: Any = 1):
    node = w.add_node(EmptyImage, width=width, height=height, batch_size=batch, color=color)
    return node.output["IMAGE"]


def save(w: Workflow, images: Any, label: Any) -> None:
    w.add_node(SaveImage, images=images, filename_prefix=label)


def paste(w, destination, source, x=0, y=0, mask=None, resize=False):
    extra = {} if mask is None else {"mask": mask}
    return w.add_node(
        ImageCompositeMasked,
        destination=destination,
        source=source,
        x=x,
        y=y,
        resize_source=resize,
        **extra,
    ).output["IMAGE"]


def stitch(w, first, second, direction="right", match=False, spacing=0, color="white"):
    return w.add_node(
        ImageStitch,
        image1=first,
        image2=second,
        direction=direction,
        match_image_size=match,
        spacing_width=spacing,
        spacing_color=color,
    ).output["IMAGE"]


def strip(w, colors, size):
    image = solid(w, size, size, colors[0])
    for color in colors[1:]:
        image = stitch(w, image, solid(w, size, size, color))
    return image


def tiles_of(w, image, size, overlap=0):
    return w.add_node(
        SplitImageToTileList, image=image, tile_width=size, tile_height=size, overlap=overlap
    ).output["IMAGE"]


def batch_of(w, colors, size=64):
    # A real batch of frames with different colors: a strip cut into a tile
    # list, then rebatched into a single batch.
    tiles = tiles_of(w, strip(w, colors, size), size)
    return w.add_node(RebatchImages, images=tiles, batch_size=len(colors)).output["IMAGE"]


def quadrants(w, size, colors=QUADRANTS):
    half = size // 2
    image = solid(w, size, size, colors[0])
    for color, (x, y) in zip(colors[1:], ((half, 0), (0, half), (half, half))):
        image = paste(w, image, solid(w, half, half, color), x, y)
    return image


def rotate(w, image, rotation):
    return w.add_node(ImageRotate, image=image, rotation=rotation).output["IMAGE"]


def flip(w, image, method):
    return w.add_node(ImageFlip, image=image, flip_method=method).output["IMAGE"]


def scale(w, image, method, width, height, crop="disabled"):
    return w.add_node(
        ImageScale, image=image, upscale_method=method, width=width, height=height, crop=crop
    ).output["IMAGE"]


def scale_by(w, image, method, factor):
    return w.add_node(
        ImageScaleBy, image=image, upscale_method=method, scale_by=factor
    ).output["IMAGE"]


def crop(w, image, x, y, width, height):
    region = w.add_node(PrimitiveBoundingBox, x=x, y=y, width=width, height=height)
    return w.add_node(
        ImageCropV2, image=image, crop_region=region.output["BOUNDING_BOX"]
    ).output["IMAGE"]


def solid_mask(w, value, width, height):
    return w.add_node(SolidMask, value=value, width=width, height=height).output["MASK"]


def mask_image(w, mask):
    return w.add_node(MaskToImage, mask=mask).output["IMAGE"]


def combine_masks(w, destination, source, operation, x=0, y=0):
    return w.add_node(
        MaskComposite,
        destination=destination,
        source=source,
        x=x,
        y=y,
        operation=operation,
    ).output["MASK"]


def constant(w, value):
    # The math node's values are sockets without widgets, so numbers come from primitives.
    if isinstance(value, bool):
        return w.add_node(PrimitiveBoolean, value=value).output["BOOLEAN"]
    if isinstance(value, int):
        return w.add_node(PrimitiveInt, value=value).output["INT"]
    return w.add_node(PrimitiveFloat, value=value).output["FLOAT"]


def calc(w, expression, **values):
    named = {
        f"values_{name}": value if isinstance(value, tuple) else constant(w, value)
        for name, value in values.items()
    }
    return w.add_node(ComfyMathExpression, expression=expression, **named).output


def length(w, text):
    return w.add_node(StringLength, string=text).output["length"]


def equals(w, text, expected):
    return w.add_node(
        StringCompare, string_a=text, string_b=expected, mode="Equal", case_sensitive=True
    ).output["BOOLEAN"]


def negate(w, value):
    return w.add_node(ComfyNotNode, value=value).output["BOOLEAN"]


def board(w, checks: dict, label="board"):
    # Each boolean picks a green (true) or red (false) tile; the tiles are
    # stitched in order, so a red tile names the check that failed on ComfyUI.
    green, red = solid(w, 8, 8, GREEN), solid(w, 8, 8, RED)
    image = None
    for flag in checks.values():
        tile = w.add_node(ComfySwitchNode, switch=flag, on_true=green, on_false=red)
        image = tile.output["output"] if image is None else stitch(w, image, tile.output["output"])
    save(w, image, label)
    names = list(checks)

    def check(out: Outputs) -> None:
        picture = out.picture(label)
        assert picture.size == (8 * len(names), 8)
        failed = [
            name
            for index, name in enumerate(names)
            if picture.pixel(8 * index + 4, 4) != rgb(GREEN)
        ]
        assert not failed, f"checks that came back false on ComfyUI: {failed}"

    return check


def sdxl(w):
    loader = w.add_node(CheckpointLoaderSimple, ckpt_name=CHECKPOINT)
    return loader.output["MODEL"], loader.output["CLIP"], loader.output["VAE"]


def prompt(w, clip, text):
    return w.add_node(CLIPTextEncode, text=text, clip=clip).output["CONDITIONING"]


def empty_latent(w, width, height, batch=1):
    return w.add_node(
        EmptyLatentImage, width=width, height=height, batch_size=batch
    ).output["LATENT"]


def ksample(w, model, positive, negative, latent, *, seed=7, steps=14, cfg=6.5,
            sampler="dpmpp_2m", scheduler="karras", denoise=1.0):
    return w.add_node(
        KSampler,
        model=model,
        positive=positive,
        negative=negative,
        latent_image=latent,
        seed=seed,
        steps=steps,
        cfg=cfg,
        sampler_name=sampler,
        scheduler=scheduler,
        denoise=denoise,
    ).output["LATENT"]


def decode(w, latent, vae):
    return w.add_node(VAEDecode, samples=latent, vae=vae).output["IMAGE"]


def thumbnail(w, image, label, size=16):
    save(w, scale(w, image, "area", size, size), label)


class Framed(Pipeline):
    def build(self, image, border: int, color: int):
        size = self.add_node(GetImageSize, image=image)
        width = self.add_node(
            ComfyMathExpression, expression=f"a + {2 * border}", values_a=size.output["width"]
        )
        height = self.add_node(
            ComfyMathExpression, expression=f"a + {2 * border}", values_a=size.output["height"]
        )
        frame = self.add_node(
            EmptyImage, width=width.output["INT"], height=height.output["INT"], color=color
        )
        framed = self.add_node(
            ImageCompositeMasked,
            destination=frame.output["IMAGE"],
            source=image,
            x=border,
            y=border,
            resize_source=False,
        )
        return {"IMAGE": framed.output["IMAGE"]}


class Spotlight(Pipeline):
    def build(self, image, size: int, grow: int):
        dims = self.add_node(GetImageSize, image=image)
        dark = self.add_node(
            EmptyImage, width=dims.output["width"], height=dims.output["height"], color=BLACK
        )
        base = self.add_node(
            SolidMask, value=0.0, width=dims.output["width"], height=dims.output["height"]
        )
        spot = self.add_node(SolidMask, value=1.0, width=size, height=size)
        offset = self.add_node(
            ComfyMathExpression, expression=f"(a - {size}) / 2", values_a=dims.output["width"]
        )
        placed = self.add_node(
            MaskComposite,
            destination=base.output["MASK"],
            source=spot.output["MASK"],
            x=offset.output["INT"],
            y=offset.output["INT"],
            operation="add",
        )
        grown = self.add_node(
            GrowMask, mask=placed.output["MASK"], expand=grow, tapered_corners=False
        )
        lit = self.add_node(
            ImageCompositeMasked,
            destination=dark.output["IMAGE"],
            source=image,
            x=0,
            y=0,
            resize_source=False,
            mask=grown.output["MASK"],
        )
        return {"IMAGE": lit.output["IMAGE"], "MASK": grown.output["MASK"]}


class TextToImage(Pipeline):
    def build(self, text: str, width: int = 768, height: int = 768, steps: int = 14,
              seed: int = 42):
        loader = self.add_node(CheckpointLoaderSimple, ckpt_name=CHECKPOINT)
        positive = self.add_node(CLIPTextEncode, text=text, clip=loader.output["CLIP"])
        negative = self.add_node(CLIPTextEncode, text=NEGATIVE, clip=loader.output["CLIP"])
        latent = self.add_node(EmptyLatentImage, width=width, height=height, batch_size=1)
        sampled = self.add_node(
            KSampler,
            model=loader.output["MODEL"],
            positive=positive.output["CONDITIONING"],
            negative=negative.output["CONDITIONING"],
            latent_image=latent.output["LATENT"],
            sampler_name="dpmpp_2m",
            scheduler="karras",
            seed=seed,
            steps=steps,
            cfg=6.5,
        )
        image = self.add_node(
            VAEDecode, samples=sampled.output["LATENT"], vae=loader.output["VAE"]
        )
        return {"IMAGE": image.output["IMAGE"]}


# ---------------------------------------------------------------- geometry


@case("Quatro quadrantes montados com ImageCompositeMasked, girados em 90/180/270 graus "
      "e espelhados nos dois eixos; cada cor tem de cair no quadrante certo.")
def quadrant_rotations_and_flips(w):
    image = quadrants(w, 64)
    save(w, image, "original")
    for rotation in ROTATIONS:
        save(w, rotate(w, image, rotation), ROTATION_LABELS[rotation])
    for method in FLIPS:
        save(w, flip(w, image, method), FLIP_LABELS[method])
    save(w, flip(w, rotate(w, image, "90 degrees"), FLIPS[0]), "rot90_flip_x")

    def check(out):
        def expect(label, source):
            picture = out.picture(label)
            assert picture.size == (64, 64)
            for x, y in quarter_points(64, 64):
                assert_pixel(picture, x, y, quadrant_color(*source(x, y), 64, 64))

        expect("original", lambda x, y: (x, y))
        for rotation in ROTATIONS:
            expect(
                ROTATION_LABELS[rotation],
                lambda x, y, r=rotation: rotated_source(r, x, y, 64, 64),
            )
        for method in FLIPS:
            expect(
                FLIP_LABELS[method], lambda x, y, m=method: flipped_source(m, x, y, 64, 64)
            )
        expect(
            "rot90_flip_x",
            lambda x, y: rotated_source(
                "90 degrees", *flipped_source(FLIPS[0], x, y, 64, 64), 64, 64
            ),
        )

    return check


@case("ImageStitch nas quatro direções, com ajuste de tamanho, espaçamento colorido e o "
      "preenchimento que o ComfyUI faz quando as alturas/larguras não batem.")
def stitch_in_every_direction(w):
    right = stitch(w, solid(w, 32, 32, RED), solid(w, 32, 32, GREEN), "right", True, 4)
    down = stitch(w, right, solid(w, 32, 16, BLUE), "down", True)
    left = stitch(w, down, solid(w, 16, 16, YELLOW), "left")
    up = stitch(w, left, solid(w, 20, 10, MAGENTA), "up", spacing=2, color="red")
    for label, image in (("right", right), ("down", down), ("left", left), ("up", up)):
        save(w, image, label)

    def check(out):
        assert out.picture("right").size == (68, 32)
        assert out.picture("down").size == (68, 66)
        assert out.picture("left").size == (84, 66)
        final = out.picture("up")
        assert final.size == (84, 78)
        expected = (
            (42, 5, MAGENTA), (5, 5, BLACK), (70, 5, BLACK), (40, 10, RED), (40, 11, RED),
            (8, 45, YELLOW), (8, 17, WHITE), (8, 72, WHITE), (26, 22, RED),
            (49, 22, WHITE), (66, 22, GREEN), (46, 62, BLUE),
        )
        for x, y, color in expected:
            assert_pixel(final, x, y, color)

    return check


def outpaint_feather(i, j, feathering=8, height=32, width=32, top=8, bottom=24, left=16,
                     right=0):
    # Mirrors ImagePadForOutpaint's mask for pixel (row i, column j) of the original.
    distance = min(
        i if top else height,
        height - i if bottom else height,
        j if left else width,
        width - j if right else width,
    )
    return 0.0 if distance >= feathering else ((feathering - distance) / feathering) ** 2


@case("ImagePadForOutpaint com e sem feathering: borda nova cinza 50%, original no lugar "
      "e a máscara com o degradê quadrático do ComfyUI.")
def outpaint_canvas_and_mask(w):
    image = solid(w, 32, 32, RED)
    padding = {"left": 16, "top": 8, "right": 0, "bottom": 24}
    plain = w.add_node(ImagePadForOutpaint, image=image, feathering=0, **padding)
    soft = w.add_node(ImagePadForOutpaint, image=image, feathering=8, **padding)
    save(w, plain.output["IMAGE"], "canvas")
    save(w, mask_image(w, plain.output["MASK"]), "mask")
    save(w, mask_image(w, soft.output["MASK"]), "feathered")

    def check(out):
        canvas = out.picture("canvas")
        assert canvas.size == (48, 64)
        assert_pixel(canvas, 5, 5, gray(0.5))
        assert_pixel(canvas, 21, 13, RED)
        assert_pixel(canvas, 30, 50, gray(0.5))
        mask = out.picture("mask")
        assert_pixel(mask, 5, 5, WHITE)
        assert_pixel(mask, 21, 13, BLACK)
        feathered = out.picture("feathered")
        for i, j in ((0, 10), (4, 10), (16, 16), (16, 2), (31, 20)):
            assert_pixel(feathered, 16 + j, 8 + i, gray(outpaint_feather(i, j)))

    return check


@case("Recorte com ImageCropV2 + PrimitiveBoundingBox, CenterCropImages e RandomCropImages; "
      "ampliado de volta com nearest-exact, tem de ser idêntico ao original.")
def crop_chain_back_to_quadrants(w):
    image = quadrants(w, 64)
    cropped = crop(w, image, 16, 16, 32, 32)
    centered = w.add_node(CenterCropImages, images=cropped, width=16, height=16)
    untouched = w.add_node(
        RandomCropImages, images=centered.output["images"], width=64, height=64, seed=5
    )
    restored = scale(w, untouched.output["images"], "nearest-exact", 64, 64)
    save(w, image, "original")
    save(w, cropped, "cropped")
    save(w, centered.output["images"], "centered")
    save(w, restored, "restored")

    def check(out):
        for label, size in (("cropped", 32), ("centered", 16), ("restored", 64)):
            picture = out.picture(label)
            assert picture.size == (size, size)
            assert_quadrants(picture)
        assert out.picture("restored").rows == out.picture("original").rows

    return check


def total_pixels_size(width, height, megapixels, steps):
    factor = math.sqrt(megapixels * 1024 * 1024 / (width * height))
    return round(width * factor / steps) * steps, round(height * factor / steps) * steps


@case("Imagem metade vermelha/metade azul redimensionada com todos os métodos e nós de "
      "escala; tamanhos calculados como o ComfyUI calcula e as metades preservadas.")
def scale_methods_keep_halves(w):
    halves = stitch(w, solid(w, 20, 20, RED), solid(w, 20, 20, BLUE))
    methods = ("nearest-exact", "bilinear", "area", "bicubic", "lanczos")
    for method in methods:
        save(w, scale(w, halves, method, 80, 40), f"scale_{method}")
    save(w, scale(w, halves, "bilinear", 0, 30), "keep_aspect")
    save(w, scale(w, halves, "bilinear", 20, 20, "center"), "center_crop")
    save(w, scale_by(w, halves, "area", 0.5), "half")
    save(w, scale_by(w, halves, "lanczos", 1.75), "bigger")
    largest = w.add_node(
        ImageScaleToMaxDimension, image=halves, upscale_method="lanczos", largest_size=30
    )
    save(w, largest.output["IMAGE"], "max_dimension")
    total = w.add_node(
        ImageScaleToTotalPixels,
        image=halves,
        upscale_method="bicubic",
        megapixels=0.01,
        resolution_steps=8,
    )
    save(w, total.output["IMAGE"], "total_pixels")

    def check(out):
        sizes = {f"scale_{method}": (80, 40) for method in methods}
        sizes.update(
            keep_aspect=(60, 30),
            center_crop=(20, 20),
            half=(20, 10),
            bigger=(70, 35),
            max_dimension=(30, 15),
            total_pixels=total_pixels_size(40, 20, 0.01, 8),
        )
        for label, size in sizes.items():
            picture = out.picture(label)
            assert picture.size == size, f"{label} is {picture.size}, expected {size}"
            middle = size[1] // 2
            assert_pixel(picture, size[0] * 15 // 100, middle, RED, 3)
            assert_pixel(picture, size[0] * 85 // 100, middle, BLUE, 3)

    return check


@case("SplitImageToTileList corta 128x128 em 9 tiles com sobreposição e "
      "ImageMergeTileList remonta a imagem; tiles e mosaico conferidos.")
def tile_split_and_merge(w):
    tiles = tiles_of(w, quadrants(w, 128), 64, overlap=16)
    save(w, tiles, "tile")
    merged = w.add_node(
        ImageMergeTileList, image_list=tiles, final_width=128, final_height=128, overlap=16
    )
    save(w, merged.output["IMAGE"], "merged")

    def check(out):
        tiles = out.pictures("tile")
        assert len(tiles) == 9
        assert all(tile.size == (64, 64) for tile in tiles)
        for index, color in ((0, RED), (2, GREEN), (6, BLUE), (8, WHITE)):
            assert_pixel(tiles[index], 32, 32, color)
        assert_pixel(tiles[4], 4, 4, RED)
        assert_pixel(tiles[4], 60, 60, WHITE)
        merged = out.picture("merged")
        assert merged.size == (128, 128)
        assert_quadrants(merged)

    return check


@case("Seis cores viram uma lista de tiles, um ImageGrid 3x2 com espaçamento preto e um "
      "ShuffleDataset embaralhado (mesmas cores em outra ordem).")
def grid_collage_and_shuffle(w):
    tiles = tiles_of(w, strip(w, SIX, 64), 64)
    grid = w.add_node(
        ImageGrid, images=tiles, columns=3, cell_width=32, cell_height=32, padding=4
    )
    shuffled = w.add_node(ShuffleDataset, images=tiles, seed=3)
    row = w.add_node(
        ImageGrid,
        images=shuffled.output["images"],
        columns=6,
        cell_width=32,
        cell_height=32,
        padding=0,
    )
    save(w, tiles, "tile")
    save(w, grid.output["images"], "grid")
    save(w, row.output["images"], "shuffled")

    def check(out):
        assert [tile.pixel(32, 32) for tile in out.pictures("tile")] == [rgb(c) for c in SIX]
        grid = out.picture("grid")
        assert grid.size == (104, 68)
        for index, color in enumerate(SIX):
            assert_pixel(grid, 16 + 36 * (index % 3), 16 + 36 * (index // 3), color)
        assert_pixel(grid, 33, 10, BLACK)
        assert_pixel(grid, 10, 33, BLACK)
        row = out.picture("shuffled")
        assert row.size == (192, 32)
        found = [nearest(row.pixel(16 + 32 * index, 16), SIX) for index in range(6)]
        assert sorted(found) == sorted(SIX)

    return check


@case("Lista de seis cores reagrupada com RebatchImages; ImageFromBatch com índice positivo "
      "e negativo e RepeatImageBatch; a ordem das cores tem de ser exata.")
def rebatch_pick_and_repeat(w):
    frames = batch_of(w, SIX)
    picked = w.add_node(ImageFromBatch, image=frames, batch_index=2, length=3)
    repeated = w.add_node(RepeatImageBatch, image=picked.output["IMAGE"], amount=2)
    last = w.add_node(ImageFromBatch, image=frames, batch_index=-1, length=1)
    save(w, picked.output["IMAGE"], "picked")
    save(w, repeated.output["IMAGE"], "repeated")
    save(w, last.output["IMAGE"], "last")

    def check(out):
        def colors(label):
            return [picture.pixel(10, 10) for picture in out.pictures(label)]

        assert colors("picked") == [rgb(c) for c in SIX[2:5]]
        assert colors("repeated") == [rgb(c) for c in SIX[2:5] * 2]
        assert colors("last") == [rgb(MAGENTA)]

    return check


@case("ImageCompositeMasked com máscara recortada por MaskComposite, com origem saindo da "
      "borda e com resize_source.")
def masked_composite_with_offsets(w):
    background, patch = solid(w, 64, 64, BLUE), solid(w, 32, 32, YELLOW)
    left_half = combine_masks(
        w, solid_mask(w, 1.0, 32, 32), solid_mask(w, 1.0, 16, 32), "subtract", x=16
    )
    save(w, paste(w, background, patch, 16, 16, mask=left_half), "masked")
    save(w, paste(w, background, patch, 48, 48), "corner")
    save(w, paste(w, background, patch, mask=left_half, resize=True), "resized")

    def check(out):
        masked = out.picture("masked")
        assert_pixel(masked, 20, 30, YELLOW)
        assert_pixel(masked, 40, 30, BLUE)
        assert_pixel(masked, 8, 8, BLUE)
        corner = out.picture("corner")
        assert_pixel(corner, 56, 56, YELLOW)
        assert_pixel(corner, 40, 56, BLUE)
        resized = out.picture("resized")
        assert_pixel(resized, 8, 32, YELLOW)
        assert_pixel(resized, 56, 32, BLUE)

    return check


@case("ResizeAndPadImage em paisagem, retrato e quadrantes, com preenchimento preto e "
      "branco, e o resultado costurado lado a lado.")
def letterbox_portrait_and_landscape(w):
    def letterbox(image, width, height, color, method):
        return w.add_node(
            ResizeAndPadImage,
            image=image,
            target_width=width,
            target_height=height,
            padding_color=color,
            interpolation=method,
        ).output["IMAGE"]

    landscape = letterbox(solid(w, 64, 32, RED), 48, 48, "black", "nearest-exact")
    portrait = letterbox(solid(w, 32, 64, GREEN), 48, 48, "white", "bilinear")
    squeezed = letterbox(quadrants(w, 64), 40, 24, "black", "area")
    save(w, landscape, "landscape")
    save(w, portrait, "portrait")
    save(w, stitch(w, landscape, portrait), "both")
    save(w, squeezed, "squeezed")

    def check(out):
        landscape = out.picture("landscape")
        for x, y, color in ((24, 6, BLACK), (24, 24, RED), (24, 41, BLACK)):
            assert_pixel(landscape, x, y, color)
        portrait = out.picture("portrait")
        for x, y, color in ((6, 24, WHITE), (24, 24, GREEN), (41, 24, WHITE)):
            assert_pixel(portrait, x, y, color)
        both = out.picture("both")
        assert both.size == (96, 48)
        for x, y, color in ((24, 24, RED), (72, 24, GREEN), (54, 24, WHITE), (24, 4, BLACK)):
            assert_pixel(both, x, y, color)
        squeezed = out.picture("squeezed")
        assert squeezed.size == (40, 24)
        assert_pixel(squeezed, 2, 12, BLACK)
        assert_pixel(squeezed, 37, 12, BLACK)
        for (x, y), color in zip(((14, 6), (26, 6), (14, 18), (26, 18)), QUADRANTS):
            assert_pixel(squeezed, x, y, color, 3)

    return check


@case("Rotações e espelhamentos de uma imagem 96x48 (não quadrada): as dimensões trocam e "
      "cada pixel amostrado vem exatamente da posição esperada.")
def non_square_rotations(w):
    halves = stitch(w, solid(w, 48, 48, RED), solid(w, 48, 48, BLUE))
    for rotation in ROTATIONS:
        save(w, rotate(w, halves, rotation), ROTATION_LABELS[rotation])
    for method in FLIPS:
        save(w, flip(w, halves, method), FLIP_LABELS[method])

    def check(out):
        def color(source_x, _):
            return RED if source_x < 48 else BLUE

        def expect(label, size, source):
            picture = out.picture(label)
            assert picture.size == size
            for x in range(3, size[0], 9):
                for y in range(3, size[1], 9):
                    assert_pixel(picture, x, y, color(*source(x, y)), 0)

        for rotation in ROTATIONS:
            size = (96, 48) if rotation == "180 degrees" else (48, 96)
            expect(
                ROTATION_LABELS[rotation],
                size,
                lambda x, y, r=rotation: rotated_source(r, x, y, 96, 48),
            )
        for method in FLIPS:
            expect(
                FLIP_LABELS[method],
                (96, 48),
                lambda x, y, m=method: flipped_source(m, x, y, 96, 48),
            )

    return check


@case("Ampliação 4x com nearest-exact seguida de redução 4x com area: o resultado tem de "
      "ser idêntico, pixel a pixel, à imagem de partida.")
def nearest_up_area_down_is_lossless(w):
    image = quadrants(w, 32, (ORANGE, CYAN, MAGENTA, GRAY))
    up = scale_by(w, image, "nearest-exact", 4.0)
    down = scale_by(w, up, "area", 0.25)
    save(w, image, "original")
    save(w, up, "up")
    save(w, down, "down")

    def check(out):
        up = out.picture("up")
        assert up.size == (128, 128)
        assert_quadrants(up, (ORANGE, CYAN, MAGENTA, GRAY), 0)
        assert out.picture("down").rows == out.picture("original").rows

    return check


@case("Borda nova nos quatro lados com ImagePadForOutpaint e recorte de volta com "
      "ImageCropV2: a área original volta intacta.")
def outpaint_all_sides_then_crop_back(w):
    image = quadrants(w, 32)
    padded = w.add_node(
        ImagePadForOutpaint, image=image, left=8, top=16, right=24, bottom=8, feathering=0
    ).output["IMAGE"]
    save(w, image, "original")
    save(w, padded, "padded")
    save(w, crop(w, padded, 8, 16, 32, 32), "back")

    def check(out):
        padded = out.picture("padded")
        assert padded.size == (64, 56)
        assert_pixel(padded, 2, 2, gray(0.5))
        assert_pixel(padded, 60, 50, gray(0.5))
        assert_pixel(padded, 16, 24, RED)
        assert_pixel(padded, 32, 40, WHITE)
        assert out.picture("back").rows == out.picture("original").rows

    return check


# ---------------------------------------------------------------- color and filters

BLEND_MODES = ("normal", "multiply", "screen", "overlay", "soft_light", "difference")


def blended(mode, first, second, factor):
    def g(x):
        return ((16 * x - 12) * x + 4) * x if x <= 0.25 else math.sqrt(x)

    def one(a, b):
        if mode == "normal":
            value = b
        elif mode == "multiply":
            value = a * b
        elif mode == "screen":
            value = 1 - (1 - a) * (1 - b)
        elif mode == "overlay":
            value = 2 * a * b if a <= 0.5 else 1 - 2 * (1 - a) * (1 - b)
        elif mode == "soft_light":
            if b <= 0.5:
                value = a - (1 - 2 * b) * a * (1 - a)
            else:
                value = a + (2 * b - 1) * (g(a) - a)
        else:
            value = a - b
        return a * (1 - factor) + value * factor

    return pixel_of(one(a / 255, b / 255) for a, b in zip(rgb(first), rgb(second)))


@case("Os seis modos do ImageBlend com cores intermediárias, conferidos contra as fórmulas "
      "do ComfyUI, e o blend que redimensiona a segunda imagem.")
def blend_modes_board(w):
    first, second = solid(w, 32, 32, 0xCC6633), solid(w, 32, 32, 0x66CC99)
    tiles = []
    for mode in BLEND_MODES:
        factor = 0.25 if mode == "normal" else 1.0
        tile = w.add_node(
            ImageBlend, image1=first, image2=second, blend_mode=mode, blend_factor=factor
        ).output["IMAGE"]
        save(w, tile, f"blend_{mode}")
        tiles.append(tile)
    sheet = tiles[0]
    for tile in tiles[1:]:
        sheet = stitch(w, sheet, tile)
    save(w, sheet, "all_modes")
    resized = w.add_node(
        ImageBlend,
        image1=solid(w, 64, 64, RED),
        image2=solid(w, 8, 8, BLUE),
        blend_mode="normal",
        blend_factor=0.5,
    )
    save(w, resized.output["IMAGE"], "resized")

    def check(out):
        for mode in BLEND_MODES:
            factor = 0.25 if mode == "normal" else 1.0
            expected = blended(mode, 0xCC6633, 0x66CC99, factor)
            assert_pixel(out.picture(f"blend_{mode}"), 16, 16, expected)
        assert out.picture("all_modes").size == (192, 32)
        resized = out.picture("resized")
        assert resized.size == (64, 64)
        assert_pixel(resized, 32, 32, (127, 0, 127))

    return check


@case("Cadeia AdjustBrightness -> AdjustContrast -> NormalizeImages -> ImageInvert com "
      "cada etapa salva e conferida contra a conta feita em Python.")
def brightness_contrast_normalize_invert_chain(w):
    base = solid(w, 16, 16, 0x663399)
    brighter = w.add_node(AdjustBrightness, images=base, factor=1.5).output["images"]
    contrast = w.add_node(AdjustContrast, images=brighter, factor=2.0).output["images"]
    normalized = w.add_node(NormalizeImages, images=contrast, mean=0.2, std=0.5)
    inverted = w.add_node(ImageInvert, image=normalized.output["images"])
    save(w, brighter, "brighter")
    save(w, contrast, "contrast")
    save(w, normalized.output["images"], "normalized")
    save(w, inverted.output["IMAGE"], "inverted")

    def check(out):
        values = [channel / 255 for channel in rgb(0x663399)]
        values = [min(max(v * 1.5, 0), 1) for v in values]
        assert_pixel(out.picture("brighter"), 8, 8, pixel_of(values))
        values = [min(max((v - 0.5) * 2 + 0.5, 0), 1) for v in values]
        assert_pixel(out.picture("contrast"), 8, 8, pixel_of(values))
        values = [(v - 0.2) / 0.5 for v in values]
        assert_pixel(out.picture("normalized"), 8, 8, pixel_of(values))
        assert_pixel(out.picture("inverted"), 8, 8, pixel_of(1 - v for v in values))

    return check


SOLID_FILTERS = {
    "blur": lambda w, i: w.add_node(ImageBlur, image=i, blur_radius=3, sigma=2.0).output["IMAGE"],
    "sharpen": lambda w, i: w.add_node(
        ImageSharpen, image=i, sharpen_radius=2, sigma=1.0, alpha=0.5
    ).output["IMAGE"],
    "erode": lambda w, i: w.add_node(
        Morphology, image=i, operation="erode", kernel_size=5
    ).output["IMAGE"],
    "dilate": lambda w, i: w.add_node(
        Morphology, image=i, operation="dilate", kernel_size=5
    ).output["IMAGE"],
    "open": lambda w, i: w.add_node(
        Morphology, image=i, operation="open", kernel_size=5
    ).output["IMAGE"],
    "close": lambda w, i: w.add_node(
        Morphology, image=i, operation="close", kernel_size=5
    ).output["IMAGE"],
    "quantize": lambda w, i: w.add_node(
        ImageQuantize, image=i, colors=8, dither="none"
    ).output["IMAGE"],
    "no_noise": lambda w, i: w.add_node(
        ImageAddNoise, image=i, seed=1, strength=0.0
    ).output["IMAGE"],
    "same_space": lambda w, i: w.add_node(
        ImageColorSpace, image=i, source="sRGB", destination="sRGB"
    ).output["IMAGE"],
    "brightness_1": lambda w, i: w.add_node(
        AdjustBrightness, images=i, factor=1.0
    ).output["images"],
    "contrast_1": lambda w, i: w.add_node(AdjustContrast, images=i, factor=1.0).output["images"],
    "scale_1": lambda w, i: scale_by(w, i, "bicubic", 1.0),
}


@case("Doze filtros (blur, sharpen, morfologia, quantização, ruído zero, etc.) aplicados a "
      "uma cor sólida: nenhum pode alterar a cor, nem nas bordas.")
def filters_keep_solid_colors(w):
    base = solid(w, 48, 48, 0x3366CC)
    for label, apply in SOLID_FILTERS.items():
        save(w, apply(w, base), label)

    def check(out):
        for label in SOLID_FILTERS:
            picture = out.picture(label)
            assert picture.size == (48, 48), label
            for x, y in ((24, 24), (0, 0), (47, 47)):
                assert_pixel(picture, x, y, 0x3366CC)

    return check


@case("Canny e as morfologias gradient/top_hat/bottom_hat sobre cor sólida: sem bordas, "
      "o resultado tem de ser preto.")
def edge_filters_cancel_on_solid(w):
    base = solid(w, 48, 48, 0x3366CC)
    canny = w.add_node(Canny, image=base, low_threshold=0.4, high_threshold=0.8)
    save(w, canny.output["IMAGE"], "canny")
    for operation in ("gradient", "top_hat", "bottom_hat"):
        node = w.add_node(Morphology, image=base, operation=operation, kernel_size=3)
        save(w, node.output["IMAGE"], operation)

    def check(out):
        for label in ("canny", "gradient", "top_hat", "bottom_hat"):
            picture = out.picture(label)
            for x, y in ((24, 24), (1, 1), (46, 46)):
                assert_pixel(picture, x, y, BLACK)

    return check


@case("ImageAddNoise com a mesma semente duas vezes e uma semente diferente: mesmas "
      "sementes dão imagens idênticas, sementes diferentes não.")
def seeded_image_noise(w):
    base = solid(w, 32, 32, GRAY)
    for label, seed in (("seed_7", 7), ("seed_7_again", 7), ("seed_8", 8)):
        noisy = w.add_node(ImageAddNoise, image=base, seed=seed, strength=0.3)
        save(w, noisy.output["IMAGE"], label)

    def check(out):
        first, again, other = (out.picture(label) for label in ("seed_7", "seed_7_again", "seed_8"))
        assert first.rows == again.rows
        assert first.rows != other.rows
        assert len(set(first.pixels())) > 50

    return check


@case("Degradê criado escalando um 2x1 preto/branco com bilinear, quantizado em 2 cores "
      "sem dithering, com Floyd-Steinberg e com Bayer.")
def quantize_gradient_with_dithers(w):
    ramp = scale(w, stitch(w, solid(w, 1, 1, BLACK), solid(w, 1, 1, WHITE)), "bilinear", 64, 16)
    save(w, ramp, "ramp")
    dithers = ("none", "floyd-steinberg", "bayer-4")
    for dither in dithers:
        quantized = w.add_node(ImageQuantize, image=ramp, colors=2, dither=dither)
        save(w, quantized.output["IMAGE"], f"quantized_{dither}")

    def check(out):
        ramp = out.picture("ramp")
        assert ramp.size == (64, 16)
        assert ramp.pixel(4, 8)[0] < ramp.pixel(32, 8)[0] < ramp.pixel(60, 8)[0]
        for dither in dithers:
            picture = out.picture(f"quantized_{dither}")
            colors = set(picture.pixels())
            assert len(colors) <= 2, f"{dither} produced {len(colors)} colors"
            if dither == "none":
                assert picture.pixel(2, 8)[0] < 128 < picture.pixel(61, 8)[0]
            else:
                middle = {value[0] < 128 for value in picture.pixels((24, 0, 40, 16))}
                assert middle == {True, False}, f"{dither} did not mix the two colors"

    return check


@case("ImageRGBToYUV e ImageYUVToRGB: o Y/Cb/Cr de vermelho, verde, azul e branco bate com "
      "a matriz BT.601 e a volta para RGB recupera os quadrantes.")
def yuv_round_trip_and_luma(w):
    image = quadrants(w, 32)
    yuv = w.add_node(ImageRGBToYUV, image=image)
    back = w.add_node(ImageYUVToRGB, Y=yuv.output["Y"], U=yuv.output["U"], V=yuv.output["V"])
    for channel in ("Y", "U", "V"):
        save(w, yuv.output[channel], f"channel_{channel}")
    save(w, back.output["IMAGE"], "back")

    def check(out):
        assert_quadrants(out.picture("back"), tolerance=3)
        for channel in ("Y", "U", "V"):
            picture = out.picture(f"channel_{channel}")
            for (x, y), color in zip(quarter_points(32, 32), QUADRANTS):
                r, g, b = (value / 255 for value in rgb(color))
                luma = 0.299 * r + 0.587 * g + 0.114 * b
                value = {
                    "Y": luma,
                    "U": (b - luma) * 0.564 + 0.5,
                    "V": (r - luma) * 0.713 + 0.5,
                }[channel]
                assert_pixel(picture, x, y, gray(value), 3)

    return check


def srgb_to_linear(value: float) -> float:
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


@case("ImageColorSpace sRGB -> linear -> sRGB em quatro cores: o cinza linear escurece "
      "como a curva sRGB manda e a volta recupera as cores.")
def color_space_round_trip(w):
    colors = (GRAY, 0x40C080, 0xC04020, 0x204060)
    image = quadrants(w, 32, colors)
    linear = w.add_node(ImageColorSpace, image=image, source="sRGB", destination="linear")
    back = w.add_node(
        ImageColorSpace, image=linear.output["IMAGE"], source="linear", destination="sRGB"
    )
    save(w, linear.output["IMAGE"], "linear")
    save(w, back.output["IMAGE"], "back")

    def check(out):
        assert_quadrants(out.picture("back"), colors, 3)
        linear = out.picture("linear")
        assert_pixel(linear, 8, 8, gray(srgb_to_linear(128 / 255)), 3)

    return check


@case("Canny num tabuleiro preto/branco: bordas brancas exatamente nas linhas que separam "
      "os quadrantes e preto longe delas.")
def canny_finds_quadrant_borders(w):
    board_image = quadrants(w, 64, (BLACK, WHITE, WHITE, BLACK))
    edges = w.add_node(Canny, image=board_image, low_threshold=0.1, high_threshold=0.3)
    save(w, edges.output["IMAGE"], "edges")

    def check(out):
        edges = out.picture("edges")
        assert edges.size == (64, 64)
        for x, y in quarter_points(64, 64):
            assert_pixel(edges, x, y, BLACK)
        # Kornia keeps the whole vertical border, but where the four quadrants
        # meet it drops half of the horizontal one, so only part of it is required.
        rows = sum(any(edges.pixel(x, y)[0] > 128 for x in range(29, 35)) for y in range(64))
        columns = sum(any(edges.pixel(x, y)[0] > 128 for y in range(29, 35)) for x in range(64))
        assert rows >= 58, f"the vertical border shows up in {rows} of 64 rows"
        assert columns >= 16, f"the horizontal border shows up in {columns} of 64 columns"

    return check


@case("Morphology dilate/erode/open/close num quadrado branco 8x8: cresce e encolhe "
      "exatamente 1 pixel por lado, abrir e fechar preservam o quadrado.")
def morphology_grows_and_shrinks_square(w):
    square = paste(w, solid(w, 32, 32, BLACK), solid(w, 8, 8, WHITE), 12, 12)
    for operation in ("dilate", "erode", "open", "close"):
        node = w.add_node(Morphology, image=square, operation=operation, kernel_size=3)
        save(w, node.output["IMAGE"], operation)

    def check(out):
        expected = {
            "dilate": {(11, 11): WHITE, (20, 20): WHITE, (10, 10): BLACK, (21, 21): BLACK},
            "erode": {(13, 13): WHITE, (18, 18): WHITE, (12, 12): BLACK, (19, 19): BLACK},
            "open": {(12, 12): WHITE, (19, 19): WHITE, (11, 11): BLACK},
            "close": {(12, 12): WHITE, (19, 19): WHITE, (20, 20): BLACK},
        }
        for operation, points in expected.items():
            picture = out.picture(operation)
            for (x, y), color in points.items():
                assert_pixel(picture, x, y, color)

    return check


PORTER_DUFF = {
    "SRC_OVER": ((127, 0, 127), 1.0),
    "DST_OVER": ((0, 0, 255), 1.0),
    "SRC_IN": ((255, 0, 0), 0.5),
    "XOR": ((0, 0, 255), 0.5),
    "MULTIPLY": ((0, 0, 127), 1.0),
    "SCREEN": ((127, 0, 255), 1.0),
    "CLEAR": ((0, 0, 0), 0.0),
}


@case("PorterDuffImageComposite em sete modos com fonte vermelha 50% e destino azul opaco; "
      "imagem e máscara (1 - alfa) resultantes conferidas com as equações Porter-Duff.")
def porter_duff_modes(w):
    source, destination = solid(w, 32, 32, RED), solid(w, 32, 32, BLUE)
    half, opaque = solid_mask(w, 0.5, 32, 32), solid_mask(w, 0.0, 32, 32)
    for mode in PORTER_DUFF:
        node = w.add_node(
            PorterDuffImageComposite,
            source=source,
            source_alpha=half,
            destination=destination,
            destination_alpha=opaque,
            mode=mode,
        )
        save(w, node.output["IMAGE"], f"image_{mode}")
        save(w, mask_image(w, node.output["MASK"]), f"mask_{mode}")

    def check(out):
        # Like its mask inputs, the node's MASK output is 1 - alpha.
        for mode, (color, alpha) in PORTER_DUFF.items():
            assert_pixel(out.picture(f"image_{mode}"), 16, 16, color)
            assert_pixel(out.picture(f"mask_{mode}"), 16, 16, gray(1 - alpha))

    return check


@case("JoinImageWithAlpha gera PNG RGBA, ImageInvert preserva o alfa e SplitImageWithAlpha "
      "devolve a máscara invertida (1 - alfa).")
def alpha_join_and_split(w):
    joined = w.add_node(JoinImageWithAlpha, image=solid(w, 32, 32, RED), alpha=solid_mask(w, 0.25, 32, 32))
    inverted = w.add_node(ImageInvert, image=joined.output["IMAGE"])
    split = w.add_node(SplitImageWithAlpha, image=joined.output["IMAGE"])
    save(w, joined.output["IMAGE"], "rgba")
    save(w, inverted.output["IMAGE"], "inverted")
    save(w, split.output["IMAGE"], "rgb")
    save(w, mask_image(w, split.output["MASK"]), "mask")

    def check(out):
        rgba = out.picture("rgba")
        assert rgba.channels == 4
        assert_pixel(rgba, 5, 5, RED)
        assert abs(rgba.alpha(5, 5) - 191) <= 1
        inverted = out.picture("inverted")
        assert_pixel(inverted, 5, 5, CYAN)
        assert abs(inverted.alpha(5, 5) - 191) <= 1
        rgb_part = out.picture("rgb")
        assert rgb_part.channels == 3
        assert_pixel(rgb_part, 5, 5, RED)
        assert_pixel(out.picture("mask"), 5, 5, gray(0.25))

    return check


# ---------------------------------------------------------------- masks


@case("Álgebra de máscaras com MaskComposite (add, subtract, multiply, xor, or, and) com "
      "deslocamentos, inclusive saindo da borda; cada região conferida.")
def mask_algebra_board(w):
    mask = solid_mask(w, 0.5, 64, 64)
    mask = combine_masks(w, mask, solid_mask(w, 0.25, 32, 32), "add")
    mask = combine_masks(w, mask, solid_mask(w, 1.0, 32, 32), "subtract", x=32)
    mask = combine_masks(w, mask, solid_mask(w, 0.5, 32, 32), "multiply", y=32)
    full = solid_mask(w, 1.0, 64, 64)
    results = {
        "mixed": mask,
        "xor": combine_masks(w, mask, full, "xor"),
        "or_corner": combine_masks(w, mask, solid_mask(w, 1.0, 16, 16), "or", 56, 56),
        "and": combine_masks(w, mask, full, "and"),
    }
    for label, result in results.items():
        save(w, mask_image(w, result), label)

    def check(out):
        expected = {
            "mixed": (0.75, 0.0, 0.25, 0.5),
            "xor": (0.0, 1.0, 1.0, 1.0),
            "and": (1.0, 0.0, 0.0, 0.0),
        }
        for label, values in expected.items():
            picture = out.picture(label)
            for (x, y), value in zip(quarter_points(64, 64), values):
                assert_pixel(picture, x, y, gray(value), 1)
        corner = out.picture("or_corner")
        assert_pixel(corner, 60, 60, WHITE, 1)
        assert_pixel(corner, 40, 40, gray(0.5), 1)

    return check


@case("FeatherMask em três lados, CropMask e ThresholdMask; o degradê linear do feather "
      "conferido coluna a coluna.")
def mask_feather_crop_threshold(w):
    feathered = w.add_node(
        FeatherMask, mask=solid_mask(w, 1.0, 32, 32), left=8, top=0, right=4, bottom=2
    ).output["MASK"]
    cropped = w.add_node(CropMask, mask=feathered, x=0, y=8, width=16, height=8)
    threshold = w.add_node(ThresholdMask, mask=feathered, value=0.3)
    save(w, mask_image(w, feathered), "feathered")
    save(w, mask_image(w, cropped.output["MASK"]), "cropped")
    save(w, mask_image(w, threshold.output["MASK"]), "threshold")

    def check(out):
        feathered = out.picture("feathered")
        expected = {
            (0, 10): 1 / 8, (3, 10): 4 / 8, (7, 10): 1.0, (31, 10): 1 / 4,
            (29, 10): 3 / 4, (15, 31): 1 / 2, (0, 31): 1 / 16,
        }
        for (x, y), value in expected.items():
            assert_pixel(feathered, x, y, gray(value), 1)
        cropped = out.picture("cropped")
        assert cropped.size == (16, 8)
        assert_pixel(cropped, 3, 2, gray(0.5), 1)
        threshold = out.picture("threshold")
        assert_pixel(threshold, 0, 10, BLACK)
        assert_pixel(threshold, 3, 10, WHITE)
        assert_pixel(threshold, 31, 10, BLACK)

    return check


@case("GrowMask com kernel quadrado e com cantos arredondados, e encolhendo; as bordas do "
      "quadrado conferidas pixel a pixel.")
def grow_mask_square_vs_tapered(w):
    square = combine_masks(w, solid_mask(w, 0.0, 32, 32), solid_mask(w, 1.0, 8, 8), "add", 12, 12)
    variants = {"square": (2, False), "tapered": (2, True), "shrunk": (-2, False)}
    for label, (expand, tapered) in variants.items():
        grown = w.add_node(GrowMask, mask=square, expand=expand, tapered_corners=tapered)
        save(w, mask_image(w, grown.output["MASK"]), label)

    def check(out):
        expected = {
            "square": {(10, 10): WHITE, (21, 21): WHITE, (9, 9): BLACK},
            "tapered": {(10, 12): WHITE, (11, 11): WHITE, (10, 10): BLACK},
            "shrunk": {(14, 14): WHITE, (17, 17): WHITE, (13, 13): BLACK},
        }
        for label, points in expected.items():
            picture = out.picture(label)
            for (x, y), color in points.items():
                assert_pixel(picture, x, y, color)

    return check


@case("ImageToMask nos canais vermelho, verde e azul de quadrantes coloridos: cada máscara "
      "acende só onde o canal existe.")
def image_channels_to_masks(w):
    image = quadrants(w, 32)
    for channel in ("red", "green", "blue"):
        mask = w.add_node(ImageToMask, image=image, channel=channel).output["MASK"]
        save(w, mask_image(w, mask), f"mask_{channel}")

    def check(out):
        for index, channel in enumerate(("red", "green", "blue")):
            picture = out.picture(f"mask_{channel}")
            for (x, y), color in zip(quarter_points(32, 32), QUADRANTS):
                assert_pixel(picture, x, y, WHITE if rgb(color)[index] else BLACK)

    return check


@case("ImageColorToMask isola o quadrante verde; a máscara pinta esse quadrante de amarelo "
      "com ImageCompositeMasked; uma cor ausente gera máscara vazia.")
def color_to_mask_picks_quadrant(w):
    image = quadrants(w, 32)
    green = w.add_node(ImageColorToMask, image=image, color=GREEN).output["MASK"]
    absent = w.add_node(ImageColorToMask, image=image, color=0x123456).output["MASK"]
    inverted = w.add_node(InvertMask, mask=absent).output["MASK"]
    save(w, mask_image(w, green), "green_mask")
    save(w, mask_image(w, inverted), "inverted_absent")
    save(w, paste(w, image, solid(w, 32, 32, YELLOW), mask=green), "painted")

    def check(out):
        mask = out.picture("green_mask")
        for (x, y), color in zip(quarter_points(32, 32), QUADRANTS):
            assert_pixel(mask, x, y, WHITE if color == GREEN else BLACK)
        assert set(out.picture("inverted_absent").pixels()) == {rgb(WHITE)}
        assert_quadrants(out.picture("painted"), (RED, YELLOW, BLUE, WHITE))

    return check


@case("Upload de um PNG RGBA pelo worker: LoadImage devolve RGB intacto e máscara 1 - alfa, "
      "LoadImageMask lê os canais alfa, vermelho e verde.",
      uploads={"probe_rgba.png": RGBA_PNG})
def uploaded_rgba_masks(w):
    loaded = w.add_node(LoadImage, image="probe_rgba.png")
    save(w, loaded.output["IMAGE"], "image")
    save(w, mask_image(w, loaded.output["MASK"]), "mask")
    for channel in ("alpha", "red", "green"):
        mask = w.add_node(LoadImageMask, image="probe_rgba.png", channel=channel)
        save(w, mask_image(w, mask.output["MASK"]), f"channel_{channel}")

    def check(out):
        image = out.picture("image")
        assert image.size == (32, 32)
        assert_pixel(image, 4, 4, (200, 100, 50), 0)
        assert_pixel(image, 28, 4, (200, 100, 50), 0)
        for label in ("mask", "channel_alpha"):
            mask = out.picture(label)
            assert_pixel(mask, 4, 4, BLACK, 1)
            assert_pixel(mask, 28, 4, gray(1 - 64 / 255), 1)
        assert_pixel(out.picture("channel_red"), 10, 10, (200, 200, 200), 1)
        assert_pixel(out.picture("channel_green"), 10, 10, (100, 100, 100), 1)

    return check


@case("Máscara com FeatherMask usada no ImageCompositeMasked: a mistura preto/branco "
      "segue a rampa da máscara coluna por coluna.")
def feathered_mask_blends_composite(w):
    mask = w.add_node(FeatherMask, mask=solid_mask(w, 1.0, 32, 32), left=8).output["MASK"]
    save(w, paste(w, solid(w, 32, 32, BLACK), solid(w, 32, 32, WHITE), mask=mask), "blend")

    def check(out):
        picture = out.picture("blend")
        for column in (0, 3, 6, 10, 25):
            assert_pixel(picture, column, 16, gray(min((column + 1) / 8, 1.0)), 1)

    return check


# ---------------------------------------------------------------- uploaded images


@case("PNG com um padrão diferente em cada pixel enviado ao worker e carregado com "
      "LoadImage: salvo de volta, tem de ser idêntico pixel a pixel.",
      uploads={"probe_pattern.png": PATTERN_PNG})
def uploaded_image_round_trip(w):
    loaded = w.add_node(LoadImage, image="probe_pattern.png")
    save(w, loaded.output["IMAGE"], "loaded")
    save(w, mask_image(w, loaded.output["MASK"]), "mask")

    def check(out):
        loaded = out.picture("loaded")
        assert loaded.size == (24, 16)
        for y in range(16):
            for x in range(24):
                assert_pixel(loaded, x, y, pattern(x, y), 0)
        assert set(out.picture("mask").pixels()) == {rgb(BLACK)}

    return check


@case("O padrão enviado é girado e espelhado no ComfyUI; todos os pixels de todas as saídas "
      "são conferidos contra a transformação exata.",
      uploads={"probe_pattern.png": PATTERN_PNG})
def uploaded_image_geometry(w):
    loaded = w.add_node(LoadImage, image="probe_pattern.png").output["IMAGE"]
    for rotation in ROTATIONS:
        save(w, rotate(w, loaded, rotation), ROTATION_LABELS[rotation])
    for method in FLIPS:
        save(w, flip(w, loaded, method), FLIP_LABELS[method])

    def check(out):
        def expect(label, size, source):
            picture = out.picture(label)
            assert picture.size == size
            for y in range(size[1]):
                for x in range(size[0]):
                    assert_pixel(picture, x, y, pattern(*source(x, y)), 0)

        for rotation in ROTATIONS:
            size = (24, 16) if rotation == "180 degrees" else (16, 24)
            expect(
                ROTATION_LABELS[rotation],
                size,
                lambda x, y, r=rotation: rotated_source(r, x, y, 24, 16),
            )
        for method in FLIPS:
            expect(
                FLIP_LABELS[method], (24, 16), lambda x, y, m=method: flipped_source(m, x, y, 24, 16)
            )

    return check


@case("O padrão enviado é ampliado 3x com nearest-exact e recortado com ImageCropV2 e "
      "CenterCropImages; cada pixel tem de vir do lugar certo.",
      uploads={"probe_pattern.png": PATTERN_PNG})
def uploaded_image_scale_and_crop_exact(w):
    loaded = w.add_node(LoadImage, image="probe_pattern.png").output["IMAGE"]
    save(w, scale_by(w, loaded, "nearest-exact", 3.0), "tripled")
    save(w, crop(w, loaded, 5, 3, 8, 6), "cropped")
    centered = w.add_node(CenterCropImages, images=loaded, width=12, height=8)
    save(w, centered.output["images"], "centered")

    def check(out):
        def expect(label, size, source):
            picture = out.picture(label)
            assert picture.size == size
            for y in range(size[1]):
                for x in range(size[0]):
                    assert_pixel(picture, x, y, pattern(*source(x, y)), 0)

        expect("tripled", (72, 48), lambda x, y: (x // 3, y // 3))
        expect("cropped", (8, 6), lambda x, y: (x + 5, y + 3))
        expect("centered", (12, 8), lambda x, y: (x + 6, y + 4))

    return check


@case("Faixa vermelha enviada ao worker vira máscara com LoadImageMask (canal red) e "
      "decide onde o amarelo cobre o azul.",
      uploads={"probe_stripe.png": STRIPE_PNG})
def uploaded_mask_drives_composite(w):
    mask = w.add_node(LoadImageMask, image="probe_stripe.png", channel="red").output["MASK"]
    save(w, paste(w, solid(w, 24, 16, BLUE), solid(w, 24, 16, YELLOW), mask=mask), "composite")
    save(w, mask_image(w, mask), "mask")

    def check(out):
        composite = out.picture("composite")
        assert_pixel(composite, 5, 8, YELLOW, 0)
        assert_pixel(composite, 18, 8, BLUE, 0)
        assert_pixel(out.picture("mask"), 5, 8, WHITE, 0)

    return check


# ---------------------------------------------------------------- text, math and logic


@case("Quadro de 14 verificações de texto (concatenar, substring com índice negativo, "
      "maiúsculas/minúsculas, trim, replace, unicode) feitas no ComfyUI.")
def string_basics_board(w):
    def substring(text, start, end):
        return w.add_node(StringSubstring, string=text, start=start, end=end).output["STRING"]

    def case_of(text, mode):
        return w.add_node(CaseConverter, string=text, mode=mode).output["STRING"]

    def trim(text, mode):
        return w.add_node(StringTrim, string=text, mode=mode).output["STRING"]

    def concatenate(first, second, delimiter=""):
        return w.add_node(
            StringConcatenate, string_a=first, string_b=second, delimiter=delimiter
        ).output["STRING"]

    chained = concatenate(trim("  ab ", "Both"), case_of("cd", "UPPERCASE"))
    replaced = w.add_node(StringReplace, string="a-b-c", find="-", replace="+")
    checks = {
        "concatenate": equals(w, concatenate("comfy", "ui", "-"), "comfy-ui"),
        "substring": equals(w, substring("workflow", 4, 8), "flow"),
        "negative_start": equals(w, substring("workflow", -4, 8), "flow"),
        "upper": equals(w, case_of("hello world", "UPPERCASE"), "HELLO WORLD"),
        "lower": equals(w, case_of("HeLLo", "lowercase"), "hello"),
        "capitalize": equals(w, case_of("hello world", "Capitalize"), "Hello world"),
        "title": equals(w, case_of("hello world", "Title Case"), "Hello World"),
        "trim_both": equals(w, trim("  pad  ", "Both"), "pad"),
        "trim_left": equals(w, trim("  pad  ", "Left"), "pad  "),
        "trim_right": equals(w, trim("  pad  ", "Right"), "  pad"),
        "replace": equals(w, replaced.output["STRING"], "a+b+c"),
        "unicode_length": calc(w, "a == 4", a=length(w, "ação"))["BOOL"],
        "chained": equals(w, chained, "abCD"),
        "chained_length": calc(w, "a == 4", a=length(w, chained))["BOOL"],
    }
    return board(w, checks)


@case("Quadro de comparações: StringCompare (começa/termina/igual, com e sem caixa), "
      "StringContains e a veracidade Python do ComfyNotNode.")
def string_compare_contains_board(w):
    def compare(first, second, mode, sensitive=True):
        return w.add_node(
            StringCompare, string_a=first, string_b=second, mode=mode, case_sensitive=sensitive
        ).output["BOOLEAN"]

    def contains(text, part, sensitive=True):
        return w.add_node(
            StringContains, string=text, substring=part, case_sensitive=sensitive
        ).output["contains"]

    empty = w.add_node(StringSubstring, string="abc", start=0, end=0).output["STRING"]
    zero = w.add_node(PrimitiveInt, value=0).output["INT"]
    half = w.add_node(PrimitiveFloat, value=0.5).output["FLOAT"]
    checks = {
        "starts_any_case": compare("ComfyUI", "comfy", "Starts With", False),
        "starts_exact_case": negate(w, compare("ComfyUI", "comfy", "Starts With")),
        "ends": compare("pipeline.json", ".json", "Ends With"),
        "equal_any_case": compare("abc", "ABC", "Equal", False),
        "equal_exact_case": negate(w, compare("abc", "ABC", "Equal")),
        "contains": contains("stable diffusion xl", "diffusion"),
        "contains_exact_case": negate(w, contains("Stable", "stable")),
        "contains_any_case": contains("Stable", "stable", False),
        "empty_is_falsy": negate(w, empty),
        "zero_is_falsy": negate(w, zero),
        "half_is_truthy": negate(w, negate(w, half)),
    }
    return board(w, checks)


@case("Quadro de regex: RegexMatch com flags (ignorecase, multiline, dotall), RegexExtract "
      "nos quatro modos, RegexReplace com grupos e padrão inválido.")
def regex_board(w):
    def match(text, regex, **flags):
        return w.add_node(RegexMatch, string=text, regex_pattern=regex, **flags).output["matches"]

    def extract(text, regex, mode, **options):
        return w.add_node(
            RegexExtract, string=text, regex_pattern=regex, mode=mode, **options
        ).output["STRING"]

    def replace(text, regex, replacement, **flags):
        return w.add_node(
            RegexReplace, string=text, regex_pattern=regex, replace=replacement, **flags
        ).output["STRING"]

    invalid = extract("anything", "(", "First Match")
    checks = {
        "match": match("Order #1234 shipped", r"#\d+"),
        "no_match": negate(w, match("abc", r"\d")),
        "ignore_case": match("HELLO", "hello", case_insensitive=True),
        "exact_case": negate(w, match("HELLO", "hello", case_insensitive=False)),
        "multiline": match("line1\nline2", "^line2$", multiline=True),
        "single_line": negate(w, match("line1\nline2", "^line2$", multiline=False)),
        "dotall": match("a\nb", "a.b", dotall=True),
        "no_dotall": negate(w, match("a\nb", "a.b", dotall=False)),
        "first_match": equals(w, extract("size=24px; scale=2x", r"\d+", "First Match"), "24"),
        "all_matches": equals(w, extract("size=24px; scale=2x", r"\d+", "All Matches"), "24\n2"),
        "first_group": equals(
            w, extract("key=value", r"(\w+)=(\w+)", "First Group", group_index=2), "value"
        ),
        "all_groups": equals(
            w, extract("a=1 b=2 c=3", r"(\w)=(\d)", "All Groups", group_index=1), "a\nb\nc"
        ),
        "tuple_matches": equals(w, extract("a=1 b=2", r"(\w)=(\d)", "All Matches"), "a\nb"),
        "replace_groups": equals(
            w, replace("2026-10-16", r"(\d+)-(\d+)-(\d+)", r"\3/\2/\1"), "16/10/2026"
        ),
        "replace_case": equals(w, replace("aAaA", "a", "b", case_insensitive=False), "bAbA"),
        "invalid_pattern": calc(w, "a == 0", a=length(w, invalid))["BOOL"],
    }
    return board(w, checks)


@case("Quadro de JSON e números: JsonExtractString (texto solto, bloco Markdown, chave "
      "ausente) e ComfyNumberConvert de texto, float negativo e booleano.")
def json_and_number_conversion_board(w):
    def from_json(text, key):
        return w.add_node(JsonExtractString, json_string=text, key=key).output["STRING"]

    def convert(value):
        return w.add_node(ComfyNumberConvert, value=value).output

    width = from_json('noise {"w": 40} trailing', "w")
    decimal = convert(w.add_node(PrimitiveString, value=" 2.75 ").output["STRING"])
    flag = convert(w.add_node(PrimitiveBoolean, value=True).output["BOOLEAN"])
    negative = convert(w.add_node(PrimitiveFloat, value=-3.9).output["FLOAT"])
    seven = convert(w.add_node(PrimitiveInt, value=7).output["INT"])
    checks = {
        "name": equals(w, from_json('{"w": 40, "name": "comfy"}', "name"), "comfy"),
        "loose_text": equals(w, width, "40"),
        "markdown": equals(w, from_json('```json\n{"steps": 30}\n```', "steps"), "30"),
        "missing_key": equals(w, from_json('{"a": 1}', "missing"), ""),
        "text_to_int": calc(w, "a == 40", a=convert(width)["INT"])["BOOL"],
        "decimal_float": calc(w, "a == 2.75", a=decimal["FLOAT"])["BOOL"],
        "decimal_int": calc(w, "a == 2", a=decimal["INT"])["BOOL"],
        "bool_to_float": calc(w, "a == 1", a=flag["FLOAT"])["BOOL"],
        "negative_truncates": calc(w, "a == -3", a=negative["INT"])["BOOL"],
        "int_to_float": calc(w, "a == 7.0", a=seven["FLOAT"])["BOOL"],
    }
    return board(w, checks)


@case("Quadro de ComfyMathExpression: sum/min/max/abs/round/pow/sqrt/ceil/floor/log2/int, "
      "ternário, módulo e Pitágoras; saídas INT e FLOAT viram tamanhos de imagem.")
def math_expression_board(w):
    truth = w.add_node(PrimitiveBoolean, value=True).output["BOOLEAN"]
    checks = {
        "sum": calc(w, "sum(values) == 10", a=1, b=2, c=3, d=4)["BOOL"],
        "min": calc(w, "min(a, b, c) == -2", a=5, b=-2, c=3)["BOOL"],
        "max": calc(w, "max(values) == 9.5", a=1, b=9.5, c=3)["BOOL"],
        "abs": calc(w, "abs(a) == 7", a=-7)["BOOL"],
        "round": calc(w, "round(a, 1) == 3.1", a=3.14159)["BOOL"],
        "pow": calc(w, "pow(a, b) == 1024", a=2, b=10)["BOOL"],
        "sqrt": calc(w, "sqrt(a) == 12", a=144)["BOOL"],
        "ceil_floor": calc(w, "ceil(a) == 3 and floor(a) == 2", a=2.5)["BOOL"],
        "log2": calc(w, "log2(a) == 8", a=256)["BOOL"],
        "int": calc(w, "int(a / b) == 3", a=10, b=3)["BOOL"],
        "ternary": calc(w, "(a if b else c) == 42", a=42, b=truth, c=0)["BOOL"],
        "modulo": calc(w, "a % b == 1", a=10, b=3)["BOOL"],
        "pythagoras": calc(w, "a ** 2 + b ** 2 == c ** 2", a=3, b=4, c=5)["BOOL"],
    }
    check_board = board(w, checks)
    width = calc(w, "a * 2.6", a=10)["INT"]
    factor = calc(w, "a / 4", a=2)["FLOAT"]
    save(w, solid(w, width, 8, ORANGE), "from_int")
    save(w, scale_by(w, solid(w, 64, 64, ORANGE), "nearest-exact", factor), "from_float")

    def check(out):
        check_board(out)
        assert out.picture("from_int").size == (26, 8)
        assert out.picture("from_float").size == (32, 32)

    return check


@case("ComfySwitchNode é preguiçoso: o ramo não escolhido divide por zero e mesmo assim o "
      "job passa; switches escolhem tamanhos e imagens, com ComfyNotNode.")
def switch_selects_lazily(w):
    broken = calc(w, "a / b", a=1, b=0)["INT"]
    forty_eight = w.add_node(PrimitiveInt, value=48).output["INT"]
    thirty_two = w.add_node(PrimitiveInt, value=32).output["INT"]
    off = w.add_node(ComfySwitchNode, switch=False, on_true=broken, on_false=forty_eight)
    on = w.add_node(ComfySwitchNode, switch=True, on_true=thirty_two, on_false=broken)
    save(w, solid(w, off.output["output"], on.output["output"], GREEN), "sized")
    found = w.add_node(StringContains, string="stable diffusion", substring="diffusion")
    picked = w.add_node(
        ComfySwitchNode,
        switch=found.output["contains"],
        on_true=solid(w, 8, 8, GREEN),
        on_false=solid(w, 8, 8, RED),
    )
    save(w, picked.output["output"], "picked")
    flipped = negate(w, w.add_node(PrimitiveBoolean, value=True).output["BOOLEAN"])
    other = w.add_node(
        ComfySwitchNode, switch=flipped, on_true=solid(w, 8, 8, RED), on_false=solid(w, 8, 8, BLUE)
    )
    save(w, other.output["output"], "negated")

    def check(out):
        sized = out.picture("sized")
        assert sized.size == (48, 32)
        assert_pixel(sized, 10, 10, GREEN)
        assert_pixel(out.picture("picked"), 4, 4, GREEN)
        assert_pixel(out.picture("negated"), 4, 4, BLUE)

    return check


@case("GetImageSize alimenta ComfyMathExpression e EmptyImage: largura, altura e batch "
      "derivados no servidor, inclusive ImageScale com altura 0 (proporcional).")
def math_drives_image_size(w):
    base = solid(w, 40, 24, ORANGE, batch=3)
    size = w.add_node(GetImageSize, image=base).output
    width = calc(w, "a * 2 + 8", a=size["width"])["INT"]
    height = calc(w, "a * 3", a=size["height"])["INT"]
    save(w, solid(w, width, height, CYAN, batch=size["batch_size"]), "canvas")
    save(w, scale(w, base, "bilinear", width, 0), "proportional")

    def check(out):
        canvases = out.pictures("canvas")
        assert len(canvases) == 3
        assert all(canvas.size == (88, 72) for canvas in canvases)
        proportional = out.pictures("proportional")
        assert len(proportional) == 3
        assert proportional[0].size == (88, 53)

    return check


ASPECT_RATIOS = {
    "1:1 (Square)": (1, 1),
    "16:9 (Widescreen)": (16, 9),
    "2:3 (Portrait Photo)": (2, 3),
    "9:16 (Portrait Widescreen)": (9, 16),
    "4:3 (Standard)": (4, 3),
}
RESOLUTIONS = (
    ("1:1 (Square)", 0.1, 8),
    ("16:9 (Widescreen)", 0.25, 16),
    ("2:3 (Portrait Photo)", 0.2, 32),
    ("9:16 (Portrait Widescreen)", 0.15, 8),
    ("4:3 (Standard)", 0.3, 64),
)


def resolution(ratio, megapixels, multiple):
    width_ratio, height_ratio = ASPECT_RATIOS[ratio]
    factor = math.sqrt(megapixels * 1024 * 1024 / (width_ratio * height_ratio))
    return (
        round(width_ratio * factor / multiple) * multiple,
        round(height_ratio * factor / multiple) * multiple,
    )


@case("ResolutionSelector em cinco proporções, megapixels e múltiplos: cada EmptyImage tem "
      "exatamente o tamanho que a fórmula do ComfyUI produz.")
def resolution_selector_aspect_ratios(w):
    for index, (ratio, megapixels, multiple) in enumerate(RESOLUTIONS):
        size = w.add_node(
            ResolutionSelector, aspect_ratio=ratio, megapixels=megapixels, multiple=multiple
        ).output
        save(w, solid(w, size["width"], size["height"], hues(5)[index]), f"ratio_{index}")

    def check(out):
        for index, (ratio, megapixels, multiple) in enumerate(RESOLUTIONS):
            picture = out.picture(f"ratio_{index}")
            assert picture.size == resolution(ratio, megapixels, multiple), ratio
            assert picture.width % multiple == 0 and picture.height % multiple == 0

    return check


@case("ColorToRGBInt: o inteiro pinta um EmptyImage, o alfa de #RRGGBBAA vira largura via "
      "matemática e o hex colore o texto do TextOverlay.")
def color_to_rgb_int_paints(w):
    spring = w.add_node(ColorToRGBInt, color="#00ff7f").output
    save(w, solid(w, 16, 16, spring["rgb_int"]), "painted")
    steel = w.add_node(ColorToRGBInt, color="#336699cc").output
    save(w, solid(w, calc(w, "round(a * 100)", a=steel["alpha"])["INT"], 8, WHITE), "alpha_width")
    text = w.add_node(
        TextOverlay,
        images=solid(w, 128, 64, WHITE),
        text="Cor",
        font_size=40.0,
        color=steel["hex"],
        position="top",
        align="left",
        outline=False,
    )
    save(w, text.output["images"], "text")

    def check(out):
        assert_pixel(out.picture("painted"), 8, 8, (0, 255, 127), 0)
        assert out.picture("alpha_width").size == (80, 8)
        target = rgb(0x336699)
        assert any(
            all(abs(a - b) <= 12 for a, b in zip(value, target))
            for value in out.picture("text").pixels()
        ), "the text color never shows up"

    return check


@case("Todos os primitivos (Int, Float, Boolean, String, StringMultiline, BoundingBox, "
      "Seed) alimentam um mesmo fluxo de imagem com tamanho calculado.")
def primitives_feed_everything(w):
    width = w.add_node(PrimitiveInt, value=40).output["INT"]
    factor = w.add_node(PrimitiveFloat, value=1.5).output["FLOAT"]
    use_blue = w.add_node(PrimitiveBoolean, value=True).output["BOOLEAN"]
    word = w.add_node(PrimitiveString, value="comfy").output["STRING"]
    lines = w.add_node(PrimitiveStringMultiline, value="a\nb\nc").output["STRING"]
    height = calc(w, "a * b", a=length(w, word), b=length(w, lines))["INT"]
    color = w.add_node(
        ComfySwitchNode,
        switch=use_blue,
        on_true=w.add_node(PrimitiveInt, value=BLUE).output["INT"],
        on_false=w.add_node(PrimitiveInt, value=RED).output["INT"],
    ).output["output"]
    base = solid(w, width, height, color)
    bigger = scale_by(w, base, "nearest-exact", factor)
    cropped = crop(w, bigger, 10, 5, 20, 10)
    seed = w.add_node(SeedNode, seed=1234).output["seed"]
    noisy = w.add_node(ImageAddNoise, image=cropped, seed=seed, strength=0.0)
    save(w, base, "base")
    save(w, bigger, "bigger")
    save(w, noisy.output["IMAGE"], "cropped")

    def check(out):
        assert out.picture("base").size == (40, 25)
        assert out.picture("bigger").size == (60, 38)
        cropped = out.picture("cropped")
        assert cropped.size == (20, 10)
        assert_pixel(cropped, 10, 5, BLUE)

    return check


@case("TextOverlay no topo/esquerda e embaixo/direita com e sem contorno: o texto aparece "
      "só na faixa esperada e o resto da imagem fica intacto.")
def text_overlay_variants(w):
    def overlay(background, position, align, outline):
        return w.add_node(
            TextOverlay,
            images=background,
            text="ComfyUI",
            font_size=20.0,
            color="#ffffff",
            position=position,
            align=align,
            outline=outline,
        ).output["images"]

    save(w, overlay(solid(w, 256, 128, BLACK), "top", "left", False), "top_left")
    save(w, overlay(solid(w, 256, 128, GRAY), "bottom", "right", True), "bottom_right")

    def check(out):
        top = out.picture("top_left")
        assert top.size == (256, 128)
        assert any(value[0] > 200 for value in top.pixels((0, 0, 128, 64)))
        assert set(top.pixels((0, 100, 256, 128))) == {rgb(BLACK)}
        bottom = out.picture("bottom_right")
        assert any(value[0] > 200 for value in bottom.pixels((128, 64, 256, 128)))
        assert set(bottom.pixels((0, 0, 128, 40))) == {rgb(GRAY)}

    return check


@case("StringLength vira batch_size do EmptyImage e amount do RepeatImageBatch: o número de "
      "arquivos devolvidos prova a conta.")
def string_length_drives_batch(w):
    frames = solid(w, 16, 16, YELLOW, batch=length(w, "abc"))
    repeated = w.add_node(RepeatImageBatch, image=frames, amount=length(w, "ab"))
    save(w, frames, "frames")
    save(w, repeated.output["IMAGE"], "repeated")

    def check(out):
        assert len(out.of("frames")) == 3
        assert len(out.of("repeated")) == 6

    return check


@case("O filename_prefix do SaveImage vem de StringConcatenate + CaseConverter + "
      "StringReplace calculados no servidor: o nome do arquivo prova o resultado.")
def string_ops_build_filename_prefix(w):
    lower = w.add_node(CaseConverter, string="GALERIA", mode="lowercase").output["STRING"]
    snake = w.add_node(StringReplace, string="caso 01", find=" ", replace="_").output["STRING"]
    prefix = w.add_node(StringConcatenate, string_a=lower, string_b=snake, delimiter="-")
    save(w, solid(w, 16, 16, ORANGE), prefix.output["STRING"])

    def check(out):
        assert len(out.of("galeria-caso_01")) == 1, out.names()
        assert_pixel(out.picture("galeria-caso_01"), 8, 8, ORANGE)

    return check


# ---------------------------------------------------------------- animation and video


@case("Seis cores viram um lote e um PNG animado (SaveAnimatedPNG): o arquivo tem 6 quadros "
      "no acTL e o primeiro quadro é vermelho.")
def animated_png_frames(w):
    w.add_node(
        SaveAnimatedPNG, images=batch_of(w, SIX), filename_prefix="loop", fps=3.0, compress_level=6
    )

    def check(out):
        data = out.one("loop")
        assert png_size(data) == (64, 64)
        assert apng_frames(data) == 6
        assert_pixel(decode_png(data), 10, 10, RED, 0)

    return check


@case("O mesmo lote vira WEBP animado sem perdas (SaveAnimatedWEBP): canvas 64x64 e um "
      "quadro ANMF por cor.")
def animated_webp_frames(w):
    w.add_node(
        SaveAnimatedWEBP,
        images=batch_of(w, SIX),
        method="default",
        filename_prefix="loop",
        fps=3.0,
        lossless=True,
        quality=100,
    )

    def check(out):
        size, frames = webp_info(out.one("loop"))
        assert size == (64, 64)
        assert frames == 6

    return check


@case("SaveWEBM com VP9 e com AV1 a partir de 8 quadros 128x128: dois WebM com os codecs "
      "certos no cabeçalho Matroska.")
def webm_vp9_and_av1(w):
    frames = batch_of(w, hues(8), 128)
    for codec in ("vp9", "av1"):
        w.add_node(SaveWEBM, images=frames, codec=codec, filename_prefix=codec, fps=8.0, crf=32.0)

    def check(out):
        assert out.name("vp9").endswith(".webm")
        assert_matroska(out.one("vp9"), b"V_VP9")
        assert out.name("av1").endswith(".webm")
        assert_matroska(out.one("av1"), b"V_AV1")

    return check


VIDEO_TARGETS = {
    "mp4_h264": ("mp4", "h264", ".mp4"),
    "mp4_av1": ("mp4", "av1", ".mp4"),
    "mkv_h264": ("mkv", "h264", ".mkv"),
    "webm_av1": ("webm", "av1", ".webm"),
    "auto": ("auto", "auto", ".mp4"),
}


@case("SaveVideo em todas as combinações de contêiner e codec (mp4/mkv/webm, h264/av1, "
      "auto): extensão, contêiner e codec conferidos em cada arquivo.")
def save_video_containers(w):
    video = w.add_node(CreateVideo, images=batch_of(w, hues(8), 128), fps=8.0).output["VIDEO"]
    for label, (container, codec, _) in VIDEO_TARGETS.items():
        w.add_node(SaveVideo, video=video, filename_prefix=label, format=container, format_codec=codec)

    def check(out):
        for label, (_, _, extension) in VIDEO_TARGETS.items():
            assert out.name(label).endswith(extension), out.names()
        assert_mp4(out.one("mp4_h264"), b"avc1")
        assert_mp4(out.one("mp4_av1"), b"av01")
        assert_matroska(out.one("mkv_h264"), b"V_MPEG4/ISO/AVC")
        assert_matroska(out.one("webm_av1"), b"V_AV1")
        assert_mp4(out.one("auto"))

    return check


@case("Vídeo -> GetVideoComponents -> CreateVideo de novo (fps, bit_depth e color_space "
      "ligados como combos) -> componentes: os seis quadros voltam idênticos e o fps vira "
      "largura via matemática.")
def video_round_trip_frames(w):
    video = w.add_node(CreateVideo, images=batch_of(w, SIX), fps=6.0).output["VIDEO"]
    parts = w.add_node(GetVideoComponents, video=video).output
    rebuilt = w.add_node(
        CreateVideo,
        images=parts["images"],
        fps=parts["fps"],
        bit_depth=parts["bit_depth"],
        color_space=parts["color_space"],
    ).output["VIDEO"]
    again = w.add_node(GetVideoComponents, video=rebuilt).output
    save(w, again["images"], "frame")
    save(w, solid(w, calc(w, "a * 10", a=again["fps"])["INT"], 8, WHITE), "fps_width")

    def check(out):
        frames = out.pictures("frame")
        assert [frame.pixel(20, 20) for frame in frames] == [rgb(c) for c in SIX]
        assert out.picture("fps_width").size == (60, 8)

    return check


def h264_profile(data: bytes) -> int:
    # avcC starts with configurationVersion, then AVCProfileIndication.
    position = data.index(b"avcC")
    return data[position + 5]


@case("VideoTemporalCrop e VideoFrameSample (uniform e head) sobre 16 quadros de cores "
      "diferentes: contagem de quadros e cores amostradas conferidas.")
def temporal_crop_and_frame_sample(w):
    colors = hues(16)
    video = w.add_node(CreateVideo, images=batch_of(w, colors), fps=8.0).output["VIDEO"]
    cropped = w.add_node(VideoTemporalCrop, video=video, start_frame=4, length=6)
    uniform = w.add_node(VideoFrameSample, video=video, num_frames=4, strategy="uniform")
    head = w.add_node(VideoFrameSample, video=video, num_frames=3, strategy="head")
    for label, node in (("cropped", cropped), ("uniform", uniform), ("head", head)):
        frames = w.add_node(GetVideoComponents, video=node.output["video"]).output["images"]
        save(w, frames, label)

    def check(out):
        assert 5 <= len(out.of("cropped")) <= 7, out.names()
        assert 2 <= len(out.of("head")) <= 4, out.names()
        uniform = out.pictures("uniform")
        assert len(uniform) == 4
        for frame, index in zip(uniform, (0, 5, 10, 15)):
            assert_pixel(frame, 32, 32, colors[index], 48)

    return check


@case("O nó 'Video Slice' (class_type com espaço) corta 1 s a partir de 0,5 s de um vídeo "
      "de 2 s; a fatia é salva em MP4 e decomposta em quadros.")
def video_slice_node(w):
    video = w.add_node(CreateVideo, images=batch_of(w, hues(16)), fps=8.0).output["VIDEO"]
    sliced = w.add_node(VideoSlice, video=video, start_time=0.5, duration=1.0).output["VIDEO"]
    w.add_node(SaveVideo, video=sliced, filename_prefix="slice", format="mp4", format_codec="h264")
    save(w, w.add_node(GetVideoComponents, video=sliced).output["images"], "frame")

    def check(out):
        assert_mp4(out.one("slice"), b"avc1")
        assert 7 <= len(out.of("frame")) <= 9, out.names()

    return check


@case("Áudio silencioso com AudioAdjustVolume e TrimAudioDuration entra no CreateVideo; o "
      "MP4 sai com faixa AAC e o WebM com Opus.")
def video_with_audio_tracks(w):
    audio = w.add_node(EmptyAudio, duration=1.0, sample_rate=44100, channels=2).output["AUDIO"]
    audio = w.add_node(AudioAdjustVolume, audio=audio, volume=-6).output["AUDIO"]
    audio = w.add_node(TrimAudioDuration, audio=audio, start_index=0.0, duration=1.0).output["AUDIO"]
    video = w.add_node(CreateVideo, images=batch_of(w, hues(8), 128), fps=8.0, audio=audio)
    for label, container, codec in (("with_aac", "mp4", "h264"), ("with_opus", "webm", "av1")):
        w.add_node(
            SaveVideo,
            video=video.output["VIDEO"],
            filename_prefix=label,
            format=container,
            format_codec=codec,
        )

    def check(out):
        assert_mp4(out.one("with_aac"), b"avc1", b"mp4a")
        assert_matroska(out.one("with_opus"), b"V_AV1", b"A_OPUS")

    return check


@case("Cadeia de áudio com SplitAudioChannels, JoinAudioChannels, AudioConcat, AudioMerge, "
      "AudioEqualizer3Band e TrimAudioDuration, gravada num MKV com H.264 + AAC.")
def audio_channel_ops_into_video(w):
    first = w.add_node(EmptyAudio, duration=2.0, sample_rate=48000, channels=2).output["AUDIO"]
    channels = w.add_node(SplitAudioChannels, audio=first).output
    joined = w.add_node(JoinAudioChannels, audio_left=channels["left"], audio_right=channels["right"])
    second = w.add_node(EmptyAudio, duration=1.0, sample_rate=48000, channels=2).output["AUDIO"]
    longer = w.add_node(AudioConcat, audio1=joined.output["audio"], audio2=second, direction="after")
    merged = w.add_node(
        AudioMerge, audio1=longer.output["AUDIO"], audio2=longer.output["AUDIO"], merge_method="mean"
    )
    shaped = w.add_node(
        AudioEqualizer3Band, audio=merged.output["AUDIO"], low_gain_dB=3.0, high_gain_dB=-3.0
    )
    trimmed = w.add_node(TrimAudioDuration, audio=shaped.output["AUDIO"], start_index=0.5, duration=1.5)
    video = w.add_node(
        CreateVideo, images=batch_of(w, hues(12), 128), fps=8.0, audio=trimmed.output["AUDIO"]
    )
    w.add_node(
        SaveVideo, video=video.output["VIDEO"], filename_prefix="mixed", format="mkv", format_codec="h264"
    )

    def check(out):
        assert out.name("mixed").endswith(".mkv")
        assert_matroska(out.one("mixed"), b"V_MPEG4/ISO/AVC", b"A_AAC")

    return check


@case("Quatro rotações dos quadrantes viram os quadros de um vídeo; GetVideoComponents + "
      "ImageFromBatch recuperam o 2º e o último quadro exatos, e um WEBP animado.")
def rotation_frames_through_video(w):
    image = quadrants(w, 64)
    rotated = [image] + [rotate(w, image, rotation) for rotation in ROTATIONS]
    sheet = rotated[0]
    for frame in rotated[1:]:
        sheet = stitch(w, sheet, frame)
    frames = w.add_node(RebatchImages, images=tiles_of(w, sheet, 64), batch_size=4).output["IMAGE"]
    video = w.add_node(CreateVideo, images=frames, fps=4.0).output["VIDEO"]
    parts = w.add_node(GetVideoComponents, video=video).output["images"]
    second = w.add_node(ImageFromBatch, image=parts, batch_index=1, length=1)
    last = w.add_node(ImageFromBatch, image=parts, batch_index=-1, length=1)
    save(w, second.output["IMAGE"], "second")
    save(w, last.output["IMAGE"], "last")
    w.add_node(
        SaveAnimatedWEBP, images=parts, method="fastest", filename_prefix="spin", fps=4.0,
        lossless=True, quality=90,
    )

    def check(out):
        for label, rotation in (("second", "90 degrees"), ("last", "270 degrees")):
            picture = out.picture(label)
            for x, y in quarter_points(64, 64):
                assert_pixel(picture, x, y, quadrant_color(*rotated_source(rotation, x, y, 64, 64), 64, 64), 0)
        assert webp_info(out.one("spin")) == ((64, 64), 4)

    return check


@case("CreateVideo com bit_depth 10, decomposto e recriado com os combos bit_depth e "
      "color_space do GetVideoComponents: o MP4 sai em H.264 High 10, o de 8 bits não.")
def video_bit_depth_and_color_space(w):
    frames = batch_of(w, hues(6), 128)
    deep = w.add_node(
        CreateVideo, images=frames, fps=10.0, bit_depth=10, color_space="sRGB"
    ).output["VIDEO"]
    parts = w.add_node(GetVideoComponents, video=deep).output
    rebuilt = w.add_node(
        CreateVideo,
        images=parts["images"],
        fps=parts["fps"],
        bit_depth=parts["bit_depth"],
        color_space=parts["color_space"],
    ).output["VIDEO"]
    shallow = w.add_node(CreateVideo, images=frames, fps=10.0, bit_depth=8).output["VIDEO"]
    for label, video in (("deep", deep), ("rebuilt", rebuilt), ("shallow", shallow)):
        w.add_node(SaveVideo, video=video, filename_prefix=label, format="mp4", format_codec="h264")

    def check(out):
        for label in ("deep", "rebuilt", "shallow"):
            assert_mp4(out.one(label), b"avc1")
        assert h264_profile(out.one("deep")) == 110
        assert h264_profile(out.one("rebuilt")) == 110
        assert h264_profile(out.one("shallow")) != 110

    return check


# ---------------------------------------------------------------- library features

API_JSON = {
    "canvas": {
        "class_type": "EmptyImage",
        "inputs": {"width": 48, "height": 32, "batch_size": 1, "color": GREEN},
    },
    "turn": {
        "class_type": "ImageRotate",
        "inputs": {"image": ["canvas", 0], "rotation": "90 degrees"},
    },
    "frame": {
        "class_type": "ImagePadForOutpaint",
        "inputs": {
            "image": ["turn", 0], "left": 8, "top": 8, "right": 8, "bottom": 8, "feathering": 0,
        },
    },
    "save": {
        "class_type": "SaveImage",
        "inputs": {"images": ["frame", 0], "filename_prefix": "loaded"},
    },
}


@case("JSON no formato da API com ids em texto carregado com load_workflow, alterado com "
      "update_node e estendido com add_node usando get_node_output.")
def loaded_api_json_runs(w):
    w.load_workflow(json.loads(json.dumps(API_JSON)))
    w.update_node("canvas", color=ORANGE)
    inverted = w.add_node(ImageInvert, image=w.get_node_output("frame")["IMAGE"])
    save(w, inverted.output["IMAGE"], "inverted")

    def check(out):
        loaded = out.picture("loaded")
        assert loaded.size == (48, 64)
        assert_pixel(loaded, 24, 32, ORANGE)
        assert_pixel(loaded, 2, 2, gray(0.5))
        inverted = out.picture("inverted")
        assert_pixel(inverted, 24, 32, (0, 127, 255))
        assert_pixel(inverted, 2, 2, gray(0.5))

    return check


@case("update_node troca o link de entrada e a cor depois de montado, e uma atribuição "
      "direta (node.width = 56) passa pelas mesmas validações.")
def update_node_rewires_inputs(w):
    red = w.add_node(EmptyImage, width=16, height=16, color=RED)
    blue = w.add_node(EmptyImage, width=40, height=24, color=BLUE)
    inverted = w.add_node(ImageInvert, image=red.output["IMAGE"])
    save(w, inverted.output["IMAGE"], "inverted")
    w.update_node(inverted.node_id, image=blue.output["IMAGE"])
    w.update_node(blue.node_id, color=GREEN)
    blue.width = 56

    def check(out):
        inverted = out.picture("inverted")
        assert inverted.size == (56, 24)
        assert_pixel(inverted, 10, 10, MAGENTA)

    return check


@case("remove_node apaga um ramo inteiro (folha primeiro); remover um nó com dependentes "
      "é recusado; só o ramo que ficou volta do ComfyUI.")
def remove_node_prunes_branch(w):
    base = solid(w, 24, 24, CYAN)
    save(w, base, "kept")
    flipped = w.add_node(ImageFlip, image=base, flip_method=FLIPS[0])
    dropped = w.add_node(SaveImage, images=flipped.output["IMAGE"], filename_prefix="dropped")
    with pytest.raises(NodeHasDependentsError):
        w.remove_node(flipped.node_id)
    w.remove_node(dropped.node_id)
    w.remove_node(flipped.node_id)

    def check(out):
        assert out.of("dropped") == []
        assert_pixel(out.picture("kept"), 12, 12, CYAN)

    return check


@case("Pipeline Framed (GetImageSize + matemática + EmptyImage + composite) aplicada três "
      "vezes em cadeia: as molduras ficam aninhadas na ordem certa.")
def nested_framed_pipelines(w):
    image = solid(w, 32, 32, ORANGE)
    for border, color in ((4, WHITE), (8, BLACK), (2, RED)):
        image = w.add_pipeline(Framed, image=image, border=border, color=color).output["IMAGE"]
    save(w, image, "framed")

    def check(out):
        framed = out.picture("framed")
        assert framed.size == (60, 60)
        for x, y, color in ((0, 0, RED), (1, 30, RED), (5, 5, BLACK), (12, 12, WHITE), (30, 30, ORANGE)):
            assert_pixel(framed, x, y, color)

    return check


@case("Pipeline Spotlight com duas saídas (IMAGE e MASK): GrowMask, MaskComposite com "
      "posição calculada e composite sobre fundo preto.")
def pipeline_with_two_outputs(w):
    spotlight = w.add_pipeline(Spotlight, image=solid(w, 64, 64, YELLOW), size=16, grow=4)
    save(w, spotlight.output["IMAGE"], "lit")
    save(w, mask_image(w, spotlight.output["MASK"]), "mask")

    def check(out):
        lit = out.picture("lit")
        for x, y, color in ((32, 32, YELLOW), (21, 21, YELLOW), (18, 18, BLACK), (45, 45, BLACK)):
            assert_pixel(lit, x, y, color)
        mask = out.picture("mask")
        assert_pixel(mask, 20, 20, WHITE)
        assert_pixel(mask, 19, 19, BLACK)

    return check


@case("Workflow montado, exportado em JSON, recarregado em outro Workflow, alterado com "
      "update_node e estendido antes de rodar.")
def reload_then_mutate(w):
    source = Workflow()
    image = quadrants(source, 32)
    save(source, image, "first")
    canvas_id = next(
        node_id
        for node_id, node in source.nodes.items()
        if isinstance(node, EmptyImage) and node.width == 32
    )
    w.load_workflow(sent(source))
    w.update_node(canvas_id, color=MAGENTA)
    save(w, rotate(w, w.get_node_output(image[0])["IMAGE"], "180 degrees"), "turned")

    def check(out):
        assert_quadrants(out.picture("first"), (MAGENTA, GREEN, BLUE, WHITE))
        assert_quadrants(out.picture("turned"), (WHITE, BLUE, GREEN, MAGENTA))

    return check


@case("Um único job com 24 SaveImage de cores diferentes (roda de matizes): os 24 arquivos "
      "voltam, cada um com a sua cor exata.")
def two_dozen_outputs_in_one_job(w):
    colors = hues(24)
    for index, color in enumerate(colors):
        save(w, solid(w, 16, 16, color), f"hue_{index:02}")

    def check(out):
        for index, color in enumerate(colors):
            assert_pixel(out.picture(f"hue_{index:02}"), 8, 8, color, 0)

    return check


@case("Cadeia de 40 nós alternando ImageInvert e ImageFlip: com número par de cada, a saída "
      "tem de ser idêntica à entrada.")
def forty_node_chain_returns_to_start(w):
    image = quadrants(w, 32, (ORANGE, CYAN, MAGENTA, GRAY))
    save(w, image, "start")
    for step in range(40):
        if step % 2:
            image = flip(w, image, FLIPS[1])
        else:
            image = w.add_node(ImageInvert, image=image).output["IMAGE"]
    save(w, image, "end")

    def check(out):
        assert out.picture("end").rows == out.picture("start").rows

    return check


# ---------------------------------------------------------------- failures


@rejected("Largura 0 num EmptyImage: a biblioteca recusa e o ComfyUI também "
          "('smaller than min').", library_says="width", comfy_says="smaller than min of 1")
def comfy_rejects_value_below_minimum():
    return {
        "1": {"class_type": "EmptyImage", "inputs": {"width": 0, "height": 8, "batch_size": 1, "color": 0}},
        "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0], "filename_prefix": "never"}},
    }


@rejected("blend_mode inexistente no ImageBlend: recusado pela biblioteca e pelo ComfyUI "
          "('Value not in list').", library_says="blend_mode", comfy_says="Value not in list")
def comfy_rejects_unknown_combo_value():
    return {
        "1": {"class_type": "EmptyImage", "inputs": {"width": 8, "height": 8, "batch_size": 1, "color": 0}},
        "2": {
            "class_type": "ImageBlend",
            "inputs": {"image1": ["1", 0], "image2": ["1", 0], "blend_factor": 0.5, "blend_mode": "glow"},
        },
        "3": {"class_type": "SaveImage", "inputs": {"images": ["2", 0], "filename_prefix": "never"}},
    }


@rejected("ImageInvert sem a entrada obrigatória image: recusado pelos dois lados "
          "('Required input is missing').", library_says="image", comfy_says="Required input is missing")
def comfy_rejects_missing_required_input():
    return {
        "1": {"class_type": "ImageInvert", "inputs": {}},
        "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0], "filename_prefix": "never"}},
    }


@rejected("class_type que não existe: a biblioteca e o ComfyUI apontam o nó desconhecido.",
          library_says="NotARealNode", comfy_says="NotARealNode")
def comfy_rejects_unknown_node_type():
    return {
        "1": {"class_type": "NotARealNode", "inputs": {}},
        "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0], "filename_prefix": "never"}},
    }


@rejected("Link para um nó que não está no workflow: recusado pela biblioteca e o ComfyUI "
          "falha ao validar.", library_says="ghost", comfy_says="Exception when validating node")
def comfy_rejects_link_to_missing_node():
    return {
        "1": {"class_type": "SaveImage", "inputs": {"images": ["ghost", 0], "filename_prefix": "never"}},
    }


@rejected("Imagem ligada numa entrada MASK: a biblioteca e o ComfyUI recusam o tipo errado.",
          library_says="which is IMAGE, but the input takes MASK",
          comfy_says="received_type(IMAGE) mismatch input_type(MASK)")
def comfy_rejects_image_into_mask():
    return {
        "1": {"class_type": "EmptyImage", "inputs": {"width": 8, "height": 8, "batch_size": 1, "color": 0}},
        "2": {"class_type": "MaskToImage", "inputs": {"mask": ["1", 0]}},
        "3": {"class_type": "SaveImage", "inputs": {"images": ["2", 0], "filename_prefix": "never"}},
    }


@case("Divisão por zero no ComfyMathExpression: a biblioteca aceita o grafo, mas o ComfyUI "
      "falha na execução com a mensagem do Python.", fails_with="division by zero")
def division_by_zero_fails_at_runtime(w):
    save(w, solid(w, calc(w, "a / b", a=64, b=0)["INT"], 8, RED), "never")


@case("ComfyNumberConvert recebe o texto 'doze': falha em tempo de execução com a mensagem "
      "do ComfyUI.", fails_with="Cannot convert string to number")
def number_convert_rejects_text(w):
    text = w.add_node(PrimitiveString, value="doze").output["STRING"]
    number = w.add_node(ComfyNumberConvert, value=text).output["INT"]
    save(w, solid(w, number, 8, RED), "never")


@case("Placeholder '__list__' nos loaders de checkpoint, LoRA, VAE, upscale, modelo de "
      "difusão e text encoders (DualCLIPLoader): o worker responde com o inventário de "
      "cada pasta (registrado no result.json).",
      fails_with=(CHECKPOINT, "diffusion_models", "text_encoders"))
def worker_lists_models_on_placeholder(w):
    model = w.add_node(CheckpointLoaderSimple, ckpt_name="__list__")
    lora = w.add_node(
        LoraLoader,
        model=model.output["MODEL"],
        clip=model.output["CLIP"],
        lora_name="__list__",
        strength_model=1.0,
        strength_clip=1.0,
    )
    vae = w.add_node(VAELoader, vae_name="__list__")
    upscaler = w.add_node(UpscaleModelLoader, model_name="__list__")
    unet = w.add_node(UNETLoader, unet_name="__list__", weight_dtype="default")
    text_encoders = w.add_node(
        DualCLIPLoader, clip_name1="__list__", clip_name2="__list__", type="sdxl"
    )
    for output in (
        lora.output["MODEL"],
        vae.output["VAE"],
        upscaler.output["UPSCALE_MODEL"],
        unet.output["MODEL"],
        text_encoders.output["CLIP"],
    ):
        w.add_node(PreviewAny, source=output)


@case("Checkpoint inexistente: o worker recusa antes de rodar, nomeia o arquivo que faltou "
      "e lista os checkpoints disponíveis.",
      fails_with=("nao_existe.safetensors", CHECKPOINT))
def missing_checkpoint_is_reported(w):
    loader = w.add_node(CheckpointLoaderSimple, ckpt_name="nao_existe.safetensors")
    positive = prompt(w, loader.output["CLIP"], "a cat")
    negative = prompt(w, loader.output["CLIP"], NEGATIVE)
    latent = ksample(w, loader.output["MODEL"], positive, negative, empty_latent(w, 512, 512))
    save(w, decode(w, latent, loader.output["VAE"]), "never")


# ---------------------------------------------------------------- SDXL


@case("SDXL com quatro combinações de sampler/scheduler (euler, dpmpp_2m, dpmpp_2m_sde, "
      "uni_pc) e uma folha de contato 2x2 montada com ImageScale + ImageStitch.", gpu=True)
def sdxl_sampler_and_scheduler_sheet(w):
    model, clip, vae = sdxl(w)
    positive = prompt(w, clip, "a lighthouse on a rocky coast at sunset, oil painting")
    negative = prompt(w, clip, NEGATIVE)
    latent = empty_latent(w, 768, 768)
    combos = (("euler", "normal"), ("dpmpp_2m", "karras"), ("dpmpp_2m_sde", "exponential"),
              ("uni_pc", "sgm_uniform"))
    small = []
    for sampler, scheduler in combos:
        image = decode(w, ksample(w, model, positive, negative, latent, sampler=sampler,
                                  scheduler=scheduler), vae)
        save(w, image, f"{sampler}_{scheduler}")
        small.append(scale(w, image, "area", 384, 384))
    top, bottom = stitch(w, small[0], small[1]), stitch(w, small[2], small[3])
    save(w, stitch(w, top, bottom, "down"), "sheet")

    def check(out):
        images = [out.one(f"{sampler}_{scheduler}") for sampler, scheduler in combos]
        for image in images:
            assert_generated(image, (768, 768))
        assert len(set(images)) == 4
        assert png_size(out.one("sheet")) == (768, 768)

    return check


@case("Semente reprodutível no SDXL: dois KSampler com a mesma semente dão o mesmo "
      "resultado e uma semente diferente muda a imagem.", gpu=True)
def sdxl_seed_reproducibility(w):
    model, clip, vae = sdxl(w)
    positive = prompt(w, clip, "a bowl of ramen on a wooden table, food photography")
    negative = prompt(w, clip, NEGATIVE)
    latent = empty_latent(w, 640, 640)
    for label, seed in (("seed_a", 101), ("seed_a_again", 101), ("seed_b", 202)):
        image = decode(w, ksample(w, model, positive, negative, latent, seed=seed, steps=12), vae)
        thumbnail(w, image, label, 32)
        if label == "seed_a":
            save(w, image, "full")

    def check(out):
        first, again, other = (out.picture(label) for label in ("seed_a", "seed_a_again", "seed_b"))
        assert mean_difference(first, again) < 1.5
        assert mean_difference(first, other) > 8
        assert_generated(out.one("full"), (640, 640))

    return check


@case("O prompt chega ao modelo: 'parede vermelha' sai avermelhada e 'parede azul' sai "
      "azulada, medido na cor média de miniaturas.", gpu=True)
def sdxl_prompt_reaches_the_model(w):
    model, clip, vae = sdxl(w)
    negative = prompt(w, clip, NEGATIVE)
    latent = empty_latent(w, 768, 768)
    for label, color in (("red_wall", "red"), ("blue_wall", "blue")):
        text = f"a plain wall painted bright {color}, solid {color} color, {color}, minimalism"
        image = decode(w, ksample(w, model, prompt(w, clip, text), negative, latent), vae)
        save(w, image, label)
        thumbnail(w, image, f"{label}_thumb", 8)

    def check(out):
        r, g, b = out.picture("red_wall_thumb").mean()
        assert r > g + 40 and r > b + 40, (r, g, b)
        r, g, b = out.picture("blue_wall_thumb").mean()
        assert b > r + 40, (r, g, b)

    return check


@case("Lote de 4 latentes no SDXL: LatentFromBatch pega o 2º e o 3º, RepeatLatentBatch "
      "duplica o 4º; as miniaturas batem com as do lote original.", gpu=True)
def sdxl_batch_and_latent_picks(w):
    model, clip, vae = sdxl(w)
    positive = prompt(w, clip, "a hot air balloon over green hills, watercolor")
    negative = prompt(w, clip, NEGATIVE)
    batch = ksample(w, model, positive, negative, empty_latent(w, 576, 576, batch=4), steps=12)
    images = decode(w, batch, vae)
    save(w, images, "batch")
    thumbnail(w, images, "batch_thumb", 24)
    picked = w.add_node(LatentFromBatch, samples=batch, batch_index=1, length=2).output["LATENT"]
    thumbnail(w, decode(w, picked, vae), "picked_thumb", 24)
    fourth = w.add_node(LatentFromBatch, samples=batch, batch_index=3, length=1).output["LATENT"]
    repeated = w.add_node(RepeatLatentBatch, samples=fourth, amount=2).output["LATENT"]
    thumbnail(w, decode(w, repeated, vae), "repeated_thumb", 24)

    def check(out):
        batch = out.of("batch")
        assert len(batch) == 4 and len(set(batch)) == 4
        for image in batch:
            assert_generated(image, (576, 576))
        thumbs = out.pictures("batch_thumb")
        picked = out.pictures("picked_thumb")
        repeated = out.pictures("repeated_thumb")
        assert mean_difference(picked[0], thumbs[1]) < 2
        assert mean_difference(picked[1], thumbs[2]) < 2
        assert all(mean_difference(copy, thumbs[3]) < 2 for copy in repeated)

    return check


@case("KSamplerAdvanced em duas etapas (0-12 com ruído restante, 12-20 sem novo ruído) dá "
      "o mesmo resultado que uma passada única de 20 passos com euler.", gpu=True)
def sdxl_two_stage_advanced_matches_single_pass(w):
    model, clip, vae = sdxl(w)
    positive = prompt(w, clip, "a red fox in the snow, wildlife photography")
    negative = prompt(w, clip, NEGATIVE)
    latent = empty_latent(w, 640, 640)

    def advanced(source, add_noise, start, end, leftover):
        return w.add_node(
            KSamplerAdvanced,
            model=model,
            add_noise=add_noise,
            noise_seed=33,
            steps=20,
            cfg=6.5,
            sampler_name="euler",
            scheduler="normal",
            positive=positive,
            negative=negative,
            latent_image=source,
            start_at_step=start,
            end_at_step=end,
            return_with_leftover_noise=leftover,
        ).output["LATENT"]

    first = advanced(latent, "enable", 0, 12, "enable")
    two_stage = decode(w, advanced(first, "disable", 12, 10000, "disable"), vae)
    single = decode(w, advanced(latent, "enable", 0, 10000, "disable"), vae)
    save(w, two_stage, "two_stage")
    save(w, single, "single")
    thumbnail(w, two_stage, "two_stage_thumb", 32)
    thumbnail(w, single, "single_thumb", 32)

    def check(out):
        assert_generated(out.one("two_stage"), (640, 640))
        assert_generated(out.one("single"), (640, 640))
        difference = mean_difference(out.picture("two_stage_thumb"), out.picture("single_thumb"))
        assert difference < 3, difference

    return check


@case("SamplerCustom com KarrasScheduler dividido por SplitSigmas em duas metades (a 2ª sem "
      "ruído) equivale à passada completa; o scheduler exponencial difere.", gpu=True)
def sdxl_sampler_custom_split_sigmas(w):
    model, clip, vae = sdxl(w)
    positive = prompt(w, clip, "a glass terrarium with moss and tiny mushrooms, macro photo")
    negative = prompt(w, clip, NEGATIVE)
    latent = empty_latent(w, 640, 640)
    sampler = w.add_node(KSamplerSelect, sampler_name="euler").output["SAMPLER"]
    sigmas = w.add_node(KarrasScheduler, steps=16).output["SIGMAS"]
    halves = w.add_node(SplitSigmas, sigmas=sigmas, step=8).output

    def custom(source, schedule, add_noise):
        return w.add_node(
            SamplerCustom,
            model=model,
            positive=positive,
            negative=negative,
            sampler=sampler,
            sigmas=schedule,
            latent_image=source,
            add_noise=add_noise,
            noise_seed=11,
            cfg=6.5,
        ).output

    first = custom(latent, halves["high_sigmas"], True)
    split = decode(w, custom(first["output"], halves["low_sigmas"], False)["output"], vae)
    full = decode(w, custom(latent, sigmas, True)["output"], vae)
    exponential = w.add_node(ExponentialScheduler, steps=16).output["SIGMAS"]
    other = decode(w, custom(latent, exponential, True)["output"], vae)
    save(w, split, "split")
    save(w, decode(w, first["denoised_output"], vae), "halfway_preview")
    for label, image in (("split", split), ("full", full), ("exponential", other)):
        thumbnail(w, image, f"{label}_thumb", 32)

    def check(out):
        assert_generated(out.one("split"), (640, 640))
        assert png_size(out.one("halfway_preview")) == (640, 640)
        split, full, other = (out.picture(f"{label}_thumb") for label in ("split", "full", "exponential"))
        assert mean_difference(split, full) < 3
        assert mean_difference(full, other) > 1

    return check


@case("SamplerCustomAdvanced com três guiders (CFGGuider, DualCFGGuider, BasicGuider), "
      "samplers e schedulers diferentes: três imagens distintas.", gpu=True)
def sdxl_custom_advanced_guiders(w):
    model, clip, vae = sdxl(w)
    positive = prompt(w, clip, "an astronaut riding a horse on the moon, digital art")
    style = prompt(w, clip, "vivid neon colors, synthwave")
    negative = prompt(w, clip, NEGATIVE)
    latent = empty_latent(w, 640, 640)
    noise = w.add_node(RandomNoise, noise_seed=21).output["NOISE"]
    guiders = {
        "cfg": w.add_node(CFGGuider, model=model, positive=positive, negative=negative, cfg=6.0),
        "dual": w.add_node(
            DualCFGGuider,
            model=model,
            cond1=positive,
            cond2=style,
            negative=negative,
            style="regular",
            cfg_conds=6.0,
            cfg_cond2_negative=3.0,
        ),
        "basic": w.add_node(BasicGuider, model=model, conditioning=positive),
    }
    samplers = {
        "cfg": w.add_node(SamplerEulerAncestral, eta=1.0, s_noise=1.0).output["SAMPLER"],
        "dual": w.add_node(
            SamplerDPMPP_2M_SDE, solver_type="midpoint", noise_device="gpu"
        ).output["SAMPLER"],
        "basic": w.add_node(KSamplerSelect, sampler_name="dpmpp_2m").output["SAMPLER"],
    }
    schedulers = {"cfg": "karras", "dual": "exponential", "basic": "sgm_uniform"}
    for label, guider in guiders.items():
        sigmas = w.add_node(BasicScheduler, model=model, scheduler=schedulers[label], steps=14)
        sampled = w.add_node(
            SamplerCustomAdvanced,
            noise=noise,
            guider=guider.output["GUIDER"],
            sampler=samplers[label],
            sigmas=sigmas.output["SIGMAS"],
            latent_image=latent,
        )
        save(w, decode(w, sampled.output["output"], vae), label)

    def check(out):
        images = [out.one(label) for label in ("cfg", "dual", "basic")]
        for image in images:
            assert_generated(image, (640, 640))
        assert len(set(images)) == 3

    return check


@case("img2img no SDXL a partir de quadrantes coloridos desenhados no próprio ComfyUI "
      "(denoise 0,5): o layout de cores sobrevive à geração.", gpu=True)
def sdxl_img2img_keeps_color_layout(w):
    model, clip, vae = sdxl(w)
    colors = (RED, GREEN, BLUE, YELLOW)
    source = quadrants(w, 768, colors)
    latent = w.add_node(VAEEncode, pixels=source, vae=vae).output["LATENT"]
    positive = prompt(w, clip, "an abstract painting made of four large color blocks, thick brush strokes")
    image = decode(w, ksample(w, model, positive, prompt(w, clip, NEGATIVE), latent, denoise=0.5, steps=16), vae)
    save(w, source, "source")
    save(w, image, "painted")
    thumbnail(w, image, "painted_thumb", 8)

    def check(out):
        assert_generated(out.one("painted"), (768, 768))
        thumb = out.picture("painted_thumb")
        r, g, b = thumb.mean((0, 0, 4, 4))
        assert r > g + 30 and r > b + 30, ("top left", r, g, b)
        r, g, b = thumb.mean((4, 0, 8, 4))
        assert g > r + 30 and g > b + 30, ("top right", r, g, b)
        r, g, b = thumb.mean((0, 4, 4, 8))
        assert b > r + 30 and b > g + 30, ("bottom left", r, g, b)
        r, g, b = thumb.mean((4, 4, 8, 8))
        assert r > b + 30 and g > b + 30, ("bottom right", r, g, b)

    return check


@case("Hires fix no SDXL: base 640, LatentUpscaleBy (bislerp 1,5x) e LatentUpscale com "
      "crop central, cada um refinado com denoise parcial.", gpu=True)
def sdxl_hires_fix_upscales_latent(w):
    model, clip, vae = sdxl(w)
    positive = prompt(w, clip, "a cozy cabin in a snowy pine forest at night, warm windows")
    negative = prompt(w, clip, NEGATIVE)
    base = ksample(w, model, positive, negative, empty_latent(w, 640, 640), steps=12)
    save(w, decode(w, base, vae), "base")
    bigger = w.add_node(LatentUpscaleBy, samples=base, upscale_method="bislerp", scale_by=1.5)
    hires = ksample(w, model, positive, negative, bigger.output["LATENT"], steps=10, denoise=0.45)
    save(w, decode(w, hires, vae), "hires")
    resized = w.add_node(
        LatentUpscale, samples=base, upscale_method="nearest-exact", width=832, height=832, crop="center"
    )
    refined = ksample(w, model, positive, negative, resized.output["LATENT"], steps=10, denoise=0.4)
    save(w, decode(w, refined, vae), "resized")

    def check(out):
        assert_generated(out.one("base"), (640, 640))
        assert_generated(out.one("hires"), (960, 960))
        assert_generated(out.one("resized"), (832, 832))

    return check


def inpaint_canvas(w):
    canvas = solid(w, 768, 768, GRAY)
    mask = combine_masks(w, solid_mask(w, 0.0, 768, 768), solid_mask(w, 1.0, 256, 256), "add", 256, 256)
    return canvas, mask


@case("Inpainting com VAEEncodeForInpaint numa tela cinza com máscara central; o resultado "
      "é colado de volta com a máscara e o lado de fora fica exato.", gpu=True)
def sdxl_inpaint_keeps_outside_pixels(w):
    model, clip, vae = sdxl(w)
    canvas, mask = inpaint_canvas(w)
    latent = w.add_node(VAEEncodeForInpaint, pixels=canvas, vae=vae, mask=mask, grow_mask_by=8)
    positive = prompt(w, clip, "a shiny red apple on a gray table, studio photo")
    painted = decode(w, ksample(w, model, positive, prompt(w, clip, NEGATIVE), latent.output["LATENT"]), vae)
    final = paste(w, canvas, painted, mask=mask)
    save(w, final, "final")
    save(w, crop(w, final, 0, 0, 64, 64), "outside")
    save(w, crop(w, final, 352, 352, 64, 64), "inside")

    def check(out):
        assert_generated(out.one("final"), (768, 768))
        assert set(out.picture("outside").pixels()) == {(128, 128, 128)}
        inside = out.picture("inside")
        assert len(set(inside.pixels())) > 10, "nothing was painted inside the mask"

    return check


@case("Inpainting com InpaintModelConditioning e com SetLatentNoiseMask: fora da máscara a "
      "imagem continua cinza, dentro aparece conteúdo novo.", gpu=True)
def sdxl_inpaint_conditioning_and_noise_mask(w):
    model, clip, vae = sdxl(w)
    canvas, mask = inpaint_canvas(w)
    positive = prompt(w, clip, "a blue ceramic vase with flowers, studio photo")
    negative = prompt(w, clip, NEGATIVE)
    conditioned = w.add_node(
        InpaintModelConditioning,
        positive=positive,
        negative=negative,
        vae=vae,
        pixels=canvas,
        mask=mask,
        noise_mask=True,
    ).output
    first = decode(w, ksample(w, model, conditioned["positive"], conditioned["negative"], conditioned["latent"]), vae)
    encoded = w.add_node(VAEEncode, pixels=canvas, vae=vae).output["LATENT"]
    masked = w.add_node(SetLatentNoiseMask, samples=encoded, mask=mask).output["LATENT"]
    second = decode(w, ksample(w, model, positive, negative, masked, denoise=0.9), vae)
    for label, image in (("conditioned", first), ("noise_masked", second)):
        save(w, image, label)
        save(w, crop(w, image, 0, 0, 64, 64), f"{label}_outside")
        save(w, crop(w, image, 352, 352, 64, 64), f"{label}_inside")

    def check(out):
        for label in ("conditioned", "noise_masked"):
            assert_generated(out.one(label), (768, 768))
            r, g, b = out.picture(f"{label}_outside").mean()
            assert all(abs(value - 128) < 25 for value in (r, g, b)), (label, r, g, b)
            assert len(set(out.picture(f"{label}_inside").pixels())) > 10, label

    return check


@case("Outpainting no SDXL: imagem gerada é estendida 256 px para a direita com "
      "ImagePadForOutpaint e a área nova é preenchida pelo modelo.", gpu=True)
def sdxl_outpaint_extends_canvas(w):
    model, clip, vae = sdxl(w)
    positive = prompt(w, clip, "a wide beach with palm trees and turquoise water, travel photo")
    negative = prompt(w, clip, NEGATIVE)
    image = decode(w, ksample(w, model, positive, negative, empty_latent(w, 640, 640), steps=12), vae)
    padded = w.add_node(ImagePadForOutpaint, image=image, left=0, top=0, right=256, bottom=0, feathering=40)
    latent = w.add_node(
        VAEEncodeForInpaint, pixels=padded.output["IMAGE"], vae=vae, mask=padded.output["MASK"], grow_mask_by=16
    )
    extended = decode(w, ksample(w, model, positive, negative, latent.output["LATENT"], steps=14), vae)
    save(w, padded.output["IMAGE"], "padded")
    save(w, extended, "extended")

    def check(out):
        assert png_size(out.one("padded")) == (896, 640)
        assert_generated(out.one("extended"), (896, 640))

    return check


@case("Prompts regionais: tulipas vermelhas à esquerda (ConditioningSetArea / SetMask) e "
      "lago azul à direita; a metade esquerda sai mais vermelha.", gpu=True)
def sdxl_regional_prompts_left_and_right(w):
    model, clip, vae = sdxl(w)
    negative = prompt(w, clip, NEGATIVE)
    scene = prompt(w, clip, "a landscape painting, wide view")
    red = prompt(w, clip, "a field of bright red tulips, red flowers")
    blue = prompt(w, clip, "a deep blue lake under a clear blue sky")
    latent = empty_latent(w, 1024, 512)

    def combined(*conditionings):
        result = conditionings[0]
        for other in conditionings[1:]:
            result = w.add_node(ConditioningCombine, conditioning_1=result, conditioning_2=other).output["CONDITIONING"]
        return result

    left = w.add_node(ConditioningSetArea, conditioning=red, width=512, height=512, x=0, y=0, strength=1.0)
    right = w.add_node(ConditioningSetArea, conditioning=blue, width=512, height=512, x=512, y=0, strength=1.0)
    areas = combined(scene, left.output["CONDITIONING"], right.output["CONDITIONING"])
    left_mask = combine_masks(w, solid_mask(w, 0.0, 1024, 512), solid_mask(w, 1.0, 512, 512), "add")
    right_mask = w.add_node(InvertMask, mask=left_mask).output["MASK"]
    masked_left = w.add_node(ConditioningSetMask, conditioning=red, mask=left_mask, set_cond_area="mask bounds", strength=1.0)
    masked_right = w.add_node(ConditioningSetMask, conditioning=blue, mask=right_mask, set_cond_area="mask bounds", strength=1.0)
    masks = combined(scene, masked_left.output["CONDITIONING"], masked_right.output["CONDITIONING"])
    for label, positive in (("areas", areas), ("masks", masks)):
        image = decode(w, ksample(w, model, positive, negative, latent, steps=16), vae)
        save(w, image, label)
        save(w, scale(w, image, "area", 16, 8), f"{label}_thumb")

    def check(out):
        for label in ("areas", "masks"):
            assert_generated(out.one(label), (1024, 512))
            thumb = out.picture(f"{label}_thumb")
            left_r, _, left_b = thumb.mean((0, 0, 8, 8))
            right_r, _, right_b = thumb.mean((8, 0, 16, 8))
            assert (left_r - left_b) > (right_r - right_b) + 25, (label, left_r, left_b, right_r, right_b)

    return check


@case("Aritmética de condicionamento: ConditioningAverage, Concat, SetTimestepRange "
      "combinados, Multiply e ZeroOut no negativo; quatro imagens distintas.", gpu=True)
def sdxl_conditioning_arithmetic(w):
    model, clip, vae = sdxl(w)
    mountain = prompt(w, clip, "a snowy mountain peak, dramatic clouds")
    beach = prompt(w, clip, "a tropical beach with palm trees")
    negative = w.add_node(ConditioningZeroOut, conditioning=prompt(w, clip, NEGATIVE)).output["CONDITIONING"]
    early = w.add_node(ConditioningSetTimestepRange, conditioning=mountain, start=0.0, end=0.5)
    late = w.add_node(ConditioningSetTimestepRange, conditioning=beach, start=0.5, end=1.0)
    positives = {
        "average": w.add_node(ConditioningAverage, conditioning_to=mountain, conditioning_from=beach,
                              conditioning_to_strength=0.5).output["CONDITIONING"],
        "concat": w.add_node(ConditioningConcat, conditioning_to=mountain, conditioning_from=beach).output["CONDITIONING"],
        "timesteps": w.add_node(ConditioningCombine, conditioning_1=early.output["CONDITIONING"],
                                conditioning_2=late.output["CONDITIONING"]).output["CONDITIONING"],
        "multiplied": w.add_node(ConditioningMultiply, conditioning=mountain, multiplier=1.2).output["CONDITIONING"],
    }
    latent = empty_latent(w, 640, 640)
    for label, positive in positives.items():
        save(w, decode(w, ksample(w, model, positive, negative, latent, steps=12), vae), label)

    def check(out):
        images = [out.one(label) for label in positives]
        for image in images:
            assert_generated(image, (640, 640))
        assert len(set(images)) == 4

    return check


UNET_PATCHES_A = {
    "freeu": lambda w, m: w.add_node(FreeU, model=m).output["MODEL"],
    "freeu_v2": lambda w, m: w.add_node(FreeU_V2, model=m).output["MODEL"],
    "pag": lambda w, m: w.add_node(PerturbedAttentionGuidance, model=m, scale=2.0).output["MODEL"],
    # At 640 px the depth-1 attention is 40x40 tokens; 128 px tiles (16 latent) split
    # it 2x2, and swap_size 1 keeps that choice deterministic.
    "hypertile_depth_1": lambda w, m: w.add_node(
        HyperTile, model=m, tile_size=128, swap_size=1, max_depth=1
    ).output["MODEL"],
}
# ToMe and HyperTile's default max_depth=0 only touch full-resolution attention,
# which SDXL does not have, so on SDXL they leave the image as it was.
UNET_NO_OPS_A = {
    "tome": lambda w, m: w.add_node(TomePatchModel, model=m, ratio=0.3).output["MODEL"],
    "hypertile_depth_0": lambda w, m: w.add_node(HyperTile, model=m, tile_size=256).output["MODEL"],
}
UNET_PATCHES_B = {
    "deep_shrink": lambda w, m: w.add_node(
        PatchModelAddDownscale, model=m, downscale_method="bicubic", upscale_method="bicubic"
    ).output["MODEL"],
    "epsilon": lambda w, m: w.add_node(EpsilonScaling, model=m, scaling_factor=1.005).output["MODEL"],
    "rescale_cfg": lambda w, m: w.add_node(RescaleCFG, model=m, multiplier=0.7).output["MODEL"],
    "sag": lambda w, m: w.add_node(SelfAttentionGuidance, model=m, scale=0.5, blur_sigma=2.0).output["MODEL"],
    "apg": lambda w, m: w.add_node(APG, model=m, eta=1.0, norm_threshold=5.0).output["MODEL"],
    "cfg_norm": lambda w, m: w.add_node(CFGNorm, model=m, strength=1.0).output["patched_model"],
    "cfg_override": lambda w, m: w.add_node(CFGOverride, model=m, cfg=3.0).output["MODEL"],
}


def patch_case(w, text, changed, unchanged):
    model, clip, vae = sdxl(w)
    positive = prompt(w, clip, text)
    negative = prompt(w, clip, NEGATIVE)
    latent = empty_latent(w, 640, 640)
    models = {"baseline": model}
    models.update({label: build(w, model) for label, build in changed.items()})
    models.update({label: build(w, model) for label, build in unchanged.items()})
    for label, patched in models.items():
        image = decode(w, ksample(w, patched, positive, negative, latent, steps=12), vae)
        save(w, image, label)
        thumbnail(w, image, f"{label}_thumb", 32)

    def check(out):
        baseline = out.one("baseline")
        assert_generated(baseline, (640, 640))
        for label in changed:
            image = out.one(label)
            assert_generated(image, (640, 640))
            assert image != baseline, f"{label} did not change the image"
        for label in unchanged:
            difference = mean_difference(out.picture(f"{label}_thumb"), out.picture("baseline_thumb"))
            assert difference < 1.5, f"{label} changed the image by {difference}"

    return check


@case("Patches de UNet no SDXL contra a linha de base: FreeU, FreeU_V2, PAG e HyperTile "
      "(tiles de 128, max_depth 1) mudam a imagem; ToMe e HyperTile com max_depth 0 não, "
      "porque o SDXL não tem atenção na resolução máxima.", gpu=True)
def sdxl_unet_patches_part_one(w):
    return patch_case(
        w, "a vintage red car parked on a cobblestone street", UNET_PATCHES_A, UNET_NO_OPS_A
    )


@case("Mais sete patches (Deep Shrink, Epsilon Scaling, RescaleCFG, SAG, APG, CFGNorm, "
      "CFGOverride) e ModelSamplingDiscrete eps, que não pode mudar nada.", gpu=True)
def sdxl_unet_patches_part_two(w):
    same = {
        "eps": lambda w, m: w.add_node(
            ModelSamplingDiscrete, model=m, sampling="eps", zsnr=False
        ).output["MODEL"],
    }
    return patch_case(w, "a steaming cup of coffee on a rainy window sill", UNET_PATCHES_B, same)


@case("ModelMergeSimple e CLIPMergeSimple de um modelo consigo mesmo não mudam a imagem; "
      "CLIPSetLastLayer -2 também não (já é o padrão do SDXL), mas -3 muda.", gpu=True)
def sdxl_merge_identity_and_clip_skip(w):
    model, clip, vae = sdxl(w)
    text = "a portrait of an old fisherman, dramatic lighting, oil painting"
    negative = prompt(w, clip, NEGATIVE)
    latent = empty_latent(w, 640, 640)
    merged_model = w.add_node(ModelMergeSimple, model1=model, model2=model, ratio=0.5).output["MODEL"]
    merged_clip = w.add_node(CLIPMergeSimple, clip1=clip, clip2=clip, ratio=0.5).output["CLIP"]

    def last_layer(layer):
        return w.add_node(CLIPSetLastLayer, clip=clip, stop_at_clip_layer=layer).output["CLIP"]

    variants = {
        "baseline": (model, clip),
        "merged_model": (merged_model, clip),
        "merged_clip": (model, merged_clip),
        "clip_skip_2": (model, last_layer(-2)),
        "clip_skip_3": (model, last_layer(-3)),
    }
    for label, (unet, encoder) in variants.items():
        image = decode(w, ksample(w, unet, prompt(w, encoder, text), negative, latent, steps=12), vae)
        save(w, image, label)
        thumbnail(w, image, f"{label}_thumb", 32)

    def check(out):
        baseline = out.picture("baseline_thumb")
        for label in variants:
            assert_generated(out.one(label), (640, 640))
        assert mean_difference(out.picture("merged_model_thumb"), baseline) < 2
        assert mean_difference(out.picture("merged_clip_thumb"), baseline) < 2
        # SDXL already encodes with the penultimate layer of both text encoders.
        assert mean_difference(out.picture("clip_skip_2_thumb"), baseline) < 2
        assert mean_difference(out.picture("clip_skip_3_thumb"), baseline) > 3

    return check


@case("CLIPTextEncodeSDXL com condicionamento de tamanho e de recorte diferentes: três "
      "imagens distintas a partir do mesmo texto.", gpu=True)
def sdxl_size_conditioning(w):
    model, clip, vae = sdxl(w)
    negative = prompt(w, clip, NEGATIVE)
    latent = empty_latent(w, 768, 768)
    variants = {
        "full_size": {"width": 1024, "height": 1024, "crop_w": 0, "crop_h": 0},
        "cropped": {"width": 1024, "height": 1024, "crop_w": 512, "crop_h": 512},
        "low_res": {"width": 512, "height": 512, "crop_w": 0, "crop_h": 0},
    }
    for label, sizes in variants.items():
        positive = w.add_node(
            CLIPTextEncodeSDXL,
            clip=clip,
            text_g="a majestic owl perched on a branch, detailed feathers",
            text_l="owl, branch, forest",
            target_width=1024,
            target_height=1024,
            **sizes,
        ).output["CONDITIONING"]
        save(w, decode(w, ksample(w, model, positive, negative, latent, steps=12), vae), label)

    def check(out):
        images = [out.one(label) for label in variants]
        for image in images:
            assert_generated(image, (768, 768))
        assert len(set(images)) == 3

    return check


@case("Operações de latente sobre uma geração 640x512: flip, rotação, crop, composite, "
      "blend, interpolate, tonemap, sharpen e (a+a)*0,5 = a.", gpu=True)
def sdxl_latent_operations(w):
    model, clip, vae = sdxl(w)
    negative = prompt(w, clip, NEGATIVE)
    latent = empty_latent(w, 640, 512)
    base = ksample(w, model, prompt(w, clip, "a sailboat on a calm sea at dawn"), negative, latent, steps=12)
    other = ksample(w, model, prompt(w, clip, "a field of sunflowers under a blue sky"), negative, latent, steps=12)

    def latent_node(node_type, **inputs):
        return w.add_node(node_type, **inputs).output["LATENT"]

    def operation(node_type, **inputs):
        return w.add_node(node_type, **inputs).output["LATENT_OPERATION"]

    doubled = latent_node(LatentAdd, samples1=base, samples2=base)
    variants = {
        "base": base,
        "flipped": latent_node(LatentFlip, samples=base, flip_method="y-axis: horizontally"),
        "rotated": latent_node(LatentRotate, samples=base, rotation="90 degrees"),
        "cropped": latent_node(LatentCrop, samples=base, width=384, height=384, x=64, y=64),
        "doubled_half": latent_node(LatentMultiply, samples=doubled, multiplier=0.5),
        "blended": latent_node(LatentBlend, samples1=base, samples2=other, blend_factor=0.5),
        "interpolated": latent_node(LatentInterpolate, samples1=base, samples2=other, ratio=0.5),
        "tonemapped": latent_node(
            LatentApplyOperation,
            samples=base,
            operation=operation(LatentOperationTonemapReinhard, multiplier=1.0),
        ),
        "sharpened": latent_node(
            LatentApplyOperation,
            samples=base,
            operation=operation(LatentOperationSharpen, sharpen_radius=9, sigma=1.0, alpha=0.1),
        ),
        "composited": latent_node(
            LatentComposite,
            samples_to=base,
            samples_from=latent_node(LatentCrop, samples=other, width=256, height=256, x=0, y=0),
            x=64,
            y=64,
            feather=16,
        ),
        "subtracted": latent_node(LatentSubtract, samples1=base, samples2=base),
    }
    for label, samples in variants.items():
        image = decode(w, samples, vae)
        save(w, image, label)
        if label in ("base", "flipped", "doubled_half"):
            thumbnail(w, image, f"{label}_thumb", 16)

    def check(out):
        sizes = {"rotated": (512, 640), "cropped": (384, 384)}
        for label in variants:
            size = sizes.get(label, (640, 512))
            if label == "subtracted":
                assert png_size(out.one(label)) == size
            else:
                assert_generated(out.one(label), size, 30_000)
        base = out.picture("base_thumb")
        assert mean_difference(out.picture("doubled_half_thumb"), base) < 1
        assert mean_difference(out.picture("flipped_thumb"), base.mirrored()) < 25

    return check


@case("VAEDecodeTiled e VAEEncodeTiled numa geração 1024x1024: o decode em tiles bate com o "
      "normal e o encode em tiles alimenta um refinamento.", gpu=True)
def sdxl_tiled_vae_round_trip(w):
    model, clip, vae = sdxl(w)
    positive = prompt(w, clip, "an ornate gothic cathedral interior, stained glass, light rays")
    negative = prompt(w, clip, NEGATIVE)
    base = ksample(w, model, positive, negative, empty_latent(w, 1024, 1024), steps=12)
    full = decode(w, base, vae)
    tiled = w.add_node(
        VAEDecodeTiled, samples=base, vae=vae, tile_size=512, overlap=64, temporal_size=64, temporal_overlap=8
    ).output["IMAGE"]
    encoded = w.add_node(
        VAEEncodeTiled, pixels=tiled, vae=vae, tile_size=512, overlap=64, temporal_size=64, temporal_overlap=8
    ).output["LATENT"]
    refined = decode(w, ksample(w, model, positive, negative, encoded, steps=8, denoise=0.3), vae)
    save(w, tiled, "tiled")
    save(w, refined, "refined")
    thumbnail(w, full, "full_thumb", 32)
    thumbnail(w, tiled, "tiled_thumb", 32)

    def check(out):
        assert_generated(out.one("tiled"), (1024, 1024))
        assert_generated(out.one("refined"), (1024, 1024))
        assert mean_difference(out.picture("tiled_thumb"), out.picture("full_thumb")) < 4

    return check


@case("Seis gerações SDXL em lote (LatentBatchSeedBehavior random) viram um MP4 H.264, um "
      "WEBP animado e voltam como quadros via GetVideoComponents.", gpu=True)
def sdxl_frames_to_video(w):
    model, clip, vae = sdxl(w)
    positive = prompt(w, clip, "a jellyfish glowing in the deep ocean, bioluminescence")
    negative = prompt(w, clip, NEGATIVE)
    batch = w.add_node(LatentBatchSeedBehavior, samples=empty_latent(w, 512, 512, batch=6), seed_behavior="random")
    frames = decode(w, ksample(w, model, positive, negative, batch.output["LATENT"], steps=10), vae)
    video = w.add_node(CreateVideo, images=frames, fps=6.0).output["VIDEO"]
    w.add_node(SaveVideo, video=video, filename_prefix="clip", format="mp4", format_codec="h264")
    w.add_node(
        SaveAnimatedWEBP, images=frames, method="default", filename_prefix="loop", fps=6.0,
        lossless=False, quality=80,
    )
    parts = w.add_node(GetVideoComponents, video=video).output["images"]
    third = w.add_node(ImageFromBatch, image=parts, batch_index=3, length=1)
    save(w, third.output["IMAGE"], "frame_3")

    def check(out):
        assert_mp4(out.one("clip"), b"avc1")
        assert webp_info(out.one("loop")) == ((512, 512), 6)
        assert_generated(out.one("frame_3"), (512, 512))

    return check


@case("Pipeline TextToImage + ImageSharpen + TextOverlay + ImageScaleToMaxDimension + Canny "
      "costurados num pôster: tamanhos conferidos em cada etapa.", gpu=True)
def sdxl_postprocess_pipeline(w):
    image = w.add_pipeline(
        TextToImage, text="a futuristic city skyline at dusk, flying cars", width=896, height=640
    ).output["IMAGE"]
    sharp = w.add_node(ImageSharpen, image=image, sharpen_radius=1, sigma=1.0, alpha=0.3).output["IMAGE"]
    titled = w.add_node(
        TextOverlay,
        images=sharp,
        text="ComfyUI via RunPod",
        font_size=6.0,
        color="#ffcc00",
        position="bottom",
        align="right",
        outline=True,
    ).output["images"]
    small = w.add_node(ImageScaleToMaxDimension, image=titled, upscale_method="lanczos", largest_size=512)
    edges = w.add_node(Canny, image=small.output["IMAGE"], low_threshold=0.2, high_threshold=0.5)
    save(w, titled, "titled")
    save(w, small.output["IMAGE"], "small")
    save(w, stitch(w, small.output["IMAGE"], edges.output["IMAGE"], "down"), "poster")

    def check(out):
        assert_generated(out.one("titled"), (896, 640))
        assert png_size(out.one("small")) == (512, 366)
        assert png_size(out.one("poster")) == (512, 732)

    return check


# ---------------------------------------------------------------- the special one

SPECIAL_NEGATIVE = (
    "nsfw, nude, naked, swimsuit, bikini, underwear, blurry, low quality, deformed, "
    "bad anatomy, extra limbs, extra fingers, watermark, text"
)
SPECIAL_SCENES = (
    (
        "menino_correndo_na_rua",
        "a cheerful young boy running down the middle of a quiet city street, motion, "
        "sneakers, t-shirt and shorts, sunny afternoon, cinematic photo, shallow depth of "
        "field, highly detailed",
        1001,
    ),
    (
        "menina_na_lagoa",
        "a young girl wearing a yellow summer dress standing on the grassy shore of a calm "
        "lagoon, skipping a stone across the water, lush green trees, golden hour, "
        "cinematic photo, highly detailed, family friendly",
        2002,
    ),
)


def special_workflow() -> Workflow:
    w = Workflow()
    model, clip, vae = sdxl(w)
    negative = prompt(w, clip, SPECIAL_NEGATIVE)
    for label, text, seed in SPECIAL_SCENES:
        positive = prompt(w, clip, text)
        base = ksample(
            w, model, positive, negative, empty_latent(w, 1216, 832), seed=seed, steps=30, cfg=7.0
        )
        larger = w.add_node(LatentUpscaleBy, samples=base, upscale_method="bislerp", scale_by=1.25)
        refined = ksample(
            w, model, positive, negative, larger.output["LATENT"], seed=seed, steps=18, denoise=0.4
        )
        save(w, decode(w, refined, vae), label)
    return w


# ---------------------------------------------------------------- recording and gallery


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def keep_files(job: dict, folder: Path) -> list[tuple[str, bytes]]:
    output = job.get("output")
    if not isinstance(output, dict):
        return []
    files = []
    for item in output.get("images") or []:
        if item.get("type") != "base64":
            continue
        # Only the base name, so a file name from the server stays inside the folder.
        name = Path(item["filename"]).name
        data = base64.b64decode(item["data"])
        (folder / name).write_bytes(data)
        files.append((name, data))
    return files


def run_and_record(endpoint, folder_name, title, description, payload, *, timeout,
                   uploads=None, check=None, fails_with=()):
    folder = GALLERY / folder_name
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    write_json(folder / "workflow.json", payload)
    for name, data in (uploads or {}).items():
        (folder / f"entrada-{name}").write_bytes(data)
    result: dict[str, Any] = {
        "title": title,
        "description": description,
        "status": "running",
        "nodes": len(payload),
        "class_types": sorted({node["class_type"] for node in payload.values()}),
        "expected_failure": list(fails_with),
    }
    write_json(folder / "result.json", result)
    write_gallery()
    started = time.monotonic()
    job: dict = {}
    try:
        job = endpoint.run(payload, timeout=timeout, images=uploads)
        files = keep_files(job, folder)
        if fails_with:
            assert job.get("status") == "FAILED", f"expected a failure, the job is {job.get('status')}"
            report = json.dumps(job, ensure_ascii=False)
            for text in fails_with:
                assert text in report, f"{text!r} is not in the job error: {job.get('error')}"
        else:
            assert job.get("status") == "COMPLETED", (
                f"the job is {job.get('status')}: {job.get('error')} {job.get('output')}"
            )
            if check is not None:
                check(Outputs(files))
    except BaseException as error:
        result.update(status="failed", message=str(error)[:6000])
        raise
    else:
        result["status"] = "passed"
    finally:
        output = job.get("output") if isinstance(job.get("output"), dict) else {}
        result.update(
            seconds=round(time.monotonic() - started, 1),
            job_id=job.get("id"),
            job_status=job.get("status"),
            job_error=job.get("error"),
            worker_messages=output.get("errors") or output.get("details"),
            execution_ms=job.get("executionTime"),
            queue_ms=job.get("delayTime"),
        )
        write_json(folder / "result.json", result)
        write_gallery()


STATUS_LABELS = {"passed": "passou", "failed": "falhou", "running": "rodando"}


def media_html(folder: Path, path: Path) -> str:
    source = html.escape(f"{folder.name}/{path.name}", quote=True)
    caption = html.escape(path.name)
    if path.suffix in (".png", ".webp"):
        small = path.suffix == ".png" and png_size(path.read_bytes()[:24])[0] < 256
        css = ' class="pixel"' if small else ""
        body = f'<a href="{source}"><img src="{source}" loading="lazy"{css} alt="{caption}"></a>'
    elif path.suffix in (".mp4", ".webm", ".mkv"):
        body = f'<video src="{source}" controls loop playsinline preload="metadata"></video>'
    elif path.suffix == ".wav":
        body = f'<audio src="{source}" controls preload="none"></audio>'
    else:
        body = f'<a class="file" href="{source}">{caption}</a>'
    return f"<figure>{body}<figcaption>{caption}</figcaption></figure>"


def card_html(folder: Path, result: dict) -> str:
    status = result.get("status", "running")
    media = [
        media_html(folder, path)
        for path in sorted(folder.iterdir())
        if path.is_file() and path.suffix != ".json"
    ]
    details = []
    if result.get("seconds") is not None:
        details.append(f"{result['seconds']} s")
    details.append(f"{result.get('nodes', '?')} nós")
    if result.get("expected_failure"):
        details.append("falha esperada")
    message = result.get("message") or result.get("job_error")
    error = f"<pre>{html.escape(str(message))}</pre>" if message else ""
    worker = result.get("worker_messages")
    worker_html = ""
    if worker:
        text = html.escape(json.dumps(worker, indent=2, ensure_ascii=False))
        worker_html = f"<details><summary>mensagens do worker</summary><pre>{text}</pre></details>"
    types = ", ".join(html.escape(name) for name in result.get("class_types", []))
    name = html.escape(folder.name)
    links = f'<a href="{name}/workflow.json">workflow.json</a> · <a href="{name}/result.json">result.json</a>'
    files = "".join(media) or '<p class="meta">sem arquivos</p>'
    return (
        f'<article class="card {status}"><header><h2>{name}</h2>'
        f'<span class="badge">{STATUS_LABELS.get(status, status)}</span></header>'
        f"<p>{html.escape(result.get('description', ''))}</p>"
        f'<p class="meta">{" · ".join(details)} · {links}</p>{error}{worker_html}'
        f'<div class="media">{files}</div>'
        f'<details><summary>tipos de nó</summary><p class="meta">{types}</p></details></article>'
    )


GALLERY_STYLE = """
:root { --bg: #f6f5f2; --card: #ffffff; --ink: #1d1d1f; --muted: #6b6b70; --line: #e3e1dc;
  --ok: #1f8a4c; --bad: #c0362c; --run: #b7791f; --accent: #3d5afe; }
@media (prefers-color-scheme: dark) { :root { --bg: #141416; --card: #1d1d21; --ink: #ececf1;
  --muted: #a0a0aa; --line: #2c2c33; --ok: #4cc38a; --bad: #ff6b5f; --run: #f0b54a;
  --accent: #8c9eff; } }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1280px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 26px; margin: 0 0 4px; }
.summary { color: var(--muted); margin: 0 0 24px; }
.summary b { color: var(--ink); }
.cards { display: grid; gap: 16px; }
.card { background: var(--card); border: 1px solid var(--line); border-left: 4px solid var(--run);
  border-radius: 10px; padding: 14px 16px; }
.card.passed { border-left-color: var(--ok); }
.card.failed { border-left-color: var(--bad); }
.card header { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.card h2 { font-size: 16px; margin: 0; font-family: ui-monospace, monospace; word-break: break-all; }
.badge { font-size: 12px; font-weight: 600; padding: 2px 10px; border-radius: 99px; color: #fff;
  background: var(--run); white-space: nowrap; }
.passed .badge { background: var(--ok); }
.failed .badge { background: var(--bad); }
.card p { margin: 6px 0; }
.meta { color: var(--muted); font-size: 13px; }
.meta a { color: var(--accent); }
pre { background: var(--bg); border: 1px solid var(--line); border-radius: 6px; padding: 8px;
  overflow-x: auto; white-space: pre-wrap; font-size: 12px; }
.media { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 10px;
  margin-top: 10px; }
figure { margin: 0; }
figure audio { width: 100%; }
figure img, figure video { width: 100%; height: auto; border-radius: 6px; border: 1px solid var(--line);
  background: repeating-conic-gradient(#8882 0 25%, transparent 0 50%) 0 0 / 12px 12px; }
img.pixel { image-rendering: pixelated; }
figcaption { font-size: 11px; color: var(--muted); word-break: break-all; }
a.file { display: block; padding: 24px 8px; text-align: center; border: 1px dashed var(--line);
  border-radius: 6px; color: var(--accent); }
details summary { cursor: pointer; color: var(--muted); font-size: 13px; margin-top: 6px; }
"""


def write_gallery() -> None:
    GALLERY.mkdir(parents=True, exist_ok=True)
    cards, counts = [], {"passed": 0, "failed": 0, "running": 0}
    for folder in sorted(path for path in GALLERY.iterdir() if path.is_dir()):
        try:
            result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        status = result.get("status", "running")
        counts[status] = counts.get(status, 0) + 1
        cards.append(card_html(folder, result))
    summary = (
        f"<b>{len(cards)}</b> workflows · <b>{counts['passed']}</b> passaram · "
        f"<b>{counts['failed']}</b> falharam · <b>{counts['running']}</b> rodando · "
        f"atualizado em {time.strftime('%d/%m/%Y %H:%M:%S')}"
    )
    page = (
        '<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>Galeria de workflows</title><style>{GALLERY_STYLE}</style></head><body><main>"
        "<h1>Galeria de workflows no RunPod</h1>"
        f'<p class="summary">{summary}</p><div class="cards">{"".join(cards)}</div>'
        "</main></body></html>"
    )
    temporary = GALLERY / f".index-{os.getpid()}-{threading.get_ident()}.html"
    temporary.write_text(page, encoding="utf-8")
    os.replace(temporary, GALLERY / "index.html")


# ---------------------------------------------------------------- tests

BUILT = [c for c in CASES if c.library_says is None]
RAW = [c for c in CASES if c.library_says is not None]


@pytest.mark.parametrize("case", BUILT, ids=[c.name for c in BUILT])
def test_workflow_survives_a_reload(case):
    payload, _ = case.payload()
    workflow = Workflow()

    workflow.load_workflow(json.loads(json.dumps(payload)))

    assert sent(workflow) == payload


@pytest.mark.parametrize("case", RAW, ids=[c.name for c in RAW])
def test_library_refuses_what_comfy_ui_refuses(case):
    with pytest.raises(InvalidWorkflowError) as excinfo:
        Workflow().load_workflow(case.build())

    assert case.library_says in str(excinfo.value)


def test_there_are_a_hundred_different_workflows():
    payloads = {json.dumps(case.payload()[0], sort_keys=True) for case in CASES}

    assert len(CASES) == 100
    assert len({case.name for case in CASES}) == 100
    assert len(payloads) == 100


def test_the_workflows_cover_many_node_types():
    used = {node["class_type"] for case in CASES for node in case.payload()[0].values()}

    assert len(used) >= 165


def test_special_workflow_survives_a_reload():
    payload = sent(special_workflow())
    workflow = Workflow()

    workflow.load_workflow(payload)

    assert sent(workflow) == payload


def filtered_png(width: int, height: int, pixel: Callable) -> bytes:
    # Each row uses another PNG filter (none, sub, up, average, paeth), as encoders do.
    def predictor(kind, left, up, up_left):
        if kind == 1:
            return left
        if kind == 2:
            return up
        if kind == 3:
            return (left + up) // 2
        if kind == 4:
            p = left + up - up_left
            pa, pb, pc = abs(p - left), abs(p - up), abs(p - up_left)
            return left if pa <= pb and pa <= pc else up if pb <= pc else up_left
        return 0

    raw = [bytes(v for x in range(width) for v in pixel(x, y)) for y in range(height)]
    body, previous = b"", bytes(width * 3)
    for y, row in enumerate(raw):
        kind = y % 5
        line = bytes(
            (row[i] - predictor(
                kind,
                row[i - 3] if i >= 3 else 0,
                previous[i],
                previous[i - 3] if i >= 3 else 0,
            )) & 0xFF
            for i in range(len(row))
        )
        body += bytes([kind]) + line
        previous = row
    plain = encode_png(width, height, pixel)
    start = plain.index(b"IDAT") - 4
    (length,) = struct.unpack(">I", plain[start : start + 4])
    idat = zlib.compress(body)
    chunk = struct.pack(">I", len(idat)) + b"IDAT" + idat + struct.pack(">I", zlib.crc32(b"IDAT" + idat))
    return plain[:start] + chunk + plain[start + 12 + length :]


def test_png_decoder_undoes_every_row_filter():
    picture = decode_png(filtered_png(24, 16, pattern))

    assert picture.size == (24, 16)
    assert all(picture.pixel(x, y) == pattern(x, y) for y in range(16) for x in range(24))


class FakeEndpoint:
    def __init__(self, job: dict) -> None:
        self.job = job

    def run(self, workflow: dict, timeout: float, images=None) -> dict:
        return self.job


def red_job() -> dict:
    data = base64.b64encode(encode_png(4, 4, lambda x, y: (255, 0, 0))).decode()
    output = {"images": [{"filename": "red_00001_.png", "type": "base64", "data": data}]}
    return {"id": "job-1", "status": "COMPLETED", "output": output}


def test_a_passing_job_is_recorded_in_the_gallery(tmp_path, monkeypatch):
    monkeypatch.setitem(globals(), "GALLERY", tmp_path)

    def check(out):
        assert_pixel(out.picture("red"), 1, 1, RED, 0)

    run_and_record(
        FakeEndpoint(red_job()), "001-red", "red", "um quadrado vermelho",
        {"1": {"class_type": "EmptyImage", "inputs": {}}}, timeout=1, check=check,
    )

    result = json.loads((tmp_path / "001-red" / "result.json").read_text(encoding="utf-8"))
    page = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert result["status"] == "passed"
    assert (tmp_path / "001-red" / "red_00001_.png").exists()
    assert "001-red/red_00001_.png" in page and "passou" in page


def test_a_failing_check_is_recorded_and_still_fails(tmp_path, monkeypatch):
    monkeypatch.setitem(globals(), "GALLERY", tmp_path)

    def check(out):
        assert_pixel(out.picture("red"), 1, 1, BLUE, 0)

    with pytest.raises(AssertionError, match="expected"):
        run_and_record(
            FakeEndpoint(red_job()), "002-blue", "blue", "devia ser azul",
            {"1": {"class_type": "EmptyImage", "inputs": {}}}, timeout=1, check=check,
        )

    result = json.loads((tmp_path / "002-blue" / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    assert "expected (0, 0, 255)" in result["message"]
    assert "falhou" in (tmp_path / "index.html").read_text(encoding="utf-8")


@pytest.mark.runpod
@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_workflow_runs_on_comfy_ui(case, runpod_endpoint):
    payload, check = case.payload()
    number = CASES.index(case) + 1

    run_and_record(
        runpod_endpoint,
        f"{number:03}-{case.name}",
        case.name,
        case.description,
        payload,
        timeout=case.timeout,
        uploads=case.uploads,
        check=check,
        fails_with=case.fails_with,
    )


@pytest.mark.runpod
def test_special_boy_in_the_street_and_girl_at_the_lagoon(runpod_endpoint):
    def check(out):
        for label, _, _ in SPECIAL_SCENES:
            assert_generated(out.one(label), (1520, 1040), 200_000)

    run_and_record(
        runpod_endpoint,
        "000-especial-menino-e-menina",
        "especial",
        "Especial: um menino correndo no meio da rua e uma menina de vestido na beira de uma "
        "lagoa jogando pedrinhas, SDXL 1216x832 com 30 passos e refinamento em 1520x1040.",
        sent(special_workflow()),
        timeout=GPU_TIMEOUT,
        check=check,
    )
