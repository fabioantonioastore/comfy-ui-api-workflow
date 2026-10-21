from typing import Literal, get_args

from ...base import BaseNode
from ...validation import validate_bool, validate_choice, validate_range


class CLIPTextEncode(BaseNode):
    __slots__ = ("text", "clip")
    CATEGORY = "model/conditioning"
    OUTPUTS = {"CONDITIONING": 0}
    RETURN_TYPES = ("CONDITIONING",)
    INPUT_SOCKET_TYPES = {"text": "STRING", "clip": "CLIP"}

    def __init__(
        self, node_id: str, text: str | tuple[str, int], clip: tuple[str, int]
    ) -> None:
        super().__init__(node_id=node_id)
        self.text = text
        self.clip = clip


class CLIPSetLastLayer(BaseNode):
    __slots__ = ("clip", "stop_at_clip_layer")
    CATEGORY = "model/conditioning"
    OUTPUTS = {"CLIP": 0}
    RETURN_TYPES = ("CLIP",)
    INPUT_SOCKET_TYPES = {"clip": "CLIP", "stop_at_clip_layer": "INT"}

    def __init__(
        self,
        node_id: str,
        clip: tuple[str, int],
        stop_at_clip_layer: int | tuple[str, int] = -1,
    ) -> None:
        super().__init__(node_id=node_id)
        self.clip = clip
        self.stop_at_clip_layer = stop_at_clip_layer
        self.__validate()

    def __validate(self) -> None:
        validate_range("stop_at_clip_layer", self.stop_at_clip_layer, -24, -1)


CLIPVisionEncode_Crops = Literal["center", "none"]
CLIPVISIONENCODE_CROPS = frozenset(get_args(CLIPVisionEncode_Crops))


class CLIPVisionEncode(BaseNode):
    __slots__ = ("clip_vision", "image", "crop")
    CATEGORY = "model/conditioning"
    OUTPUTS = {"CLIP_VISION_OUTPUT": 0}
    RETURN_TYPES = ("CLIP_VISION_OUTPUT",)
    INPUT_SOCKET_TYPES = {
        "clip_vision": "CLIP_VISION",
        "image": "IMAGE",
        "crop": "V1 COMBO",
    }

    def __init__(
        self,
        node_id: str,
        clip_vision: tuple[str, int],
        image: tuple[str, int],
        crop: CLIPVisionEncode_Crops | tuple[str, int],
    ) -> None:
        super().__init__(node_id=node_id)
        self.clip_vision = clip_vision
        self.image = image
        self.crop = crop
        self.__validate()

    def __validate(self) -> None:
        validate_choice("crop", self.crop, CLIPVISIONENCODE_CROPS)


StyleModelApply_Strength_Types = Literal["multiply", "attn_bias"]
STYLEMODELAPPLY_STRENGTH_TYPES = frozenset(get_args(StyleModelApply_Strength_Types))


class StyleModelApply(BaseNode):
    __slots__ = (
        "conditioning",
        "style_model",
        "clip_vision_output",
        "strength_type",
        "strength",
    )
    CATEGORY = "model/conditioning"
    OUTPUTS = {"CONDITIONING": 0}
    RETURN_TYPES = ("CONDITIONING",)
    INPUT_SOCKET_TYPES = {
        "conditioning": "CONDITIONING",
        "style_model": "STYLE_MODEL",
        "clip_vision_output": "CLIP_VISION_OUTPUT",
        "strength": "FLOAT",
        "strength_type": "V1 COMBO",
    }

    def __init__(
        self,
        node_id: str,
        conditioning: tuple[str, int],
        style_model: tuple[str, int],
        clip_vision_output: tuple[str, int],
        strength_type: StyleModelApply_Strength_Types | tuple[str, int],
        strength: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.conditioning = conditioning
        self.style_model = style_model
        self.clip_vision_output = clip_vision_output
        self.strength_type = strength_type
        self.strength = strength
        self.__validate()

    def __validate(self) -> None:
        validate_choice(
            "strength_type", self.strength_type, STYLEMODELAPPLY_STRENGTH_TYPES
        )
        validate_range("strength", self.strength, 0.0, 10.0)


class unCLIPConditioning(BaseNode):
    __slots__ = ("conditioning", "clip_vision_output", "strength", "noise_augmentation")
    CATEGORY = "model/conditioning"
    OUTPUTS = {"CONDITIONING": 0}
    RETURN_TYPES = ("CONDITIONING",)
    INPUT_SOCKET_TYPES = {
        "conditioning": "CONDITIONING",
        "clip_vision_output": "CLIP_VISION_OUTPUT",
        "strength": "FLOAT",
        "noise_augmentation": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        conditioning: tuple[str, int],
        clip_vision_output: tuple[str, int],
        strength: float | tuple[str, int] = 1.0,
        noise_augmentation: float | tuple[str, int] = 0.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.conditioning = conditioning
        self.clip_vision_output = clip_vision_output
        self.strength = strength
        self.noise_augmentation = noise_augmentation
        self.__validate()

    def __validate(self) -> None:
        validate_range("strength", self.strength, -10.0, 10.0)
        validate_range("noise_augmentation", self.noise_augmentation, 0.0, 1.0)


class InpaintModelConditioning(BaseNode):
    __slots__ = ("positive", "negative", "vae", "pixels", "mask", "noise_mask")
    CATEGORY = "model/conditioning"
    OUTPUTS = {"positive": 0, "negative": 1, "latent": 2}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    INPUT_SOCKET_TYPES = {
        "positive": "CONDITIONING",
        "negative": "CONDITIONING",
        "vae": "VAE",
        "pixels": "IMAGE",
        "mask": "MASK",
        "noise_mask": "BOOLEAN",
    }

    def __init__(
        self,
        node_id: str,
        positive: tuple[str, int],
        negative: tuple[str, int],
        vae: tuple[str, int],
        pixels: tuple[str, int],
        mask: tuple[str, int],
        noise_mask: bool | tuple[str, int] = True,
    ) -> None:
        super().__init__(node_id=node_id)
        self.positive = positive
        self.negative = negative
        self.vae = vae
        self.pixels = pixels
        self.mask = mask
        self.noise_mask = noise_mask
        self.__validate()

    def __validate(self) -> None:
        validate_bool("noise_mask", self.noise_mask)


class CLIPTextEncodeControlnet(BaseNode):
    __slots__ = ("clip", "conditioning", "text")
    CATEGORY = "model/conditioning"
    OUTPUTS = {"CONDITIONING": 0}
    RETURN_TYPES = ("CONDITIONING",)
    INPUT_SOCKET_TYPES = {
        "clip": "CLIP",
        "conditioning": "CONDITIONING",
        "text": "STRING",
    }

    def __init__(
        self,
        node_id: str,
        clip: tuple[str, int],
        conditioning: tuple[str, int],
        text: str | tuple[str, int],
    ) -> None:
        super().__init__(node_id=node_id)
        self.clip = clip
        self.conditioning = conditioning
        self.text = text


class T5TokenizerOptions(BaseNode):
    __slots__ = ("clip", "min_padding", "min_length")
    CATEGORY = "model/conditioning"
    OUTPUTS = {"CLIP": 0}
    RETURN_TYPES = ("CLIP",)
    INPUT_SOCKET_TYPES = {"clip": "CLIP", "min_padding": "INT", "min_length": "INT"}

    def __init__(
        self,
        node_id: str,
        clip: tuple[str, int],
        min_padding: int | tuple[str, int] = 0,
        min_length: int | tuple[str, int] = 0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.clip = clip
        self.min_padding = min_padding
        self.min_length = min_length
        self.__validate()

    def __validate(self) -> None:
        validate_range("min_padding", self.min_padding, 0, 10_000)
        validate_range("min_length", self.min_length, 0, 10_000)


class SaveConditioning(BaseNode):
    __slots__ = ("conditioning", "filename_prefix")
    CATEGORY = "model/conditioning"
    OUTPUTS = {"conditioning": 0}
    RETURN_TYPES = ("CONDITIONING",)
    INPUT_SOCKET_TYPES = {"conditioning": "CONDITIONING", "filename_prefix": "STRING"}

    def __init__(
        self,
        node_id: str,
        conditioning: tuple[str, int],
        filename_prefix: str | tuple[str, int] = "conditioning/ComfyUI",
    ) -> None:
        super().__init__(node_id=node_id)
        self.conditioning = conditioning
        self.filename_prefix = filename_prefix


class ReferenceTimbreAudio(BaseNode):
    __slots__ = ("conditioning", "latent")
    CATEGORY = "model/conditioning"
    OUTPUTS = {"CONDITIONING": 0}
    RETURN_TYPES = ("CONDITIONING",)
    INPUT_SOCKET_TYPES = {"conditioning": "CONDITIONING", "latent": "LATENT"}

    def __init__(
        self,
        node_id: str,
        conditioning: tuple[str, int],
        latent: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.conditioning = conditioning
        if latent is not None:
            self.latent = latent


class ReferenceLatent(BaseNode):
    __slots__ = ("conditioning", "latent")
    CATEGORY = "model/conditioning"
    OUTPUTS = {"CONDITIONING": 0}
    RETURN_TYPES = ("CONDITIONING",)
    INPUT_SOCKET_TYPES = {"conditioning": "CONDITIONING", "latent": "LATENT"}

    def __init__(
        self,
        node_id: str,
        conditioning: tuple[str, int],
        latent: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.conditioning = conditioning
        if latent is not None:
            self.latent = latent


class SeedVR2Conditioning(BaseNode):
    __slots__ = ("model", "vae_conditioning")
    CATEGORY = "model/conditioning"
    OUTPUTS = {"positive": 0, "negative": 1}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    INPUT_SOCKET_TYPES = {"model": "MODEL", "vae_conditioning": "LATENT"}

    def __init__(
        self, node_id: str, model: tuple[str, int], vae_conditioning: tuple[str, int]
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.vae_conditioning = vae_conditioning


PiDConditioning_Latent_Formats = Literal["flux", "sd3", "sdxl", "qwenimage"]
PIDCONDITIONING_LATENT_FORMATS = frozenset(get_args(PiDConditioning_Latent_Formats))


class PiDConditioning(BaseNode):
    __slots__ = ("positive", "latent", "latent_format", "degrade_sigma")
    CATEGORY = "model/conditioning"
    OUTPUTS = {"CONDITIONING": 0}
    RETURN_TYPES = ("CONDITIONING",)
    INPUT_SOCKET_TYPES = {
        "positive": "CONDITIONING",
        "latent": "LATENT",
        "latent_format": "COMBO",
        "degrade_sigma": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        positive: tuple[str, int],
        latent: tuple[str, int],
        latent_format: PiDConditioning_Latent_Formats | tuple[str, int] = "flux",
        degrade_sigma: float | tuple[str, int] = 0.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.positive = positive
        self.latent = latent
        self.latent_format = latent_format
        self.degrade_sigma = degrade_sigma
        self.__validate()

    def __validate(self) -> None:
        validate_choice(
            "latent_format", self.latent_format, PIDCONDITIONING_LATENT_FORMATS
        )
        validate_range("degrade_sigma", self.degrade_sigma, 0.0, 1.0)


class AudioEncoderEncode(BaseNode):
    __slots__ = ("audio_encoder", "audio")
    CATEGORY = "model/conditioning"
    OUTPUTS = {"AUDIO_ENCODER_OUTPUT": 0}
    RETURN_TYPES = ("AUDIO_ENCODER_OUTPUT",)
    INPUT_SOCKET_TYPES = {"audio_encoder": "AUDIO_ENCODER", "audio": "AUDIO"}

    def __init__(
        self, node_id: str, audio_encoder: tuple[str, int], audio: tuple[str, int]
    ) -> None:
        super().__init__(node_id=node_id)
        self.audio_encoder = audio_encoder
        self.audio = audio


class NormalizeVideoLatentStart(BaseNode):
    __slots__ = ("latent", "start_frame_count", "reference_frame_count")
    CATEGORY = "model/conditioning"
    OUTPUTS = {"latent": 0}
    RETURN_TYPES = ("LATENT",)
    INPUT_SOCKET_TYPES = {
        "latent": "LATENT",
        "start_frame_count": "INT",
        "reference_frame_count": "INT",
    }

    def __init__(
        self,
        node_id: str,
        latent: tuple[str, int],
        start_frame_count: int | tuple[str, int] = 4,
        reference_frame_count: int | tuple[str, int] = 5,
    ) -> None:
        super().__init__(node_id=node_id)
        self.latent = latent
        self.start_frame_count = start_frame_count
        self.reference_frame_count = reference_frame_count
        self.__validate()

    def __validate(self) -> None:
        validate_range("start_frame_count", self.start_frame_count, 1, 16_384)
        validate_range("reference_frame_count", self.reference_frame_count, 1, 16_384)


from . import (
    ace,
    autoregressive,
    bernini,
    boogu,
    controlnet,
    cosmos,
    flux,
    gligen,
    hidream,
    hunyuan_3d,
    hunyuan_image,
    hunyuan_video,
    instructpix2pix,
    joyimage,
    kandinsky,
    lotus,
    ltxv,
    lumina,
    mage,
    ming_image,
    minimax,
    minimax_music,
    photomaker,
    pixart,
    qwen_image,
    stable_audio,
    stable_cascade,
    stable_diffusion,
    stable_diffusion_upscaler,
    stable_video,
    stable_video_3d,
    stable_zero123,
    transform,
    trellis,
    triposplat,
    void,
    wan,
    yue2,
    z_image,
)
