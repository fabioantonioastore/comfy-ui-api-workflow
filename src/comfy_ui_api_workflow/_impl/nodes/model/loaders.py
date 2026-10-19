from typing import Literal, get_args

from ..base import BaseNode
from ..validation import (
    validate_bool,
    validate_choice,
    validate_not_empty,
    validate_range,
)
from ...errors.node import InvalidNodeParameterValue
from ..dynamic import DynamicInputs, Flag, Group, Number, ServerChoice, validate_dynamic


class CheckpointLoaderSimple(BaseNode):
    __slots__ = ("ckpt_name",)
    CATEGORY = "model/loaders"
    OUTPUTS = {"MODEL": 0, "CLIP": 1, "VAE": 2}
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    INPUT_SOCKET_TYPES = {"ckpt_name": "V1 COMBO"}

    def __init__(self, node_id: str, ckpt_name: str | tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        if not ckpt_name:
            raise InvalidNodeParameterValue("ckpt_name must not be empty")
        self.ckpt_name = ckpt_name


class VAELoader(BaseNode):
    __slots__ = ("vae_name",)
    CATEGORY = "model/loaders"
    OUTPUTS = {"VAE": 0}
    RETURN_TYPES = ("VAE",)
    INPUT_SOCKET_TYPES = {"vae_name": "V1 COMBO"}

    def __init__(self, node_id: str, vae_name: str | tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.vae_name = vae_name
        self.__validate()

    def __validate(self) -> None:
        validate_not_empty("vae_name", self.vae_name)


class LoraLoader(BaseNode):
    __slots__ = ("model", "clip", "lora_name", "strength_model", "strength_clip")
    CATEGORY = "model/loaders"
    OUTPUTS = {"MODEL": 0, "CLIP": 1}
    RETURN_TYPES = ("MODEL", "CLIP")
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "clip": "CLIP",
        "lora_name": "V1 COMBO",
        "strength_model": "FLOAT",
        "strength_clip": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        clip: tuple[str, int],
        lora_name: str | tuple[str, int],
        strength_model: float | tuple[str, int] = 1.0,
        strength_clip: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.clip = clip
        self.lora_name = lora_name
        self.strength_model = strength_model
        self.strength_clip = strength_clip
        self.__validate()

    def __validate(self) -> None:
        validate_not_empty("lora_name", self.lora_name)
        validate_range("strength_model", self.strength_model, -100.0, 100.0)
        validate_range("strength_clip", self.strength_clip, -100.0, 100.0)


CLIPLoader_Types = Literal[
    "stable_diffusion",
    "stable_cascade",
    "sd3",
    "stable_audio",
    "mochi",
    "ltxv",
    "pixart",
    "cosmos",
    "lumina2",
    "wan",
    "hidream",
    "chroma",
    "ace",
    "omnigen2",
    "qwen_image",
    "hunyuan_image",
    "flux2",
    "ovis",
    "longcat_image",
    "cogvideox",
    "lens",
    "pixeldit",
    "ideogram4",
    "boogu",
    "krea2",
    "joyimage",
    "mage",
    "minimax",
    "yue2",
]
CLIPLoader_Devices = Literal["default", "cpu"]
CLIPLOADER_TYPES = frozenset(get_args(CLIPLoader_Types))
CLIPLOADER_DEVICES = frozenset(get_args(CLIPLoader_Devices))


class CLIPLoader(BaseNode):
    __slots__ = ("clip_name", "type", "device")
    CATEGORY = "model/loaders"
    OUTPUTS = {"CLIP": 0}
    RETURN_TYPES = ("CLIP",)
    INPUT_SOCKET_TYPES = {
        "clip_name": "V1 COMBO",
        "type": "V1 COMBO",
        "device": "V1 COMBO",
    }

    def __init__(
        self,
        node_id: str,
        clip_name: str | tuple[str, int],
        type: CLIPLoader_Types | tuple[str, int],
        device: CLIPLoader_Devices | tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.clip_name = clip_name
        self.type = type
        if device is not None:
            self.device = device
        self.__validate()

    def __validate(self) -> None:
        validate_not_empty("clip_name", self.clip_name)
        validate_choice("type", self.type, CLIPLOADER_TYPES)
        if hasattr(self, "device"):
            validate_choice("device", self.device, CLIPLOADER_DEVICES)


UNETLoader_Weight_Dtypes = Literal[
    "default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"
]
UNETLOADER_WEIGHT_DTYPES = frozenset(get_args(UNETLoader_Weight_Dtypes))


class UNETLoader(BaseNode):
    __slots__ = ("unet_name", "weight_dtype")
    CATEGORY = "model/loaders"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {"unet_name": "V1 COMBO", "weight_dtype": "V1 COMBO"}

    def __init__(
        self,
        node_id: str,
        unet_name: str | tuple[str, int],
        weight_dtype: UNETLoader_Weight_Dtypes | tuple[str, int],
    ) -> None:
        super().__init__(node_id=node_id)
        self.unet_name = unet_name
        self.weight_dtype = weight_dtype
        self.__validate()

    def __validate(self) -> None:
        validate_not_empty("unet_name", self.unet_name)
        validate_choice("weight_dtype", self.weight_dtype, UNETLOADER_WEIGHT_DTYPES)


DualCLIPLoader_Types = Literal[
    "sdxl",
    "sd3",
    "flux",
    "hunyuan_video",
    "hidream",
    "hunyuan_image",
    "hunyuan_video_15",
    "kandinsky5",
    "kandinsky5_image",
    "ltxv",
    "newbie",
    "ace",
]
DualCLIPLoader_Devices = Literal["default", "cpu"]
DUALCLIPLOADER_TYPES = frozenset(get_args(DualCLIPLoader_Types))
DUALCLIPLOADER_DEVICES = frozenset(get_args(DualCLIPLoader_Devices))


class DualCLIPLoader(BaseNode):
    __slots__ = ("clip_name1", "clip_name2", "type", "device")
    CATEGORY = "model/loaders"
    OUTPUTS = {"CLIP": 0}
    RETURN_TYPES = ("CLIP",)
    INPUT_SOCKET_TYPES = {
        "clip_name1": "V1 COMBO",
        "clip_name2": "V1 COMBO",
        "type": "V1 COMBO",
        "device": "V1 COMBO",
    }

    def __init__(
        self,
        node_id: str,
        clip_name1: str | tuple[str, int],
        clip_name2: str | tuple[str, int],
        type: DualCLIPLoader_Types | tuple[str, int],
        device: DualCLIPLoader_Devices | tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.clip_name1 = clip_name1
        self.clip_name2 = clip_name2
        self.type = type
        if device is not None:
            self.device = device
        self.__validate()

    def __validate(self) -> None:
        validate_not_empty("clip_name1", self.clip_name1)
        validate_not_empty("clip_name2", self.clip_name2)
        validate_choice("type", self.type, DUALCLIPLOADER_TYPES)
        if hasattr(self, "device"):
            validate_choice("device", self.device, DUALCLIPLOADER_DEVICES)


class ControlNetLoader(BaseNode):
    __slots__ = ("control_net_name",)
    CATEGORY = "model/loaders"
    OUTPUTS = {"CONTROL_NET": 0}
    RETURN_TYPES = ("CONTROL_NET",)
    INPUT_SOCKET_TYPES = {"control_net_name": "V1 COMBO"}

    def __init__(self, node_id: str, control_net_name: str | tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.control_net_name = control_net_name
        self.__validate()

    def __validate(self) -> None:
        validate_not_empty("control_net_name", self.control_net_name)


class DiffControlNetLoader(BaseNode):
    __slots__ = ("model", "control_net_name")
    CATEGORY = "model/loaders"
    OUTPUTS = {"CONTROL_NET": 0}
    RETURN_TYPES = ("CONTROL_NET",)
    INPUT_SOCKET_TYPES = {"model": "MODEL", "control_net_name": "V1 COMBO"}

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        control_net_name: str | tuple[str, int],
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.control_net_name = control_net_name
        self.__validate()

    def __validate(self) -> None:
        validate_not_empty("control_net_name", self.control_net_name)


class StyleModelLoader(BaseNode):
    __slots__ = ("style_model_name",)
    CATEGORY = "model/loaders"
    OUTPUTS = {"STYLE_MODEL": 0}
    RETURN_TYPES = ("STYLE_MODEL",)
    INPUT_SOCKET_TYPES = {"style_model_name": "V1 COMBO"}

    def __init__(self, node_id: str, style_model_name: str | tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.style_model_name = style_model_name
        self.__validate()

    def __validate(self) -> None:
        validate_not_empty("style_model_name", self.style_model_name)


class CLIPVisionLoader(BaseNode):
    __slots__ = ("clip_name",)
    CATEGORY = "model/loaders"
    OUTPUTS = {"CLIP_VISION": 0}
    RETURN_TYPES = ("CLIP_VISION",)
    INPUT_SOCKET_TYPES = {"clip_name": "V1 COMBO"}

    def __init__(self, node_id: str, clip_name: str | tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.clip_name = clip_name
        self.__validate()

    def __validate(self) -> None:
        validate_not_empty("clip_name", self.clip_name)


class unCLIPCheckpointLoader(BaseNode):
    __slots__ = ("ckpt_name",)
    CATEGORY = "model/loaders"
    OUTPUTS = {"MODEL": 0, "CLIP": 1, "VAE": 2, "CLIP_VISION": 3}
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "CLIP_VISION")
    INPUT_SOCKET_TYPES = {"ckpt_name": "V1 COMBO"}

    def __init__(self, node_id: str, ckpt_name: str | tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.ckpt_name = ckpt_name
        self.__validate()

    def __validate(self) -> None:
        validate_not_empty("ckpt_name", self.ckpt_name)


class GLIGENLoader(BaseNode):
    __slots__ = ("gligen_name",)
    CATEGORY = "model/loaders"
    OUTPUTS = {"GLIGEN": 0}
    RETURN_TYPES = ("GLIGEN",)
    INPUT_SOCKET_TYPES = {"gligen_name": "V1 COMBO"}

    def __init__(self, node_id: str, gligen_name: str | tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.gligen_name = gligen_name
        self.__validate()

    def __validate(self) -> None:
        validate_not_empty("gligen_name", self.gligen_name)


class LoraLoaderModelOnly(BaseNode):
    __slots__ = ("model", "lora_name", "strength_model")
    CATEGORY = "model/loaders"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "lora_name": "V1 COMBO",
        "strength_model": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        lora_name: str | tuple[str, int],
        strength_model: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.lora_name = lora_name
        self.strength_model = strength_model
        self.__validate()

    def __validate(self) -> None:
        validate_not_empty("lora_name", self.lora_name)
        validate_range("strength_model", self.strength_model, -100.0, 100.0)


class HypernetworkLoader(BaseNode):
    __slots__ = ("model", "hypernetwork_name", "strength")
    CATEGORY = "model/loaders"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "hypernetwork_name": "COMBO",
        "strength": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        hypernetwork_name: str | tuple[str, int],
        strength: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.hypernetwork_name = hypernetwork_name
        self.strength = strength
        self.__validate()

    def __validate(self) -> None:
        validate_not_empty("hypernetwork_name", self.hypernetwork_name)
        validate_range("strength", self.strength, -10.0, 10.0)


class UpscaleModelLoader(BaseNode):
    __slots__ = ("model_name",)
    CATEGORY = "model/loaders"
    OUTPUTS = {"UPSCALE_MODEL": 0}
    RETURN_TYPES = ("UPSCALE_MODEL",)
    INPUT_SOCKET_TYPES = {"model_name": "COMBO"}

    def __init__(self, node_id: str, model_name: str | tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.model_name = model_name
        self.__validate()

    def __validate(self) -> None:
        validate_not_empty("model_name", self.model_name)


class ImageOnlyCheckpointLoader(BaseNode):
    __slots__ = ("ckpt_name",)
    CATEGORY = "model/loaders"
    OUTPUTS = {"MODEL": 0, "CLIP_VISION": 1, "VAE": 2}
    RETURN_TYPES = ("MODEL", "CLIP_VISION", "VAE")
    INPUT_SOCKET_TYPES = {"ckpt_name": "V1 COMBO"}

    def __init__(self, node_id: str, ckpt_name: str | tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.ckpt_name = ckpt_name
        self.__validate()

    def __validate(self) -> None:
        validate_not_empty("ckpt_name", self.ckpt_name)


class LoraModelLoader(BaseNode):
    __slots__ = ("model", "lora", "strength_model", "bypass")
    CATEGORY = "model/loaders"
    OUTPUTS = {"model": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "lora": "LORA_MODEL",
        "strength_model": "FLOAT",
        "bypass": "BOOLEAN",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        lora: tuple[str, int],
        strength_model: float | tuple[str, int] = 1.0,
        bypass: bool | tuple[str, int] = False,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.lora = lora
        self.strength_model = strength_model
        self.bypass = bypass
        self.__validate()

    def __validate(self) -> None:
        validate_range("strength_model", self.strength_model, -100.0, 100.0)
        validate_bool("bypass", self.bypass)


class PhotoMakerLoader(BaseNode):
    __slots__ = ("photomaker_model_name",)
    CATEGORY = "model/loaders"
    OUTPUTS = {"PHOTOMAKER": 0}
    RETURN_TYPES = ("PHOTOMAKER",)
    INPUT_SOCKET_TYPES = {"photomaker_model_name": "COMBO"}

    def __init__(
        self, node_id: str, photomaker_model_name: str | tuple[str, int]
    ) -> None:
        super().__init__(node_id=node_id)
        self.photomaker_model_name = photomaker_model_name
        self.__validate()

    def __validate(self) -> None:
        validate_not_empty("photomaker_model_name", self.photomaker_model_name)


class ConditioningLoader(BaseNode):
    __slots__ = ("conditioning_name",)
    CATEGORY = "model/loaders"
    OUTPUTS = {"CONDITIONING": 0}
    RETURN_TYPES = ("CONDITIONING",)
    INPUT_SOCKET_TYPES = {"conditioning_name": "COMBO"}

    def __init__(self, node_id: str, conditioning_name: str | tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.conditioning_name = conditioning_name
        self.__validate()

    def __validate(self) -> None:
        validate_not_empty("conditioning_name", self.conditioning_name)


class TripleCLIPLoader(BaseNode):
    __slots__ = ("clip_name1", "clip_name2", "clip_name3")
    CATEGORY = "model/loaders"
    OUTPUTS = {"CLIP": 0}
    RETURN_TYPES = ("CLIP",)
    INPUT_SOCKET_TYPES = {
        "clip_name1": "COMBO",
        "clip_name2": "COMBO",
        "clip_name3": "COMBO",
    }

    def __init__(
        self,
        node_id: str,
        clip_name1: str | tuple[str, int],
        clip_name2: str | tuple[str, int],
        clip_name3: str | tuple[str, int],
    ) -> None:
        super().__init__(node_id=node_id)
        self.clip_name1 = clip_name1
        self.clip_name2 = clip_name2
        self.clip_name3 = clip_name3
        self.__validate()

    def __validate(self) -> None:
        validate_not_empty("clip_name1", self.clip_name1)
        validate_not_empty("clip_name2", self.clip_name2)
        validate_not_empty("clip_name3", self.clip_name3)


class LatentUpscaleModelLoader(BaseNode):
    __slots__ = ("model_name",)
    CATEGORY = "model/loaders"
    OUTPUTS = {"LATENT_UPSCALE_MODEL": 0}
    RETURN_TYPES = ("LATENT_UPSCALE_MODEL",)
    INPUT_SOCKET_TYPES = {"model_name": "COMBO"}

    def __init__(self, node_id: str, model_name: str | tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.model_name = model_name
        self.__validate()

    def __validate(self) -> None:
        validate_not_empty("model_name", self.model_name)


class LTXVAudioVAELoader(BaseNode):
    __slots__ = ("ckpt_name",)
    CATEGORY = "model/loaders"
    OUTPUTS = {"Audio VAE": 0}
    RETURN_TYPES = ("VAE",)
    INPUT_SOCKET_TYPES = {"ckpt_name": "COMBO"}

    def __init__(self, node_id: str, ckpt_name: str | tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.ckpt_name = ckpt_name
        self.__validate()

    def __validate(self) -> None:
        validate_not_empty("ckpt_name", self.ckpt_name)


LTXAVTextEncoderLoader_Devices = Literal["default", "cpu"]
LTXAVTEXTENCODERLOADER_DEVICES = frozenset(get_args(LTXAVTextEncoderLoader_Devices))


class LTXAVTextEncoderLoader(BaseNode):
    __slots__ = ("text_encoder", "ckpt_name", "device")
    CATEGORY = "model/loaders"
    OUTPUTS = {"CLIP": 0}
    RETURN_TYPES = ("CLIP",)
    INPUT_SOCKET_TYPES = {
        "text_encoder": "COMBO",
        "ckpt_name": "COMBO",
        "device": "COMBO",
    }

    def __init__(
        self,
        node_id: str,
        text_encoder: str | tuple[str, int],
        ckpt_name: str | tuple[str, int],
        device: LTXAVTextEncoderLoader_Devices | tuple[str, int],
    ) -> None:
        super().__init__(node_id=node_id)
        self.text_encoder = text_encoder
        self.ckpt_name = ckpt_name
        self.device = device
        self.__validate()

    def __validate(self) -> None:
        validate_not_empty("text_encoder", self.text_encoder)
        validate_not_empty("ckpt_name", self.ckpt_name)
        validate_choice("device", self.device, LTXAVTEXTENCODERLOADER_DEVICES)


class QuadrupleCLIPLoader(BaseNode):
    __slots__ = ("clip_name1", "clip_name2", "clip_name3", "clip_name4")
    CATEGORY = "model/loaders"
    OUTPUTS = {"CLIP": 0}
    RETURN_TYPES = ("CLIP",)
    INPUT_SOCKET_TYPES = {
        "clip_name1": "COMBO",
        "clip_name2": "COMBO",
        "clip_name3": "COMBO",
        "clip_name4": "COMBO",
    }

    def __init__(
        self,
        node_id: str,
        clip_name1: str | tuple[str, int],
        clip_name2: str | tuple[str, int],
        clip_name3: str | tuple[str, int],
        clip_name4: str | tuple[str, int],
    ) -> None:
        super().__init__(node_id=node_id)
        self.clip_name1 = clip_name1
        self.clip_name2 = clip_name2
        self.clip_name3 = clip_name3
        self.clip_name4 = clip_name4
        self.__validate()

    def __validate(self) -> None:
        validate_not_empty("clip_name1", self.clip_name1)
        validate_not_empty("clip_name2", self.clip_name2)
        validate_not_empty("clip_name3", self.clip_name3)
        validate_not_empty("clip_name4", self.clip_name4)


class ModelPatchLoader(BaseNode):
    __slots__ = ("name",)
    CATEGORY = "model/loaders"
    OUTPUTS = {"MODEL_PATCH": 0}
    RETURN_TYPES = ("MODEL_PATCH",)
    INPUT_SOCKET_TYPES = {"name": "V1 COMBO"}

    def __init__(self, node_id: str, name: str | tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.name = name
        self.__validate()

    def __validate(self) -> None:
        validate_not_empty("name", self.name)


class AudioEncoderLoader(BaseNode):
    __slots__ = ("audio_encoder_name",)
    CATEGORY = "model/loaders"
    OUTPUTS = {"AUDIO_ENCODER": 0}
    RETURN_TYPES = ("AUDIO_ENCODER",)
    INPUT_SOCKET_TYPES = {"audio_encoder_name": "COMBO"}

    def __init__(self, node_id: str, audio_encoder_name: str | tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.audio_encoder_name = audio_encoder_name
        self.__validate()

    def __validate(self) -> None:
        validate_not_empty("audio_encoder_name", self.audio_encoder_name)


class LoraLoaderBypass(BaseNode):
    __slots__ = ("model", "clip", "lora_name", "strength_model", "strength_clip")
    CATEGORY = "model/loaders"
    OUTPUTS = {"MODEL": 0, "CLIP": 1}
    RETURN_TYPES = ("MODEL", "CLIP")
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "clip": "CLIP",
        "lora_name": "V1 COMBO",
        "strength_model": "FLOAT",
        "strength_clip": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        clip: tuple[str, int],
        lora_name: str | tuple[str, int],
        strength_model: float | tuple[str, int] = 1.0,
        strength_clip: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.clip = clip
        self.lora_name = lora_name
        self.strength_model = strength_model
        self.strength_clip = strength_clip
        self.__validate()

    def __validate(self) -> None:
        validate_not_empty("lora_name", self.lora_name)
        validate_range("strength_model", self.strength_model, -100.0, 100.0)
        validate_range("strength_clip", self.strength_clip, -100.0, 100.0)


class LoraLoaderBypassModelOnly(BaseNode):
    __slots__ = ("model", "lora_name", "strength_model")
    CATEGORY = "model/loaders"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "lora_name": "V1 COMBO",
        "strength_model": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        lora_name: str | tuple[str, int],
        strength_model: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.lora_name = lora_name
        self.strength_model = strength_model
        self.__validate()

    def __validate(self) -> None:
        validate_not_empty("lora_name", self.lora_name)
        validate_range("strength_model", self.strength_model, -100.0, 100.0)


class LoadBackgroundRemovalModel(BaseNode):
    __slots__ = ("bg_removal_name",)
    CATEGORY = "model/loaders"
    OUTPUTS = {"bg_model": 0}
    RETURN_TYPES = ("BACKGROUND_REMOVAL",)
    INPUT_SOCKET_TYPES = {"bg_removal_name": "COMBO"}

    def __init__(self, node_id: str, bg_removal_name: str | tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.bg_removal_name = bg_removal_name
        self.__validate()

    def __validate(self) -> None:
        validate_not_empty("bg_removal_name", self.bg_removal_name)


class FrameInterpolationModelLoader(BaseNode):
    __slots__ = ("model_name",)
    CATEGORY = "model/loaders"
    OUTPUTS = {"INTERP_MODEL": 0}
    RETURN_TYPES = ("INTERP_MODEL",)
    INPUT_SOCKET_TYPES = {"model_name": "COMBO"}

    def __init__(self, node_id: str, model_name: str | tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.model_name = model_name
        self.__validate()

    def __validate(self) -> None:
        validate_not_empty("model_name", self.model_name)


class OpticalFlowLoader(BaseNode):
    __slots__ = ("model_name",)
    CATEGORY = "model/loaders"
    OUTPUTS = {"OPTICAL_FLOW": 0}
    RETURN_TYPES = ("OPTICAL_FLOW",)
    INPUT_SOCKET_TYPES = {"model_name": "COMBO"}

    def __init__(self, node_id: str, model_name: str | tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.model_name = model_name
        self.__validate()

    def __validate(self) -> None:
        validate_not_empty("model_name", self.model_name)


class LoadMoGeModel(BaseNode):
    __slots__ = ("model_name",)
    CATEGORY = "model/loaders"
    OUTPUTS = {"MOGE_MODEL": 0}
    RETURN_TYPES = ("MOGE_MODEL",)
    INPUT_SOCKET_TYPES = {"model_name": "COMBO"}

    def __init__(self, node_id: str, model_name: str | tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.model_name = model_name
        self.__validate()

    def __validate(self) -> None:
        validate_not_empty("model_name", self.model_name)


class LoadMediaPipeFaceLandmarker(BaseNode):
    __slots__ = ("model_name",)
    CATEGORY = "model/loaders"
    OUTPUTS = {"FACE_DETECTION_MODEL": 0}
    RETURN_TYPES = ("FACE_DETECTION_MODEL",)
    INPUT_SOCKET_TYPES = {"model_name": "COMBO"}

    def __init__(self, node_id: str, model_name: str | tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.model_name = model_name
        self.__validate()

    def __validate(self) -> None:
        validate_not_empty("model_name", self.model_name)


LoadDA3Model_Weight_Dtypes = Literal["default", "fp16", "bf16", "fp32"]
LOADDA3MODEL_WEIGHT_DTYPES = frozenset(get_args(LoadDA3Model_Weight_Dtypes))


class LoadDA3Model(BaseNode):
    __slots__ = ("model_name", "weight_dtype")
    CATEGORY = "model/loaders"
    OUTPUTS = {"DA3_MODEL": 0}
    RETURN_TYPES = ("DA3_MODEL",)
    INPUT_SOCKET_TYPES = {"model_name": "COMBO", "weight_dtype": "COMBO"}

    def __init__(
        self,
        node_id: str,
        model_name: str | tuple[str, int],
        weight_dtype: LoadDA3Model_Weight_Dtypes | tuple[str, int] = "default",
    ) -> None:
        super().__init__(node_id=node_id)
        self.model_name = model_name
        self.weight_dtype = weight_dtype
        self.__validate()

    def __validate(self) -> None:
        validate_not_empty("model_name", self.model_name)
        validate_choice("weight_dtype", self.weight_dtype, LOADDA3MODEL_WEIGHT_DTYPES)


class LoadLoraModel(DynamicInputs, BaseNode):
    __slots__ = (
        "model",
        "loras_0_lora_name",
        "loras_0_strength",
        "loras_0_enabled",
        "loras_1_lora_name",
        "loras_1_strength",
        "loras_1_enabled",
        "loras_2_lora_name",
        "loras_2_strength",
        "loras_2_enabled",
        "loras_3_lora_name",
        "loras_3_strength",
        "loras_3_enabled",
        "loras_4_lora_name",
        "loras_4_strength",
        "loras_4_enabled",
        "loras_5_lora_name",
        "loras_5_strength",
        "loras_5_enabled",
        "loras_6_lora_name",
        "loras_6_strength",
        "loras_6_enabled",
        "loras_7_lora_name",
        "loras_7_strength",
        "loras_7_enabled",
        "loras_8_lora_name",
        "loras_8_strength",
        "loras_8_enabled",
        "loras_9_lora_name",
        "loras_9_strength",
        "loras_9_enabled",
        "loras_10_lora_name",
        "loras_10_strength",
        "loras_10_enabled",
        "loras_11_lora_name",
        "loras_11_strength",
        "loras_11_enabled",
        "loras_12_lora_name",
        "loras_12_strength",
        "loras_12_enabled",
        "loras_13_lora_name",
        "loras_13_strength",
        "loras_13_enabled",
        "loras_14_lora_name",
        "loras_14_strength",
        "loras_14_enabled",
        "loras_15_lora_name",
        "loras_15_strength",
        "loras_15_enabled",
        "loras_16_lora_name",
        "loras_16_strength",
        "loras_16_enabled",
        "loras_17_lora_name",
        "loras_17_strength",
        "loras_17_enabled",
        "loras_18_lora_name",
        "loras_18_strength",
        "loras_18_enabled",
        "loras_19_lora_name",
        "loras_19_strength",
        "loras_19_enabled",
    )
    CATEGORY = "model/loaders"
    OUTPUTS = {"MODEL": 0}
    RETURN_TYPES = ("MODEL",)
    INPUT_NAMES = {
        "loras_0_lora_name": "loras.0.lora_name",
        "loras_0_strength": "loras.0.strength",
        "loras_0_enabled": "loras.0.enabled",
        "loras_1_lora_name": "loras.1.lora_name",
        "loras_1_strength": "loras.1.strength",
        "loras_1_enabled": "loras.1.enabled",
        "loras_2_lora_name": "loras.2.lora_name",
        "loras_2_strength": "loras.2.strength",
        "loras_2_enabled": "loras.2.enabled",
        "loras_3_lora_name": "loras.3.lora_name",
        "loras_3_strength": "loras.3.strength",
        "loras_3_enabled": "loras.3.enabled",
        "loras_4_lora_name": "loras.4.lora_name",
        "loras_4_strength": "loras.4.strength",
        "loras_4_enabled": "loras.4.enabled",
        "loras_5_lora_name": "loras.5.lora_name",
        "loras_5_strength": "loras.5.strength",
        "loras_5_enabled": "loras.5.enabled",
        "loras_6_lora_name": "loras.6.lora_name",
        "loras_6_strength": "loras.6.strength",
        "loras_6_enabled": "loras.6.enabled",
        "loras_7_lora_name": "loras.7.lora_name",
        "loras_7_strength": "loras.7.strength",
        "loras_7_enabled": "loras.7.enabled",
        "loras_8_lora_name": "loras.8.lora_name",
        "loras_8_strength": "loras.8.strength",
        "loras_8_enabled": "loras.8.enabled",
        "loras_9_lora_name": "loras.9.lora_name",
        "loras_9_strength": "loras.9.strength",
        "loras_9_enabled": "loras.9.enabled",
        "loras_10_lora_name": "loras.10.lora_name",
        "loras_10_strength": "loras.10.strength",
        "loras_10_enabled": "loras.10.enabled",
        "loras_11_lora_name": "loras.11.lora_name",
        "loras_11_strength": "loras.11.strength",
        "loras_11_enabled": "loras.11.enabled",
        "loras_12_lora_name": "loras.12.lora_name",
        "loras_12_strength": "loras.12.strength",
        "loras_12_enabled": "loras.12.enabled",
        "loras_13_lora_name": "loras.13.lora_name",
        "loras_13_strength": "loras.13.strength",
        "loras_13_enabled": "loras.13.enabled",
        "loras_14_lora_name": "loras.14.lora_name",
        "loras_14_strength": "loras.14.strength",
        "loras_14_enabled": "loras.14.enabled",
        "loras_15_lora_name": "loras.15.lora_name",
        "loras_15_strength": "loras.15.strength",
        "loras_15_enabled": "loras.15.enabled",
        "loras_16_lora_name": "loras.16.lora_name",
        "loras_16_strength": "loras.16.strength",
        "loras_16_enabled": "loras.16.enabled",
        "loras_17_lora_name": "loras.17.lora_name",
        "loras_17_strength": "loras.17.strength",
        "loras_17_enabled": "loras.17.enabled",
        "loras_18_lora_name": "loras.18.lora_name",
        "loras_18_strength": "loras.18.strength",
        "loras_18_enabled": "loras.18.enabled",
        "loras_19_lora_name": "loras.19.lora_name",
        "loras_19_strength": "loras.19.strength",
        "loras_19_enabled": "loras.19.enabled",
    }
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "loras_0_lora_name": "COMBO",
        "loras_0_strength": "FLOAT",
        "loras_0_enabled": "BOOLEAN",
        "loras_1_lora_name": "COMBO",
        "loras_1_strength": "FLOAT",
        "loras_1_enabled": "BOOLEAN",
        "loras_2_lora_name": "COMBO",
        "loras_2_strength": "FLOAT",
        "loras_2_enabled": "BOOLEAN",
        "loras_3_lora_name": "COMBO",
        "loras_3_strength": "FLOAT",
        "loras_3_enabled": "BOOLEAN",
        "loras_4_lora_name": "COMBO",
        "loras_4_strength": "FLOAT",
        "loras_4_enabled": "BOOLEAN",
        "loras_5_lora_name": "COMBO",
        "loras_5_strength": "FLOAT",
        "loras_5_enabled": "BOOLEAN",
        "loras_6_lora_name": "COMBO",
        "loras_6_strength": "FLOAT",
        "loras_6_enabled": "BOOLEAN",
        "loras_7_lora_name": "COMBO",
        "loras_7_strength": "FLOAT",
        "loras_7_enabled": "BOOLEAN",
        "loras_8_lora_name": "COMBO",
        "loras_8_strength": "FLOAT",
        "loras_8_enabled": "BOOLEAN",
        "loras_9_lora_name": "COMBO",
        "loras_9_strength": "FLOAT",
        "loras_9_enabled": "BOOLEAN",
        "loras_10_lora_name": "COMBO",
        "loras_10_strength": "FLOAT",
        "loras_10_enabled": "BOOLEAN",
        "loras_11_lora_name": "COMBO",
        "loras_11_strength": "FLOAT",
        "loras_11_enabled": "BOOLEAN",
        "loras_12_lora_name": "COMBO",
        "loras_12_strength": "FLOAT",
        "loras_12_enabled": "BOOLEAN",
        "loras_13_lora_name": "COMBO",
        "loras_13_strength": "FLOAT",
        "loras_13_enabled": "BOOLEAN",
        "loras_14_lora_name": "COMBO",
        "loras_14_strength": "FLOAT",
        "loras_14_enabled": "BOOLEAN",
        "loras_15_lora_name": "COMBO",
        "loras_15_strength": "FLOAT",
        "loras_15_enabled": "BOOLEAN",
        "loras_16_lora_name": "COMBO",
        "loras_16_strength": "FLOAT",
        "loras_16_enabled": "BOOLEAN",
        "loras_17_lora_name": "COMBO",
        "loras_17_strength": "FLOAT",
        "loras_17_enabled": "BOOLEAN",
        "loras_18_lora_name": "COMBO",
        "loras_18_strength": "FLOAT",
        "loras_18_enabled": "BOOLEAN",
        "loras_19_lora_name": "COMBO",
        "loras_19_strength": "FLOAT",
        "loras_19_enabled": "BOOLEAN",
    }
    DYNAMIC_INPUTS = {
        "loras": Group(
            rows=20,
            fields={
                "lora_name": ServerChoice(),
                "strength": Number(default=1.0, low=-100.0, high=100.0),
                "enabled": Flag(default=True),
            },
        )
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        loras_0_lora_name: str | tuple[str, int],
        loras_0_strength: float | tuple[str, int] | None = None,
        loras_0_enabled: bool | tuple[str, int] | None = None,
        loras_1_lora_name: str | tuple[str, int] | None = None,
        loras_1_strength: float | tuple[str, int] | None = None,
        loras_1_enabled: bool | tuple[str, int] | None = None,
        loras_2_lora_name: str | tuple[str, int] | None = None,
        loras_2_strength: float | tuple[str, int] | None = None,
        loras_2_enabled: bool | tuple[str, int] | None = None,
        loras_3_lora_name: str | tuple[str, int] | None = None,
        loras_3_strength: float | tuple[str, int] | None = None,
        loras_3_enabled: bool | tuple[str, int] | None = None,
        loras_4_lora_name: str | tuple[str, int] | None = None,
        loras_4_strength: float | tuple[str, int] | None = None,
        loras_4_enabled: bool | tuple[str, int] | None = None,
        loras_5_lora_name: str | tuple[str, int] | None = None,
        loras_5_strength: float | tuple[str, int] | None = None,
        loras_5_enabled: bool | tuple[str, int] | None = None,
        loras_6_lora_name: str | tuple[str, int] | None = None,
        loras_6_strength: float | tuple[str, int] | None = None,
        loras_6_enabled: bool | tuple[str, int] | None = None,
        loras_7_lora_name: str | tuple[str, int] | None = None,
        loras_7_strength: float | tuple[str, int] | None = None,
        loras_7_enabled: bool | tuple[str, int] | None = None,
        loras_8_lora_name: str | tuple[str, int] | None = None,
        loras_8_strength: float | tuple[str, int] | None = None,
        loras_8_enabled: bool | tuple[str, int] | None = None,
        loras_9_lora_name: str | tuple[str, int] | None = None,
        loras_9_strength: float | tuple[str, int] | None = None,
        loras_9_enabled: bool | tuple[str, int] | None = None,
        loras_10_lora_name: str | tuple[str, int] | None = None,
        loras_10_strength: float | tuple[str, int] | None = None,
        loras_10_enabled: bool | tuple[str, int] | None = None,
        loras_11_lora_name: str | tuple[str, int] | None = None,
        loras_11_strength: float | tuple[str, int] | None = None,
        loras_11_enabled: bool | tuple[str, int] | None = None,
        loras_12_lora_name: str | tuple[str, int] | None = None,
        loras_12_strength: float | tuple[str, int] | None = None,
        loras_12_enabled: bool | tuple[str, int] | None = None,
        loras_13_lora_name: str | tuple[str, int] | None = None,
        loras_13_strength: float | tuple[str, int] | None = None,
        loras_13_enabled: bool | tuple[str, int] | None = None,
        loras_14_lora_name: str | tuple[str, int] | None = None,
        loras_14_strength: float | tuple[str, int] | None = None,
        loras_14_enabled: bool | tuple[str, int] | None = None,
        loras_15_lora_name: str | tuple[str, int] | None = None,
        loras_15_strength: float | tuple[str, int] | None = None,
        loras_15_enabled: bool | tuple[str, int] | None = None,
        loras_16_lora_name: str | tuple[str, int] | None = None,
        loras_16_strength: float | tuple[str, int] | None = None,
        loras_16_enabled: bool | tuple[str, int] | None = None,
        loras_17_lora_name: str | tuple[str, int] | None = None,
        loras_17_strength: float | tuple[str, int] | None = None,
        loras_17_enabled: bool | tuple[str, int] | None = None,
        loras_18_lora_name: str | tuple[str, int] | None = None,
        loras_18_strength: float | tuple[str, int] | None = None,
        loras_18_enabled: bool | tuple[str, int] | None = None,
        loras_19_lora_name: str | tuple[str, int] | None = None,
        loras_19_strength: float | tuple[str, int] | None = None,
        loras_19_enabled: bool | tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.loras_0_lora_name = loras_0_lora_name
        if loras_0_strength is not None:
            self.loras_0_strength = loras_0_strength
        if loras_0_enabled is not None:
            self.loras_0_enabled = loras_0_enabled
        if loras_1_lora_name is not None:
            self.loras_1_lora_name = loras_1_lora_name
        if loras_1_strength is not None:
            self.loras_1_strength = loras_1_strength
        if loras_1_enabled is not None:
            self.loras_1_enabled = loras_1_enabled
        if loras_2_lora_name is not None:
            self.loras_2_lora_name = loras_2_lora_name
        if loras_2_strength is not None:
            self.loras_2_strength = loras_2_strength
        if loras_2_enabled is not None:
            self.loras_2_enabled = loras_2_enabled
        if loras_3_lora_name is not None:
            self.loras_3_lora_name = loras_3_lora_name
        if loras_3_strength is not None:
            self.loras_3_strength = loras_3_strength
        if loras_3_enabled is not None:
            self.loras_3_enabled = loras_3_enabled
        if loras_4_lora_name is not None:
            self.loras_4_lora_name = loras_4_lora_name
        if loras_4_strength is not None:
            self.loras_4_strength = loras_4_strength
        if loras_4_enabled is not None:
            self.loras_4_enabled = loras_4_enabled
        if loras_5_lora_name is not None:
            self.loras_5_lora_name = loras_5_lora_name
        if loras_5_strength is not None:
            self.loras_5_strength = loras_5_strength
        if loras_5_enabled is not None:
            self.loras_5_enabled = loras_5_enabled
        if loras_6_lora_name is not None:
            self.loras_6_lora_name = loras_6_lora_name
        if loras_6_strength is not None:
            self.loras_6_strength = loras_6_strength
        if loras_6_enabled is not None:
            self.loras_6_enabled = loras_6_enabled
        if loras_7_lora_name is not None:
            self.loras_7_lora_name = loras_7_lora_name
        if loras_7_strength is not None:
            self.loras_7_strength = loras_7_strength
        if loras_7_enabled is not None:
            self.loras_7_enabled = loras_7_enabled
        if loras_8_lora_name is not None:
            self.loras_8_lora_name = loras_8_lora_name
        if loras_8_strength is not None:
            self.loras_8_strength = loras_8_strength
        if loras_8_enabled is not None:
            self.loras_8_enabled = loras_8_enabled
        if loras_9_lora_name is not None:
            self.loras_9_lora_name = loras_9_lora_name
        if loras_9_strength is not None:
            self.loras_9_strength = loras_9_strength
        if loras_9_enabled is not None:
            self.loras_9_enabled = loras_9_enabled
        if loras_10_lora_name is not None:
            self.loras_10_lora_name = loras_10_lora_name
        if loras_10_strength is not None:
            self.loras_10_strength = loras_10_strength
        if loras_10_enabled is not None:
            self.loras_10_enabled = loras_10_enabled
        if loras_11_lora_name is not None:
            self.loras_11_lora_name = loras_11_lora_name
        if loras_11_strength is not None:
            self.loras_11_strength = loras_11_strength
        if loras_11_enabled is not None:
            self.loras_11_enabled = loras_11_enabled
        if loras_12_lora_name is not None:
            self.loras_12_lora_name = loras_12_lora_name
        if loras_12_strength is not None:
            self.loras_12_strength = loras_12_strength
        if loras_12_enabled is not None:
            self.loras_12_enabled = loras_12_enabled
        if loras_13_lora_name is not None:
            self.loras_13_lora_name = loras_13_lora_name
        if loras_13_strength is not None:
            self.loras_13_strength = loras_13_strength
        if loras_13_enabled is not None:
            self.loras_13_enabled = loras_13_enabled
        if loras_14_lora_name is not None:
            self.loras_14_lora_name = loras_14_lora_name
        if loras_14_strength is not None:
            self.loras_14_strength = loras_14_strength
        if loras_14_enabled is not None:
            self.loras_14_enabled = loras_14_enabled
        if loras_15_lora_name is not None:
            self.loras_15_lora_name = loras_15_lora_name
        if loras_15_strength is not None:
            self.loras_15_strength = loras_15_strength
        if loras_15_enabled is not None:
            self.loras_15_enabled = loras_15_enabled
        if loras_16_lora_name is not None:
            self.loras_16_lora_name = loras_16_lora_name
        if loras_16_strength is not None:
            self.loras_16_strength = loras_16_strength
        if loras_16_enabled is not None:
            self.loras_16_enabled = loras_16_enabled
        if loras_17_lora_name is not None:
            self.loras_17_lora_name = loras_17_lora_name
        if loras_17_strength is not None:
            self.loras_17_strength = loras_17_strength
        if loras_17_enabled is not None:
            self.loras_17_enabled = loras_17_enabled
        if loras_18_lora_name is not None:
            self.loras_18_lora_name = loras_18_lora_name
        if loras_18_strength is not None:
            self.loras_18_strength = loras_18_strength
        if loras_18_enabled is not None:
            self.loras_18_enabled = loras_18_enabled
        if loras_19_lora_name is not None:
            self.loras_19_lora_name = loras_19_lora_name
        if loras_19_strength is not None:
            self.loras_19_strength = loras_19_strength
        if loras_19_enabled is not None:
            self.loras_19_enabled = loras_19_enabled
        self.__validate()

    def __validate(self) -> None:
        validate_dynamic(self)


class LoadLoraTextEncoder(DynamicInputs, BaseNode):
    __slots__ = (
        "clip",
        "loras_0_lora_name",
        "loras_0_strength",
        "loras_0_enabled",
        "loras_1_lora_name",
        "loras_1_strength",
        "loras_1_enabled",
        "loras_2_lora_name",
        "loras_2_strength",
        "loras_2_enabled",
        "loras_3_lora_name",
        "loras_3_strength",
        "loras_3_enabled",
        "loras_4_lora_name",
        "loras_4_strength",
        "loras_4_enabled",
        "loras_5_lora_name",
        "loras_5_strength",
        "loras_5_enabled",
        "loras_6_lora_name",
        "loras_6_strength",
        "loras_6_enabled",
        "loras_7_lora_name",
        "loras_7_strength",
        "loras_7_enabled",
        "loras_8_lora_name",
        "loras_8_strength",
        "loras_8_enabled",
        "loras_9_lora_name",
        "loras_9_strength",
        "loras_9_enabled",
        "loras_10_lora_name",
        "loras_10_strength",
        "loras_10_enabled",
        "loras_11_lora_name",
        "loras_11_strength",
        "loras_11_enabled",
        "loras_12_lora_name",
        "loras_12_strength",
        "loras_12_enabled",
        "loras_13_lora_name",
        "loras_13_strength",
        "loras_13_enabled",
        "loras_14_lora_name",
        "loras_14_strength",
        "loras_14_enabled",
        "loras_15_lora_name",
        "loras_15_strength",
        "loras_15_enabled",
        "loras_16_lora_name",
        "loras_16_strength",
        "loras_16_enabled",
        "loras_17_lora_name",
        "loras_17_strength",
        "loras_17_enabled",
        "loras_18_lora_name",
        "loras_18_strength",
        "loras_18_enabled",
        "loras_19_lora_name",
        "loras_19_strength",
        "loras_19_enabled",
    )
    CATEGORY = "model/loaders"
    OUTPUTS = {"CLIP": 0}
    RETURN_TYPES = ("CLIP",)
    INPUT_NAMES = {
        "loras_0_lora_name": "loras.0.lora_name",
        "loras_0_strength": "loras.0.strength",
        "loras_0_enabled": "loras.0.enabled",
        "loras_1_lora_name": "loras.1.lora_name",
        "loras_1_strength": "loras.1.strength",
        "loras_1_enabled": "loras.1.enabled",
        "loras_2_lora_name": "loras.2.lora_name",
        "loras_2_strength": "loras.2.strength",
        "loras_2_enabled": "loras.2.enabled",
        "loras_3_lora_name": "loras.3.lora_name",
        "loras_3_strength": "loras.3.strength",
        "loras_3_enabled": "loras.3.enabled",
        "loras_4_lora_name": "loras.4.lora_name",
        "loras_4_strength": "loras.4.strength",
        "loras_4_enabled": "loras.4.enabled",
        "loras_5_lora_name": "loras.5.lora_name",
        "loras_5_strength": "loras.5.strength",
        "loras_5_enabled": "loras.5.enabled",
        "loras_6_lora_name": "loras.6.lora_name",
        "loras_6_strength": "loras.6.strength",
        "loras_6_enabled": "loras.6.enabled",
        "loras_7_lora_name": "loras.7.lora_name",
        "loras_7_strength": "loras.7.strength",
        "loras_7_enabled": "loras.7.enabled",
        "loras_8_lora_name": "loras.8.lora_name",
        "loras_8_strength": "loras.8.strength",
        "loras_8_enabled": "loras.8.enabled",
        "loras_9_lora_name": "loras.9.lora_name",
        "loras_9_strength": "loras.9.strength",
        "loras_9_enabled": "loras.9.enabled",
        "loras_10_lora_name": "loras.10.lora_name",
        "loras_10_strength": "loras.10.strength",
        "loras_10_enabled": "loras.10.enabled",
        "loras_11_lora_name": "loras.11.lora_name",
        "loras_11_strength": "loras.11.strength",
        "loras_11_enabled": "loras.11.enabled",
        "loras_12_lora_name": "loras.12.lora_name",
        "loras_12_strength": "loras.12.strength",
        "loras_12_enabled": "loras.12.enabled",
        "loras_13_lora_name": "loras.13.lora_name",
        "loras_13_strength": "loras.13.strength",
        "loras_13_enabled": "loras.13.enabled",
        "loras_14_lora_name": "loras.14.lora_name",
        "loras_14_strength": "loras.14.strength",
        "loras_14_enabled": "loras.14.enabled",
        "loras_15_lora_name": "loras.15.lora_name",
        "loras_15_strength": "loras.15.strength",
        "loras_15_enabled": "loras.15.enabled",
        "loras_16_lora_name": "loras.16.lora_name",
        "loras_16_strength": "loras.16.strength",
        "loras_16_enabled": "loras.16.enabled",
        "loras_17_lora_name": "loras.17.lora_name",
        "loras_17_strength": "loras.17.strength",
        "loras_17_enabled": "loras.17.enabled",
        "loras_18_lora_name": "loras.18.lora_name",
        "loras_18_strength": "loras.18.strength",
        "loras_18_enabled": "loras.18.enabled",
        "loras_19_lora_name": "loras.19.lora_name",
        "loras_19_strength": "loras.19.strength",
        "loras_19_enabled": "loras.19.enabled",
    }
    INPUT_SOCKET_TYPES = {
        "clip": "CLIP",
        "loras_0_lora_name": "COMBO",
        "loras_0_strength": "FLOAT",
        "loras_0_enabled": "BOOLEAN",
        "loras_1_lora_name": "COMBO",
        "loras_1_strength": "FLOAT",
        "loras_1_enabled": "BOOLEAN",
        "loras_2_lora_name": "COMBO",
        "loras_2_strength": "FLOAT",
        "loras_2_enabled": "BOOLEAN",
        "loras_3_lora_name": "COMBO",
        "loras_3_strength": "FLOAT",
        "loras_3_enabled": "BOOLEAN",
        "loras_4_lora_name": "COMBO",
        "loras_4_strength": "FLOAT",
        "loras_4_enabled": "BOOLEAN",
        "loras_5_lora_name": "COMBO",
        "loras_5_strength": "FLOAT",
        "loras_5_enabled": "BOOLEAN",
        "loras_6_lora_name": "COMBO",
        "loras_6_strength": "FLOAT",
        "loras_6_enabled": "BOOLEAN",
        "loras_7_lora_name": "COMBO",
        "loras_7_strength": "FLOAT",
        "loras_7_enabled": "BOOLEAN",
        "loras_8_lora_name": "COMBO",
        "loras_8_strength": "FLOAT",
        "loras_8_enabled": "BOOLEAN",
        "loras_9_lora_name": "COMBO",
        "loras_9_strength": "FLOAT",
        "loras_9_enabled": "BOOLEAN",
        "loras_10_lora_name": "COMBO",
        "loras_10_strength": "FLOAT",
        "loras_10_enabled": "BOOLEAN",
        "loras_11_lora_name": "COMBO",
        "loras_11_strength": "FLOAT",
        "loras_11_enabled": "BOOLEAN",
        "loras_12_lora_name": "COMBO",
        "loras_12_strength": "FLOAT",
        "loras_12_enabled": "BOOLEAN",
        "loras_13_lora_name": "COMBO",
        "loras_13_strength": "FLOAT",
        "loras_13_enabled": "BOOLEAN",
        "loras_14_lora_name": "COMBO",
        "loras_14_strength": "FLOAT",
        "loras_14_enabled": "BOOLEAN",
        "loras_15_lora_name": "COMBO",
        "loras_15_strength": "FLOAT",
        "loras_15_enabled": "BOOLEAN",
        "loras_16_lora_name": "COMBO",
        "loras_16_strength": "FLOAT",
        "loras_16_enabled": "BOOLEAN",
        "loras_17_lora_name": "COMBO",
        "loras_17_strength": "FLOAT",
        "loras_17_enabled": "BOOLEAN",
        "loras_18_lora_name": "COMBO",
        "loras_18_strength": "FLOAT",
        "loras_18_enabled": "BOOLEAN",
        "loras_19_lora_name": "COMBO",
        "loras_19_strength": "FLOAT",
        "loras_19_enabled": "BOOLEAN",
    }
    DYNAMIC_INPUTS = {
        "loras": Group(
            rows=20,
            fields={
                "lora_name": ServerChoice(),
                "strength": Number(default=1.0, low=-100.0, high=100.0),
                "enabled": Flag(default=True),
            },
        )
    }

    def __init__(
        self,
        node_id: str,
        clip: tuple[str, int],
        loras_0_lora_name: str | tuple[str, int],
        loras_0_strength: float | tuple[str, int] | None = None,
        loras_0_enabled: bool | tuple[str, int] | None = None,
        loras_1_lora_name: str | tuple[str, int] | None = None,
        loras_1_strength: float | tuple[str, int] | None = None,
        loras_1_enabled: bool | tuple[str, int] | None = None,
        loras_2_lora_name: str | tuple[str, int] | None = None,
        loras_2_strength: float | tuple[str, int] | None = None,
        loras_2_enabled: bool | tuple[str, int] | None = None,
        loras_3_lora_name: str | tuple[str, int] | None = None,
        loras_3_strength: float | tuple[str, int] | None = None,
        loras_3_enabled: bool | tuple[str, int] | None = None,
        loras_4_lora_name: str | tuple[str, int] | None = None,
        loras_4_strength: float | tuple[str, int] | None = None,
        loras_4_enabled: bool | tuple[str, int] | None = None,
        loras_5_lora_name: str | tuple[str, int] | None = None,
        loras_5_strength: float | tuple[str, int] | None = None,
        loras_5_enabled: bool | tuple[str, int] | None = None,
        loras_6_lora_name: str | tuple[str, int] | None = None,
        loras_6_strength: float | tuple[str, int] | None = None,
        loras_6_enabled: bool | tuple[str, int] | None = None,
        loras_7_lora_name: str | tuple[str, int] | None = None,
        loras_7_strength: float | tuple[str, int] | None = None,
        loras_7_enabled: bool | tuple[str, int] | None = None,
        loras_8_lora_name: str | tuple[str, int] | None = None,
        loras_8_strength: float | tuple[str, int] | None = None,
        loras_8_enabled: bool | tuple[str, int] | None = None,
        loras_9_lora_name: str | tuple[str, int] | None = None,
        loras_9_strength: float | tuple[str, int] | None = None,
        loras_9_enabled: bool | tuple[str, int] | None = None,
        loras_10_lora_name: str | tuple[str, int] | None = None,
        loras_10_strength: float | tuple[str, int] | None = None,
        loras_10_enabled: bool | tuple[str, int] | None = None,
        loras_11_lora_name: str | tuple[str, int] | None = None,
        loras_11_strength: float | tuple[str, int] | None = None,
        loras_11_enabled: bool | tuple[str, int] | None = None,
        loras_12_lora_name: str | tuple[str, int] | None = None,
        loras_12_strength: float | tuple[str, int] | None = None,
        loras_12_enabled: bool | tuple[str, int] | None = None,
        loras_13_lora_name: str | tuple[str, int] | None = None,
        loras_13_strength: float | tuple[str, int] | None = None,
        loras_13_enabled: bool | tuple[str, int] | None = None,
        loras_14_lora_name: str | tuple[str, int] | None = None,
        loras_14_strength: float | tuple[str, int] | None = None,
        loras_14_enabled: bool | tuple[str, int] | None = None,
        loras_15_lora_name: str | tuple[str, int] | None = None,
        loras_15_strength: float | tuple[str, int] | None = None,
        loras_15_enabled: bool | tuple[str, int] | None = None,
        loras_16_lora_name: str | tuple[str, int] | None = None,
        loras_16_strength: float | tuple[str, int] | None = None,
        loras_16_enabled: bool | tuple[str, int] | None = None,
        loras_17_lora_name: str | tuple[str, int] | None = None,
        loras_17_strength: float | tuple[str, int] | None = None,
        loras_17_enabled: bool | tuple[str, int] | None = None,
        loras_18_lora_name: str | tuple[str, int] | None = None,
        loras_18_strength: float | tuple[str, int] | None = None,
        loras_18_enabled: bool | tuple[str, int] | None = None,
        loras_19_lora_name: str | tuple[str, int] | None = None,
        loras_19_strength: float | tuple[str, int] | None = None,
        loras_19_enabled: bool | tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.clip = clip
        self.loras_0_lora_name = loras_0_lora_name
        if loras_0_strength is not None:
            self.loras_0_strength = loras_0_strength
        if loras_0_enabled is not None:
            self.loras_0_enabled = loras_0_enabled
        if loras_1_lora_name is not None:
            self.loras_1_lora_name = loras_1_lora_name
        if loras_1_strength is not None:
            self.loras_1_strength = loras_1_strength
        if loras_1_enabled is not None:
            self.loras_1_enabled = loras_1_enabled
        if loras_2_lora_name is not None:
            self.loras_2_lora_name = loras_2_lora_name
        if loras_2_strength is not None:
            self.loras_2_strength = loras_2_strength
        if loras_2_enabled is not None:
            self.loras_2_enabled = loras_2_enabled
        if loras_3_lora_name is not None:
            self.loras_3_lora_name = loras_3_lora_name
        if loras_3_strength is not None:
            self.loras_3_strength = loras_3_strength
        if loras_3_enabled is not None:
            self.loras_3_enabled = loras_3_enabled
        if loras_4_lora_name is not None:
            self.loras_4_lora_name = loras_4_lora_name
        if loras_4_strength is not None:
            self.loras_4_strength = loras_4_strength
        if loras_4_enabled is not None:
            self.loras_4_enabled = loras_4_enabled
        if loras_5_lora_name is not None:
            self.loras_5_lora_name = loras_5_lora_name
        if loras_5_strength is not None:
            self.loras_5_strength = loras_5_strength
        if loras_5_enabled is not None:
            self.loras_5_enabled = loras_5_enabled
        if loras_6_lora_name is not None:
            self.loras_6_lora_name = loras_6_lora_name
        if loras_6_strength is not None:
            self.loras_6_strength = loras_6_strength
        if loras_6_enabled is not None:
            self.loras_6_enabled = loras_6_enabled
        if loras_7_lora_name is not None:
            self.loras_7_lora_name = loras_7_lora_name
        if loras_7_strength is not None:
            self.loras_7_strength = loras_7_strength
        if loras_7_enabled is not None:
            self.loras_7_enabled = loras_7_enabled
        if loras_8_lora_name is not None:
            self.loras_8_lora_name = loras_8_lora_name
        if loras_8_strength is not None:
            self.loras_8_strength = loras_8_strength
        if loras_8_enabled is not None:
            self.loras_8_enabled = loras_8_enabled
        if loras_9_lora_name is not None:
            self.loras_9_lora_name = loras_9_lora_name
        if loras_9_strength is not None:
            self.loras_9_strength = loras_9_strength
        if loras_9_enabled is not None:
            self.loras_9_enabled = loras_9_enabled
        if loras_10_lora_name is not None:
            self.loras_10_lora_name = loras_10_lora_name
        if loras_10_strength is not None:
            self.loras_10_strength = loras_10_strength
        if loras_10_enabled is not None:
            self.loras_10_enabled = loras_10_enabled
        if loras_11_lora_name is not None:
            self.loras_11_lora_name = loras_11_lora_name
        if loras_11_strength is not None:
            self.loras_11_strength = loras_11_strength
        if loras_11_enabled is not None:
            self.loras_11_enabled = loras_11_enabled
        if loras_12_lora_name is not None:
            self.loras_12_lora_name = loras_12_lora_name
        if loras_12_strength is not None:
            self.loras_12_strength = loras_12_strength
        if loras_12_enabled is not None:
            self.loras_12_enabled = loras_12_enabled
        if loras_13_lora_name is not None:
            self.loras_13_lora_name = loras_13_lora_name
        if loras_13_strength is not None:
            self.loras_13_strength = loras_13_strength
        if loras_13_enabled is not None:
            self.loras_13_enabled = loras_13_enabled
        if loras_14_lora_name is not None:
            self.loras_14_lora_name = loras_14_lora_name
        if loras_14_strength is not None:
            self.loras_14_strength = loras_14_strength
        if loras_14_enabled is not None:
            self.loras_14_enabled = loras_14_enabled
        if loras_15_lora_name is not None:
            self.loras_15_lora_name = loras_15_lora_name
        if loras_15_strength is not None:
            self.loras_15_strength = loras_15_strength
        if loras_15_enabled is not None:
            self.loras_15_enabled = loras_15_enabled
        if loras_16_lora_name is not None:
            self.loras_16_lora_name = loras_16_lora_name
        if loras_16_strength is not None:
            self.loras_16_strength = loras_16_strength
        if loras_16_enabled is not None:
            self.loras_16_enabled = loras_16_enabled
        if loras_17_lora_name is not None:
            self.loras_17_lora_name = loras_17_lora_name
        if loras_17_strength is not None:
            self.loras_17_strength = loras_17_strength
        if loras_17_enabled is not None:
            self.loras_17_enabled = loras_17_enabled
        if loras_18_lora_name is not None:
            self.loras_18_lora_name = loras_18_lora_name
        if loras_18_strength is not None:
            self.loras_18_strength = loras_18_strength
        if loras_18_enabled is not None:
            self.loras_18_enabled = loras_18_enabled
        if loras_19_lora_name is not None:
            self.loras_19_lora_name = loras_19_lora_name
        if loras_19_strength is not None:
            self.loras_19_strength = loras_19_strength
        if loras_19_enabled is not None:
            self.loras_19_enabled = loras_19_enabled
        self.__validate()

    def __validate(self) -> None:
        validate_dynamic(self)
