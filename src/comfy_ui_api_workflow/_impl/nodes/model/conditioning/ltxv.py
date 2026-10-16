from ...base import BaseNode
from ...validation import validate_bool, validate_range


class LTXVImgToVideo(BaseNode):
    __slots__ = (
        "positive",
        "negative",
        "vae",
        "image",
        "width",
        "height",
        "length",
        "batch_size",
        "strength",
    )
    CATEGORY = "model/conditioning/ltxv"
    OUTPUTS = {"positive": 0, "negative": 1, "latent": 2}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    INPUT_SOCKET_TYPES = {
        "positive": "CONDITIONING",
        "negative": "CONDITIONING",
        "vae": "VAE",
        "image": "IMAGE",
        "width": "INT",
        "height": "INT",
        "length": "INT",
        "batch_size": "INT",
        "strength": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        positive: tuple[str, int],
        negative: tuple[str, int],
        vae: tuple[str, int],
        image: tuple[str, int],
        width: int | tuple[str, int] = 768,
        height: int | tuple[str, int] = 512,
        length: int | tuple[str, int] = 97,
        batch_size: int | tuple[str, int] = 1,
        strength: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.positive = positive
        self.negative = negative
        self.vae = vae
        self.image = image
        self.width = width
        self.height = height
        self.length = length
        self.batch_size = batch_size
        self.strength = strength
        self.__validate()

    def __validate(self) -> None:
        validate_range("width", self.width, 64, 16_384)
        validate_range("height", self.height, 64, 16_384)
        validate_range("length", self.length, 9, 16_384)
        validate_range("batch_size", self.batch_size, 1, 4096)
        validate_range("strength", self.strength, 0.0, 1.0)


class LTXVImgToVideoInplace(BaseNode):
    __slots__ = ("vae", "image", "latent", "strength", "bypass")
    CATEGORY = "model/conditioning/ltxv"
    OUTPUTS = {"latent": 0}
    RETURN_TYPES = ("LATENT",)
    INPUT_SOCKET_TYPES = {
        "vae": "VAE",
        "image": "IMAGE",
        "latent": "LATENT",
        "strength": "FLOAT",
        "bypass": "BOOLEAN",
    }

    def __init__(
        self,
        node_id: str,
        vae: tuple[str, int],
        image: tuple[str, int],
        latent: tuple[str, int],
        strength: float | tuple[str, int] = 1.0,
        bypass: bool | tuple[str, int] = False,
    ) -> None:
        super().__init__(node_id=node_id)
        self.vae = vae
        self.image = image
        self.latent = latent
        self.strength = strength
        self.bypass = bypass
        self.__validate()

    def __validate(self) -> None:
        validate_range("strength", self.strength, 0.0, 1.0)
        validate_bool("bypass", self.bypass)


class LTXVConditioning(BaseNode):
    __slots__ = ("positive", "negative", "frame_rate")
    CATEGORY = "model/conditioning/ltxv"
    OUTPUTS = {"positive": 0, "negative": 1}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    INPUT_SOCKET_TYPES = {
        "positive": "CONDITIONING",
        "negative": "CONDITIONING",
        "frame_rate": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        positive: tuple[str, int],
        negative: tuple[str, int],
        frame_rate: float | tuple[str, int] = 25.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.positive = positive
        self.negative = negative
        self.frame_rate = frame_rate
        self.__validate()

    def __validate(self) -> None:
        validate_range("frame_rate", self.frame_rate, 0.0, 1000.0)


class GetICLoRAParameters(BaseNode):
    __slots__ = ("iclora_model",)
    CATEGORY = "model/conditioning/ltxv"
    OUTPUTS = {"iclora_parameters": 0}
    RETURN_TYPES = ("IC_LORA_PARAMETERS",)
    INPUT_SOCKET_TYPES = {"iclora_model": "MODEL"}

    def __init__(self, node_id: str, iclora_model: tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.iclora_model = iclora_model


class LTXVAddGuide(BaseNode):
    __slots__ = (
        "positive",
        "negative",
        "vae",
        "latent",
        "image",
        "frame_idx",
        "strength",
        "attention_mask",
        "iclora_parameters",
    )
    CATEGORY = "model/conditioning/ltxv"
    OUTPUTS = {"positive": 0, "negative": 1, "latent": 2}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    INPUT_SOCKET_TYPES = {
        "positive": "CONDITIONING",
        "negative": "CONDITIONING",
        "vae": "VAE",
        "latent": "LATENT",
        "image": "IMAGE",
        "frame_idx": "INT",
        "strength": "FLOAT",
        "attention_mask": "MASK",
        "iclora_parameters": "IC_LORA_PARAMETERS",
    }

    def __init__(
        self,
        node_id: str,
        positive: tuple[str, int],
        negative: tuple[str, int],
        vae: tuple[str, int],
        latent: tuple[str, int],
        image: tuple[str, int],
        frame_idx: int | tuple[str, int] = 0,
        strength: float | tuple[str, int] = 1.0,
        attention_mask: tuple[str, int] | None = None,
        iclora_parameters: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.positive = positive
        self.negative = negative
        self.vae = vae
        self.latent = latent
        self.image = image
        self.frame_idx = frame_idx
        self.strength = strength
        if attention_mask is not None:
            self.attention_mask = attention_mask
        if iclora_parameters is not None:
            self.iclora_parameters = iclora_parameters
        self.__validate()

    def __validate(self) -> None:
        validate_range("frame_idx", self.frame_idx, -9999, 9999)
        validate_range("strength", self.strength, 0.0, 10.0)


class LTXVAddLatentGuide(BaseNode):
    __slots__ = (
        "positive",
        "negative",
        "vae",
        "latent",
        "guiding_latent",
        "latent_idx",
        "strength",
        "attention_mask",
    )
    CATEGORY = "model/conditioning/ltxv"
    OUTPUTS = {"positive": 0, "negative": 1, "latent": 2}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    INPUT_SOCKET_TYPES = {
        "positive": "CONDITIONING",
        "negative": "CONDITIONING",
        "vae": "VAE",
        "latent": "LATENT",
        "guiding_latent": "LATENT",
        "latent_idx": "INT",
        "strength": "FLOAT",
        "attention_mask": "MASK",
    }

    def __init__(
        self,
        node_id: str,
        positive: tuple[str, int],
        negative: tuple[str, int],
        vae: tuple[str, int],
        latent: tuple[str, int],
        guiding_latent: tuple[str, int],
        latent_idx: int | tuple[str, int] = 0,
        strength: float | tuple[str, int] = 1.0,
        attention_mask: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.positive = positive
        self.negative = negative
        self.vae = vae
        self.latent = latent
        self.guiding_latent = guiding_latent
        self.latent_idx = latent_idx
        self.strength = strength
        if attention_mask is not None:
            self.attention_mask = attention_mask
        self.__validate()

    def __validate(self) -> None:
        validate_range("latent_idx", self.latent_idx, -9999, 9999)
        validate_range("strength", self.strength, 0.0, 1.0)


class LTXVCropGuides(BaseNode):
    __slots__ = ("positive", "negative", "latent")
    CATEGORY = "model/conditioning/ltxv"
    OUTPUTS = {"positive": 0, "negative": 1, "latent": 2}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    INPUT_SOCKET_TYPES = {
        "positive": "CONDITIONING",
        "negative": "CONDITIONING",
        "latent": "LATENT",
    }

    def __init__(
        self,
        node_id: str,
        positive: tuple[str, int],
        negative: tuple[str, int],
        latent: tuple[str, int],
    ) -> None:
        super().__init__(node_id=node_id)
        self.positive = positive
        self.negative = negative
        self.latent = latent


class LTXVReferenceAudio(BaseNode):
    __slots__ = (
        "model",
        "positive",
        "negative",
        "reference_audio",
        "audio_vae",
        "identity_guidance_scale",
        "start_percent",
        "end_percent",
    )
    CATEGORY = "model/conditioning/ltxv"
    OUTPUTS = {"MODEL": 0, "positive": 1, "negative": 2}
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING")
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "positive": "CONDITIONING",
        "negative": "CONDITIONING",
        "reference_audio": "AUDIO",
        "audio_vae": "VAE",
        "identity_guidance_scale": "FLOAT",
        "start_percent": "FLOAT",
        "end_percent": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        positive: tuple[str, int],
        negative: tuple[str, int],
        reference_audio: tuple[str, int],
        audio_vae: tuple[str, int],
        identity_guidance_scale: float | tuple[str, int] = 3.0,
        start_percent: float | tuple[str, int] = 0.0,
        end_percent: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.positive = positive
        self.negative = negative
        self.reference_audio = reference_audio
        self.audio_vae = audio_vae
        self.identity_guidance_scale = identity_guidance_scale
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.__validate()

    def __validate(self) -> None:
        validate_range(
            "identity_guidance_scale", self.identity_guidance_scale, 0.0, 100.0
        )
        validate_range("start_percent", self.start_percent, 0.0, 1.0)
        validate_range("end_percent", self.end_percent, 0.0, 1.0)


class LTXVDurationPredictor(BaseNode):
    __slots__ = (
        "model",
        "positive",
        "duration_head",
        "frame_rate",
        "min_seconds",
        "max_seconds",
    )
    CATEGORY = "model/conditioning/ltxv"
    OUTPUTS = {"num_frames": 0, "seconds": 1}
    RETURN_TYPES = ("INT", "FLOAT")
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "positive": "CONDITIONING",
        "duration_head": "MODEL_PATCH",
        "frame_rate": "FLOAT",
        "min_seconds": "FLOAT",
        "max_seconds": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        positive: tuple[str, int],
        duration_head: tuple[str, int],
        frame_rate: float | tuple[str, int] = 24.0,
        min_seconds: float | tuple[str, int] = 1.0,
        max_seconds: float | tuple[str, int] = 20.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.positive = positive
        self.duration_head = duration_head
        self.frame_rate = frame_rate
        self.min_seconds = min_seconds
        self.max_seconds = max_seconds
        self.__validate()

    def __validate(self) -> None:
        validate_range("frame_rate", self.frame_rate, 1.0, 120.0)
        validate_range("min_seconds", self.min_seconds, 0.5, 120.0)
        validate_range("max_seconds", self.max_seconds, 0.5, 120.0)


class LTXVAddGeneratedKeyframes(BaseNode):
    __slots__ = (
        "positive",
        "negative",
        "vae",
        "latent",
        "interval_frames",
        "keyframes",
        "frame_indices",
    )
    CATEGORY = "model/conditioning/ltxv"
    OUTPUTS = {"positive": 0, "negative": 1, "latent": 2}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    INPUT_SOCKET_TYPES = {
        "positive": "CONDITIONING",
        "negative": "CONDITIONING",
        "vae": "VAE",
        "latent": "LATENT",
        "interval_frames": "INT",
        "keyframes": "LATENT",
        "frame_indices": "STRING",
    }

    def __init__(
        self,
        node_id: str,
        positive: tuple[str, int],
        negative: tuple[str, int],
        vae: tuple[str, int],
        latent: tuple[str, int],
        interval_frames: int | tuple[str, int] = 24,
        keyframes: tuple[str, int] | None = None,
        frame_indices: str | tuple[str, int] = "",
    ) -> None:
        super().__init__(node_id=node_id)
        self.positive = positive
        self.negative = negative
        self.vae = vae
        self.latent = latent
        self.interval_frames = interval_frames
        if keyframes is not None:
            self.keyframes = keyframes
        self.frame_indices = frame_indices
        self.__validate()

    def __validate(self) -> None:
        validate_range("interval_frames", self.interval_frames, 1, 1024)


class LTXVSeparateGeneratedKeyframes(BaseNode):
    __slots__ = ("positive", "negative", "latent", "keyframes_to_batch")
    CATEGORY = "model/conditioning/ltxv"
    OUTPUTS = {"positive": 0, "negative": 1, "latent": 2, "keyframes": 3}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "LATENT")
    INPUT_SOCKET_TYPES = {
        "positive": "CONDITIONING",
        "negative": "CONDITIONING",
        "latent": "LATENT",
        "keyframes_to_batch": "BOOLEAN",
    }

    def __init__(
        self,
        node_id: str,
        positive: tuple[str, int],
        negative: tuple[str, int],
        latent: tuple[str, int],
        keyframes_to_batch: bool | tuple[str, int] = False,
    ) -> None:
        super().__init__(node_id=node_id)
        self.positive = positive
        self.negative = negative
        self.latent = latent
        self.keyframes_to_batch = keyframes_to_batch
        self.__validate()

    def __validate(self) -> None:
        validate_bool("keyframes_to_batch", self.keyframes_to_batch)


class LTXVGeneratedKeyframesToGuides(BaseNode):
    __slots__ = (
        "positive",
        "negative",
        "vae",
        "latent",
        "keyframes",
        "strength",
        "override_frame_indices",
    )
    CATEGORY = "model/conditioning/ltxv"
    OUTPUTS = {"positive": 0, "negative": 1, "latent": 2}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    INPUT_SOCKET_TYPES = {
        "positive": "CONDITIONING",
        "negative": "CONDITIONING",
        "vae": "VAE",
        "latent": "LATENT",
        "keyframes": "LATENT",
        "strength": "FLOAT",
        "override_frame_indices": "STRING",
    }

    def __init__(
        self,
        node_id: str,
        positive: tuple[str, int],
        negative: tuple[str, int],
        vae: tuple[str, int],
        latent: tuple[str, int],
        keyframes: tuple[str, int],
        strength: float | tuple[str, int] = 1.0,
        override_frame_indices: str | tuple[str, int] = "",
    ) -> None:
        super().__init__(node_id=node_id)
        self.positive = positive
        self.negative = negative
        self.vae = vae
        self.latent = latent
        self.keyframes = keyframes
        self.strength = strength
        self.override_frame_indices = override_frame_indices
        self.__validate()

    def __validate(self) -> None:
        validate_range("strength", self.strength, 0.0, 10.0)
