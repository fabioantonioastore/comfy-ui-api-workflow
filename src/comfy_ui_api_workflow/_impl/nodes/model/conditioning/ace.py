from typing import Literal, get_args

from ...base import BaseNode
from ...validation import validate_bool, validate_choice, validate_range


class TextEncodeAceStepAudio(BaseNode):
    __slots__ = ("clip", "tags", "lyrics", "lyrics_strength")
    CATEGORY = "model/conditioning/ace"
    OUTPUTS = {"CONDITIONING": 0}
    RETURN_TYPES = ("CONDITIONING",)
    INPUT_SOCKET_TYPES = {
        "clip": "CLIP",
        "tags": "STRING",
        "lyrics": "STRING",
        "lyrics_strength": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        clip: tuple[str, int],
        tags: str | tuple[str, int],
        lyrics: str | tuple[str, int],
        lyrics_strength: float | tuple[str, int] = 1.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.clip = clip
        self.tags = tags
        self.lyrics = lyrics
        self.lyrics_strength = lyrics_strength
        self.__validate()

    def __validate(self) -> None:
        validate_range("lyrics_strength", self.lyrics_strength, 0.0, 10.0)


TextEncodeAceStepAudio1_5_Timesignatures = Literal["2", "3", "4", "6"]
TextEncodeAceStepAudio1_5_Keyscales = Literal[
    "C major",
    "C# major",
    "Db major",
    "D major",
    "D# major",
    "Eb major",
    "E major",
    "F major",
    "F# major",
    "Gb major",
    "G major",
    "G# major",
    "Ab major",
    "A major",
    "A# major",
    "Bb major",
    "B major",
    "C minor",
    "C# minor",
    "Db minor",
    "D minor",
    "D# minor",
    "Eb minor",
    "E minor",
    "F minor",
    "F# minor",
    "Gb minor",
    "G minor",
    "G# minor",
    "Ab minor",
    "A minor",
    "A# minor",
    "Bb minor",
    "B minor",
]
TextEncodeAceStepAudio1_5_Languages = Literal[
    "ar",
    "az",
    "bg",
    "bn",
    "ca",
    "cs",
    "da",
    "de",
    "el",
    "en",
    "es",
    "fa",
    "fi",
    "fr",
    "he",
    "hi",
    "hr",
    "ht",
    "hu",
    "id",
    "is",
    "it",
    "ja",
    "ko",
    "la",
    "lt",
    "ms",
    "ne",
    "nl",
    "no",
    "pa",
    "pl",
    "pt",
    "ro",
    "ru",
    "sa",
    "sk",
    "sr",
    "sv",
    "sw",
    "ta",
    "te",
    "th",
    "tl",
    "tr",
    "uk",
    "ur",
    "vi",
    "yue",
    "zh",
    "unknown",
]
TEXTENCODEACESTEPAUDIO1_5_TIMESIGNATURES = frozenset(
    get_args(TextEncodeAceStepAudio1_5_Timesignatures)
)
TEXTENCODEACESTEPAUDIO1_5_KEYSCALES = frozenset(
    get_args(TextEncodeAceStepAudio1_5_Keyscales)
)
TEXTENCODEACESTEPAUDIO1_5_LANGUAGES = frozenset(
    get_args(TextEncodeAceStepAudio1_5_Languages)
)


class TextEncodeAceStepAudio1_5(BaseNode):
    __slots__ = (
        "clip",
        "tags",
        "lyrics",
        "timesignature",
        "keyscale",
        "seed",
        "bpm",
        "duration",
        "language",
        "generate_audio_codes",
        "cfg_scale",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
    )
    CATEGORY = "model/conditioning/ace"
    CLASS_TYPE = "TextEncodeAceStepAudio1.5"
    OUTPUTS = {"CONDITIONING": 0}
    RETURN_TYPES = ("CONDITIONING",)
    INPUT_SOCKET_TYPES = {
        "clip": "CLIP",
        "tags": "STRING",
        "lyrics": "STRING",
        "seed": "INT",
        "bpm": "INT",
        "duration": "FLOAT",
        "timesignature": "COMBO",
        "language": "COMBO",
        "keyscale": "COMBO",
        "generate_audio_codes": "BOOLEAN",
        "cfg_scale": "FLOAT",
        "temperature": "FLOAT",
        "top_p": "FLOAT",
        "top_k": "INT",
        "min_p": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        clip: tuple[str, int],
        tags: str | tuple[str, int],
        lyrics: str | tuple[str, int],
        timesignature: TextEncodeAceStepAudio1_5_Timesignatures | tuple[str, int],
        keyscale: TextEncodeAceStepAudio1_5_Keyscales | tuple[str, int],
        seed: int | tuple[str, int] = 0,
        bpm: int | tuple[str, int] = 120,
        duration: float | tuple[str, int] = 120.0,
        language: TextEncodeAceStepAudio1_5_Languages | tuple[str, int] = "en",
        generate_audio_codes: bool | tuple[str, int] = True,
        cfg_scale: float | tuple[str, int] = 2.0,
        temperature: float | tuple[str, int] = 0.85,
        top_p: float | tuple[str, int] = 0.9,
        top_k: int | tuple[str, int] = 0,
        min_p: float | tuple[str, int] = 0.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.clip = clip
        self.tags = tags
        self.lyrics = lyrics
        self.timesignature = timesignature
        self.keyscale = keyscale
        self.seed = seed
        self.bpm = bpm
        self.duration = duration
        self.language = language
        self.generate_audio_codes = generate_audio_codes
        self.cfg_scale = cfg_scale
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.__validate()

    def __validate(self) -> None:
        validate_choice(
            "timesignature",
            self.timesignature,
            TEXTENCODEACESTEPAUDIO1_5_TIMESIGNATURES,
        )
        validate_choice("keyscale", self.keyscale, TEXTENCODEACESTEPAUDIO1_5_KEYSCALES)
        validate_range("seed", self.seed, 0, 18_446_744_073_709_551_615)
        validate_range("bpm", self.bpm, 10, 300)
        validate_range("duration", self.duration, 0.0, 2000.0)
        validate_choice("language", self.language, TEXTENCODEACESTEPAUDIO1_5_LANGUAGES)
        validate_bool("generate_audio_codes", self.generate_audio_codes)
        validate_range("cfg_scale", self.cfg_scale, 0.0, 100.0)
        validate_range("temperature", self.temperature, 0.0, 2.0)
        validate_range("top_p", self.top_p, 0.0, 2000.0)
        validate_range("top_k", self.top_k, 0, 100)
        validate_range("min_p", self.min_p, 0.0, 1.0)
