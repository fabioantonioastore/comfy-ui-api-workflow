from typing import Literal

from ...base import BaseNode
from ...dynamic import Combo, DynamicInputs, Text, validate_dynamic

FishAudioVoiceSelector_Voices = Literal[
    "Energetic Male (en)",
    "Friendly Women (en)",
    "Sarah (en)",
    "Verity (en)",
    "Polo (en)",
    "Adrian (en)",
    "E-girl (en)",
    "Narrator (en)",
    "Warm Conversational Voice (en)",
    "Warm Storyteller (en)",
    "Dramatic Character Male (en)",
    "Delia (en)",
    "Sheila (en)",
    "News Narrator (zh)",
    "Lively Female (zh)",
    "Gentle Female (zh)",
    "Energetic Female (ja)",
    "Calm Female (ja)",
    "Calm Male (ja)",
    "custom",
]


class FishAudioVoiceSelector(DynamicInputs, BaseNode):
    __slots__ = (
        "voice",
        "voice_voice_id",
    )
    CATEGORY = "partner/audio/Fish Audio"
    OUTPUTS = {"voice": 0}
    RETURN_TYPES = ("FISHAUDIO_VOICE",)
    INPUT_NAMES = {"voice_voice_id": "voice.voice_id"}
    INPUT_SOCKET_TYPES = {"voice": "COMFY_DYNAMICCOMBO_V3", "voice_voice_id": "STRING"}
    DYNAMIC_INPUTS = {
        "voice": Combo(
            {
                "Energetic Male (en)": {},
                "Friendly Women (en)": {},
                "Sarah (en)": {},
                "Verity (en)": {},
                "Polo (en)": {},
                "Adrian (en)": {},
                "E-girl (en)": {},
                "Narrator (en)": {},
                "Warm Conversational Voice (en)": {},
                "Warm Storyteller (en)": {},
                "Dramatic Character Male (en)": {},
                "Delia (en)": {},
                "Sheila (en)": {},
                "News Narrator (zh)": {},
                "Lively Female (zh)": {},
                "Gentle Female (zh)": {},
                "Energetic Female (ja)": {},
                "Calm Female (ja)": {},
                "Calm Male (ja)": {},
                "custom": {"voice_voice_id": Text(default="")},
            }
        )
    }

    def __init__(
        self,
        node_id: str,
        voice: FishAudioVoiceSelector_Voices = "Energetic Male (en)",
        voice_voice_id: str | tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.voice = voice
        if voice_voice_id is not None:
            self.voice_voice_id = voice_voice_id
        self.__validate()

    def __validate(self) -> None:
        validate_dynamic(self)
