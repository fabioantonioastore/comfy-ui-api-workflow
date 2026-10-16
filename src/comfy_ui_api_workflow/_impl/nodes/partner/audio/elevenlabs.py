from typing import Literal, get_args

from ...base import BaseNode
from ...validation import validate_choice

ElevenLabsVoiceSelector_Voices = Literal[
    "Roger (male, american)",
    "Sarah (female, american)",
    "Laura (female, american)",
    "Charlie (male, australian)",
    "George (male, british)",
    "Callum (male, american)",
    "River (neutral, american)",
    "Harry (male, american)",
    "Liam (male, american)",
    "Alice (female, british)",
    "Matilda (female, american)",
    "Will (male, american)",
    "Jessica (female, american)",
    "Eric (male, american)",
    "Bella (female, american)",
    "Chris (male, american)",
    "Brian (male, american)",
    "Daniel (male, british)",
    "Lily (female, british)",
    "Adam (male, american)",
    "Bill (male, american)",
]
ELEVENLABSVOICESELECTOR_VOICES = frozenset(get_args(ElevenLabsVoiceSelector_Voices))


class ElevenLabsVoiceSelector(BaseNode):
    __slots__ = ("voice",)
    CATEGORY = "partner/audio/ElevenLabs"
    OUTPUTS = {"voice": 0}
    RETURN_TYPES = ("ELEVENLABS_VOICE",)
    INPUT_SOCKET_TYPES = {"voice": "COMBO"}

    def __init__(
        self, node_id: str, voice: ElevenLabsVoiceSelector_Voices | tuple[str, int]
    ) -> None:
        super().__init__(node_id=node_id)
        self.voice = voice
        self.__validate()

    def __validate(self) -> None:
        validate_choice("voice", self.voice, ELEVENLABSVOICESELECTOR_VOICES)
