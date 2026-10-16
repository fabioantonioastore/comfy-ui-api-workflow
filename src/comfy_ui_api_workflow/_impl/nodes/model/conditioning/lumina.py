from typing import Literal, get_args

from ...base import BaseNode
from ...validation import validate_choice

CLIPTextEncodeLumina2_System_Prompts = Literal["superior", "alignment"]
CLIPTEXTENCODELUMINA2_SYSTEM_PROMPTS = frozenset(
    get_args(CLIPTextEncodeLumina2_System_Prompts)
)


class CLIPTextEncodeLumina2(BaseNode):
    __slots__ = ("system_prompt", "user_prompt", "clip")
    CATEGORY = "model/conditioning/lumina"
    OUTPUTS = {"CONDITIONING": 0}
    RETURN_TYPES = ("CONDITIONING",)
    INPUT_SOCKET_TYPES = {
        "system_prompt": "COMBO",
        "user_prompt": "STRING",
        "clip": "CLIP",
    }

    def __init__(
        self,
        node_id: str,
        system_prompt: CLIPTextEncodeLumina2_System_Prompts | tuple[str, int],
        user_prompt: str | tuple[str, int],
        clip: tuple[str, int],
    ) -> None:
        super().__init__(node_id=node_id)
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.clip = clip
        self.__validate()

    def __validate(self) -> None:
        validate_choice(
            "system_prompt", self.system_prompt, CLIPTEXTENCODELUMINA2_SYSTEM_PROMPTS
        )
