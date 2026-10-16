from typing import Literal, get_args

from ...base import BaseNode
from ...validation import validate_choice, validate_not_empty, validate_range


class OpenAIInputFiles(BaseNode):
    __slots__ = ("file", "OPENAI_INPUT_FILES")
    CATEGORY = "partner/text/OpenAI"
    OUTPUTS = {"OPENAI_INPUT_FILES": 0}
    RETURN_TYPES = ("OPENAI_INPUT_FILES",)
    INPUT_SOCKET_TYPES = {"file": "COMBO", "OPENAI_INPUT_FILES": "OPENAI_INPUT_FILES"}

    def __init__(
        self,
        node_id: str,
        file: str | tuple[str, int],
        OPENAI_INPUT_FILES: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.file = file
        if OPENAI_INPUT_FILES is not None:
            self.OPENAI_INPUT_FILES = OPENAI_INPUT_FILES
        self.__validate()

    def __validate(self) -> None:
        validate_not_empty("file", self.file)


OpenAIChatConfig_Truncations = Literal["auto", "disabled"]
OpenAIChatConfig_Reasoning_Efforts = Literal[
    "default", "none", "minimal", "low", "medium", "high", "xhigh", "max"
]
OPENAICHATCONFIG_TRUNCATIONS = frozenset(get_args(OpenAIChatConfig_Truncations))
OPENAICHATCONFIG_REASONING_EFFORTS = frozenset(
    get_args(OpenAIChatConfig_Reasoning_Efforts)
)


class OpenAIChatConfig(BaseNode):
    __slots__ = ("truncation", "max_output_tokens", "instructions", "reasoning_effort")
    CATEGORY = "partner/text/OpenAI"
    OUTPUTS = {"OPENAI_CHAT_CONFIG": 0}
    RETURN_TYPES = ("OPENAI_CHAT_CONFIG",)
    INPUT_SOCKET_TYPES = {
        "truncation": "COMBO",
        "max_output_tokens": "INT",
        "instructions": "STRING",
        "reasoning_effort": "COMBO",
    }

    def __init__(
        self,
        node_id: str,
        truncation: OpenAIChatConfig_Truncations | tuple[str, int] = "auto",
        max_output_tokens: int | tuple[str, int] = 4096,
        instructions: str | tuple[str, int] | None = None,
        reasoning_effort: (
            OpenAIChatConfig_Reasoning_Efforts | tuple[str, int]
        ) = "default",
    ) -> None:
        super().__init__(node_id=node_id)
        self.truncation = truncation
        self.max_output_tokens = max_output_tokens
        if instructions is not None:
            self.instructions = instructions
        self.reasoning_effort = reasoning_effort
        self.__validate()

    def __validate(self) -> None:
        validate_choice("truncation", self.truncation, OPENAICHATCONFIG_TRUNCATIONS)
        validate_range("max_output_tokens", self.max_output_tokens, 16, 16_384)
        validate_choice(
            "reasoning_effort",
            self.reasoning_effort,
            OPENAICHATCONFIG_REASONING_EFFORTS,
        )
