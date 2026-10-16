from ...base import BaseNode
from ...validation import validate_not_empty


class GeminiInputFiles(BaseNode):
    __slots__ = ("file", "GEMINI_INPUT_FILES")
    CATEGORY = "partner/text/Gemini"
    OUTPUTS = {"GEMINI_INPUT_FILES": 0}
    RETURN_TYPES = ("GEMINI_INPUT_FILES",)
    INPUT_SOCKET_TYPES = {"file": "COMBO", "GEMINI_INPUT_FILES": "GEMINI_INPUT_FILES"}

    def __init__(
        self,
        node_id: str,
        file: str | tuple[str, int],
        GEMINI_INPUT_FILES: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.file = file
        if GEMINI_INPUT_FILES is not None:
            self.GEMINI_INPUT_FILES = GEMINI_INPUT_FILES
        self.__validate()

    def __validate(self) -> None:
        validate_not_empty("file", self.file)
