from typing import Literal, get_args

from ...base import BaseNode
from ...validation import validate_choice

PixverseTemplateNode_Templates = Literal[
    "Microwave", "Suit Swagger", "Anything, Robot", "Subject 3 Fever", "kiss kiss"
]
PIXVERSETEMPLATENODE_TEMPLATES = frozenset(get_args(PixverseTemplateNode_Templates))


class PixverseTemplateNode(BaseNode):
    __slots__ = ("template",)
    CATEGORY = "partner/video/PixVerse"
    OUTPUTS = {"pixverse_template": 0}
    RETURN_TYPES = ("PIXVERSE_TEMPLATE",)
    INPUT_SOCKET_TYPES = {"template": "COMBO"}

    def __init__(
        self, node_id: str, template: PixverseTemplateNode_Templates | tuple[str, int]
    ) -> None:
        super().__init__(node_id=node_id)
        self.template = template
        self.__validate()

    def __validate(self) -> None:
        validate_choice("template", self.template, PIXVERSETEMPLATENODE_TEMPLATES)
