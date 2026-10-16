from ...base import BaseNode
from ...validation import validate_bool


class SetClipHooks(BaseNode):
    __slots__ = ("clip", "apply_to_conds", "schedule_clip", "hooks")
    CATEGORY = "advanced/hooks/clip"
    OUTPUTS = {"CLIP": 0}
    RETURN_TYPES = ("CLIP",)
    INPUT_SOCKET_TYPES = {
        "clip": "CLIP",
        "apply_to_conds": "BOOLEAN",
        "schedule_clip": "BOOLEAN",
        "hooks": "HOOKS",
    }

    def __init__(
        self,
        node_id: str,
        clip: tuple[str, int],
        apply_to_conds: bool | tuple[str, int] = True,
        schedule_clip: bool | tuple[str, int] = False,
        hooks: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.clip = clip
        self.apply_to_conds = apply_to_conds
        self.schedule_clip = schedule_clip
        if hooks is not None:
            self.hooks = hooks
        self.__validate()

    def __validate(self) -> None:
        validate_bool("apply_to_conds", self.apply_to_conds)
        validate_bool("schedule_clip", self.schedule_clip)
