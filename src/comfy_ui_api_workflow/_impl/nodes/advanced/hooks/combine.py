from ...base import BaseNode


class CombineHooks2(BaseNode):
    __slots__ = ("hooks_A", "hooks_B")
    CATEGORY = "advanced/hooks/combine"
    OUTPUTS = {"HOOKS": 0}
    RETURN_TYPES = ("HOOKS",)
    INPUT_SOCKET_TYPES = {"hooks_A": "HOOKS", "hooks_B": "HOOKS"}

    def __init__(
        self,
        node_id: str,
        hooks_A: tuple[str, int] | None = None,
        hooks_B: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        if hooks_A is not None:
            self.hooks_A = hooks_A
        if hooks_B is not None:
            self.hooks_B = hooks_B


class CombineHooks4(BaseNode):
    __slots__ = ("hooks_A", "hooks_B", "hooks_C", "hooks_D")
    CATEGORY = "advanced/hooks/combine"
    OUTPUTS = {"HOOKS": 0}
    RETURN_TYPES = ("HOOKS",)
    INPUT_SOCKET_TYPES = {
        "hooks_A": "HOOKS",
        "hooks_B": "HOOKS",
        "hooks_C": "HOOKS",
        "hooks_D": "HOOKS",
    }

    def __init__(
        self,
        node_id: str,
        hooks_A: tuple[str, int] | None = None,
        hooks_B: tuple[str, int] | None = None,
        hooks_C: tuple[str, int] | None = None,
        hooks_D: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        if hooks_A is not None:
            self.hooks_A = hooks_A
        if hooks_B is not None:
            self.hooks_B = hooks_B
        if hooks_C is not None:
            self.hooks_C = hooks_C
        if hooks_D is not None:
            self.hooks_D = hooks_D


class CombineHooks8(BaseNode):
    __slots__ = (
        "hooks_A",
        "hooks_B",
        "hooks_C",
        "hooks_D",
        "hooks_E",
        "hooks_F",
        "hooks_G",
        "hooks_H",
    )
    CATEGORY = "advanced/hooks/combine"
    OUTPUTS = {"HOOKS": 0}
    RETURN_TYPES = ("HOOKS",)
    INPUT_SOCKET_TYPES = {
        "hooks_A": "HOOKS",
        "hooks_B": "HOOKS",
        "hooks_C": "HOOKS",
        "hooks_D": "HOOKS",
        "hooks_E": "HOOKS",
        "hooks_F": "HOOKS",
        "hooks_G": "HOOKS",
        "hooks_H": "HOOKS",
    }

    def __init__(
        self,
        node_id: str,
        hooks_A: tuple[str, int] | None = None,
        hooks_B: tuple[str, int] | None = None,
        hooks_C: tuple[str, int] | None = None,
        hooks_D: tuple[str, int] | None = None,
        hooks_E: tuple[str, int] | None = None,
        hooks_F: tuple[str, int] | None = None,
        hooks_G: tuple[str, int] | None = None,
        hooks_H: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        if hooks_A is not None:
            self.hooks_A = hooks_A
        if hooks_B is not None:
            self.hooks_B = hooks_B
        if hooks_C is not None:
            self.hooks_C = hooks_C
        if hooks_D is not None:
            self.hooks_D = hooks_D
        if hooks_E is not None:
            self.hooks_E = hooks_E
        if hooks_F is not None:
            self.hooks_F = hooks_F
        if hooks_G is not None:
            self.hooks_G = hooks_G
        if hooks_H is not None:
            self.hooks_H = hooks_H
