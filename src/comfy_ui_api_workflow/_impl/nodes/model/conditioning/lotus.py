from ...base import BaseNode


class LotusConditioning(BaseNode):
    __slots__ = ()
    CATEGORY = "model/conditioning/lotus"
    OUTPUTS = {"conditioning": 0}
    RETURN_TYPES = ("CONDITIONING",)
