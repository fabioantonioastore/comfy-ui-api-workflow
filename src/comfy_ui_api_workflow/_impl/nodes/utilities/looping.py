from typing import Literal

from ..base import BaseNode
from ..dynamic import Combo, DynamicInputs, Link, Number, validate_dynamic
from ..validation import validate_bool


class EndLoop(BaseNode):
    __slots__ = (
        "accumulate",
        "output_value",
        "next_iteration_value",
        "terminations_termination0",
        "terminations_termination1",
        "terminations_termination2",
        "terminations_termination3",
        "terminations_termination4",
        "terminations_termination5",
        "terminations_termination6",
        "terminations_termination7",
        "terminations_termination8",
        "terminations_termination9",
        "terminations_termination10",
        "terminations_termination11",
        "terminations_termination12",
        "terminations_termination13",
        "terminations_termination14",
        "terminations_termination15",
        "terminations_termination16",
        "terminations_termination17",
        "terminations_termination18",
        "terminations_termination19",
        "terminations_termination20",
        "terminations_termination21",
        "terminations_termination22",
        "terminations_termination23",
        "terminations_termination24",
        "terminations_termination25",
        "terminations_termination26",
        "terminations_termination27",
        "terminations_termination28",
        "terminations_termination29",
        "terminations_termination30",
        "terminations_termination31",
        "terminations_termination32",
        "terminations_termination33",
        "terminations_termination34",
        "terminations_termination35",
        "terminations_termination36",
        "terminations_termination37",
        "terminations_termination38",
        "terminations_termination39",
        "terminations_termination40",
        "terminations_termination41",
        "terminations_termination42",
        "terminations_termination43",
        "terminations_termination44",
        "terminations_termination45",
        "terminations_termination46",
        "terminations_termination47",
        "terminations_termination48",
        "terminations_termination49",
    )
    CATEGORY = "utilities/looping"
    OUTPUTS = {"outputs": 0}
    RETURN_TYPES = ("COMFY_MATCHTYPE_V3",)
    INPUT_NAMES = {
        "terminations_termination0": "terminations.termination0",
        "terminations_termination1": "terminations.termination1",
        "terminations_termination2": "terminations.termination2",
        "terminations_termination3": "terminations.termination3",
        "terminations_termination4": "terminations.termination4",
        "terminations_termination5": "terminations.termination5",
        "terminations_termination6": "terminations.termination6",
        "terminations_termination7": "terminations.termination7",
        "terminations_termination8": "terminations.termination8",
        "terminations_termination9": "terminations.termination9",
        "terminations_termination10": "terminations.termination10",
        "terminations_termination11": "terminations.termination11",
        "terminations_termination12": "terminations.termination12",
        "terminations_termination13": "terminations.termination13",
        "terminations_termination14": "terminations.termination14",
        "terminations_termination15": "terminations.termination15",
        "terminations_termination16": "terminations.termination16",
        "terminations_termination17": "terminations.termination17",
        "terminations_termination18": "terminations.termination18",
        "terminations_termination19": "terminations.termination19",
        "terminations_termination20": "terminations.termination20",
        "terminations_termination21": "terminations.termination21",
        "terminations_termination22": "terminations.termination22",
        "terminations_termination23": "terminations.termination23",
        "terminations_termination24": "terminations.termination24",
        "terminations_termination25": "terminations.termination25",
        "terminations_termination26": "terminations.termination26",
        "terminations_termination27": "terminations.termination27",
        "terminations_termination28": "terminations.termination28",
        "terminations_termination29": "terminations.termination29",
        "terminations_termination30": "terminations.termination30",
        "terminations_termination31": "terminations.termination31",
        "terminations_termination32": "terminations.termination32",
        "terminations_termination33": "terminations.termination33",
        "terminations_termination34": "terminations.termination34",
        "terminations_termination35": "terminations.termination35",
        "terminations_termination36": "terminations.termination36",
        "terminations_termination37": "terminations.termination37",
        "terminations_termination38": "terminations.termination38",
        "terminations_termination39": "terminations.termination39",
        "terminations_termination40": "terminations.termination40",
        "terminations_termination41": "terminations.termination41",
        "terminations_termination42": "terminations.termination42",
        "terminations_termination43": "terminations.termination43",
        "terminations_termination44": "terminations.termination44",
        "terminations_termination45": "terminations.termination45",
        "terminations_termination46": "terminations.termination46",
        "terminations_termination47": "terminations.termination47",
        "terminations_termination48": "terminations.termination48",
        "terminations_termination49": "terminations.termination49",
    }
    INPUT_SOCKET_TYPES = {
        "accumulate": "BOOLEAN",
        "output_value": "COMFY_MATCHTYPE_V3",
        "next_iteration_value": "COMFY_MATCHTYPE_V3",
        "terminations_termination0": "*",
        "terminations_termination1": "*",
        "terminations_termination2": "*",
        "terminations_termination3": "*",
        "terminations_termination4": "*",
        "terminations_termination5": "*",
        "terminations_termination6": "*",
        "terminations_termination7": "*",
        "terminations_termination8": "*",
        "terminations_termination9": "*",
        "terminations_termination10": "*",
        "terminations_termination11": "*",
        "terminations_termination12": "*",
        "terminations_termination13": "*",
        "terminations_termination14": "*",
        "terminations_termination15": "*",
        "terminations_termination16": "*",
        "terminations_termination17": "*",
        "terminations_termination18": "*",
        "terminations_termination19": "*",
        "terminations_termination20": "*",
        "terminations_termination21": "*",
        "terminations_termination22": "*",
        "terminations_termination23": "*",
        "terminations_termination24": "*",
        "terminations_termination25": "*",
        "terminations_termination26": "*",
        "terminations_termination27": "*",
        "terminations_termination28": "*",
        "terminations_termination29": "*",
        "terminations_termination30": "*",
        "terminations_termination31": "*",
        "terminations_termination32": "*",
        "terminations_termination33": "*",
        "terminations_termination34": "*",
        "terminations_termination35": "*",
        "terminations_termination36": "*",
        "terminations_termination37": "*",
        "terminations_termination38": "*",
        "terminations_termination39": "*",
        "terminations_termination40": "*",
        "terminations_termination41": "*",
        "terminations_termination42": "*",
        "terminations_termination43": "*",
        "terminations_termination44": "*",
        "terminations_termination45": "*",
        "terminations_termination46": "*",
        "terminations_termination47": "*",
        "terminations_termination48": "*",
        "terminations_termination49": "*",
    }

    def __init__(
        self,
        node_id: str,
        accumulate: bool | tuple[str, int] = False,
        output_value: tuple[str, int] | None = None,
        next_iteration_value: tuple[str, int] | None = None,
        terminations_termination0: tuple[str, int] | None = None,
        terminations_termination1: tuple[str, int] | None = None,
        terminations_termination2: tuple[str, int] | None = None,
        terminations_termination3: tuple[str, int] | None = None,
        terminations_termination4: tuple[str, int] | None = None,
        terminations_termination5: tuple[str, int] | None = None,
        terminations_termination6: tuple[str, int] | None = None,
        terminations_termination7: tuple[str, int] | None = None,
        terminations_termination8: tuple[str, int] | None = None,
        terminations_termination9: tuple[str, int] | None = None,
        terminations_termination10: tuple[str, int] | None = None,
        terminations_termination11: tuple[str, int] | None = None,
        terminations_termination12: tuple[str, int] | None = None,
        terminations_termination13: tuple[str, int] | None = None,
        terminations_termination14: tuple[str, int] | None = None,
        terminations_termination15: tuple[str, int] | None = None,
        terminations_termination16: tuple[str, int] | None = None,
        terminations_termination17: tuple[str, int] | None = None,
        terminations_termination18: tuple[str, int] | None = None,
        terminations_termination19: tuple[str, int] | None = None,
        terminations_termination20: tuple[str, int] | None = None,
        terminations_termination21: tuple[str, int] | None = None,
        terminations_termination22: tuple[str, int] | None = None,
        terminations_termination23: tuple[str, int] | None = None,
        terminations_termination24: tuple[str, int] | None = None,
        terminations_termination25: tuple[str, int] | None = None,
        terminations_termination26: tuple[str, int] | None = None,
        terminations_termination27: tuple[str, int] | None = None,
        terminations_termination28: tuple[str, int] | None = None,
        terminations_termination29: tuple[str, int] | None = None,
        terminations_termination30: tuple[str, int] | None = None,
        terminations_termination31: tuple[str, int] | None = None,
        terminations_termination32: tuple[str, int] | None = None,
        terminations_termination33: tuple[str, int] | None = None,
        terminations_termination34: tuple[str, int] | None = None,
        terminations_termination35: tuple[str, int] | None = None,
        terminations_termination36: tuple[str, int] | None = None,
        terminations_termination37: tuple[str, int] | None = None,
        terminations_termination38: tuple[str, int] | None = None,
        terminations_termination39: tuple[str, int] | None = None,
        terminations_termination40: tuple[str, int] | None = None,
        terminations_termination41: tuple[str, int] | None = None,
        terminations_termination42: tuple[str, int] | None = None,
        terminations_termination43: tuple[str, int] | None = None,
        terminations_termination44: tuple[str, int] | None = None,
        terminations_termination45: tuple[str, int] | None = None,
        terminations_termination46: tuple[str, int] | None = None,
        terminations_termination47: tuple[str, int] | None = None,
        terminations_termination48: tuple[str, int] | None = None,
        terminations_termination49: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.accumulate = accumulate
        if output_value is not None:
            self.output_value = output_value
        if next_iteration_value is not None:
            self.next_iteration_value = next_iteration_value
        if terminations_termination0 is not None:
            self.terminations_termination0 = terminations_termination0
        if terminations_termination1 is not None:
            self.terminations_termination1 = terminations_termination1
        if terminations_termination2 is not None:
            self.terminations_termination2 = terminations_termination2
        if terminations_termination3 is not None:
            self.terminations_termination3 = terminations_termination3
        if terminations_termination4 is not None:
            self.terminations_termination4 = terminations_termination4
        if terminations_termination5 is not None:
            self.terminations_termination5 = terminations_termination5
        if terminations_termination6 is not None:
            self.terminations_termination6 = terminations_termination6
        if terminations_termination7 is not None:
            self.terminations_termination7 = terminations_termination7
        if terminations_termination8 is not None:
            self.terminations_termination8 = terminations_termination8
        if terminations_termination9 is not None:
            self.terminations_termination9 = terminations_termination9
        if terminations_termination10 is not None:
            self.terminations_termination10 = terminations_termination10
        if terminations_termination11 is not None:
            self.terminations_termination11 = terminations_termination11
        if terminations_termination12 is not None:
            self.terminations_termination12 = terminations_termination12
        if terminations_termination13 is not None:
            self.terminations_termination13 = terminations_termination13
        if terminations_termination14 is not None:
            self.terminations_termination14 = terminations_termination14
        if terminations_termination15 is not None:
            self.terminations_termination15 = terminations_termination15
        if terminations_termination16 is not None:
            self.terminations_termination16 = terminations_termination16
        if terminations_termination17 is not None:
            self.terminations_termination17 = terminations_termination17
        if terminations_termination18 is not None:
            self.terminations_termination18 = terminations_termination18
        if terminations_termination19 is not None:
            self.terminations_termination19 = terminations_termination19
        if terminations_termination20 is not None:
            self.terminations_termination20 = terminations_termination20
        if terminations_termination21 is not None:
            self.terminations_termination21 = terminations_termination21
        if terminations_termination22 is not None:
            self.terminations_termination22 = terminations_termination22
        if terminations_termination23 is not None:
            self.terminations_termination23 = terminations_termination23
        if terminations_termination24 is not None:
            self.terminations_termination24 = terminations_termination24
        if terminations_termination25 is not None:
            self.terminations_termination25 = terminations_termination25
        if terminations_termination26 is not None:
            self.terminations_termination26 = terminations_termination26
        if terminations_termination27 is not None:
            self.terminations_termination27 = terminations_termination27
        if terminations_termination28 is not None:
            self.terminations_termination28 = terminations_termination28
        if terminations_termination29 is not None:
            self.terminations_termination29 = terminations_termination29
        if terminations_termination30 is not None:
            self.terminations_termination30 = terminations_termination30
        if terminations_termination31 is not None:
            self.terminations_termination31 = terminations_termination31
        if terminations_termination32 is not None:
            self.terminations_termination32 = terminations_termination32
        if terminations_termination33 is not None:
            self.terminations_termination33 = terminations_termination33
        if terminations_termination34 is not None:
            self.terminations_termination34 = terminations_termination34
        if terminations_termination35 is not None:
            self.terminations_termination35 = terminations_termination35
        if terminations_termination36 is not None:
            self.terminations_termination36 = terminations_termination36
        if terminations_termination37 is not None:
            self.terminations_termination37 = terminations_termination37
        if terminations_termination38 is not None:
            self.terminations_termination38 = terminations_termination38
        if terminations_termination39 is not None:
            self.terminations_termination39 = terminations_termination39
        if terminations_termination40 is not None:
            self.terminations_termination40 = terminations_termination40
        if terminations_termination41 is not None:
            self.terminations_termination41 = terminations_termination41
        if terminations_termination42 is not None:
            self.terminations_termination42 = terminations_termination42
        if terminations_termination43 is not None:
            self.terminations_termination43 = terminations_termination43
        if terminations_termination44 is not None:
            self.terminations_termination44 = terminations_termination44
        if terminations_termination45 is not None:
            self.terminations_termination45 = terminations_termination45
        if terminations_termination46 is not None:
            self.terminations_termination46 = terminations_termination46
        if terminations_termination47 is not None:
            self.terminations_termination47 = terminations_termination47
        if terminations_termination48 is not None:
            self.terminations_termination48 = terminations_termination48
        if terminations_termination49 is not None:
            self.terminations_termination49 = terminations_termination49
        self.__validate()

    def __validate(self) -> None:
        validate_bool("accumulate", self.accumulate)


StartLoop_Modes = Literal["simple", "For", "List"]


class StartLoop(DynamicInputs, BaseNode):
    __slots__ = (
        "mode",
        "mode_num_iterations",
        "mode_start_iteration_index",
        "mode_max_iteration",
        "mode_step",
        "mode_list",
        "cache_iterations",
        "parent_iteration",
        "initial_iteration_value",
    )
    CATEGORY = "utilities/looping"
    OUTPUTS = {
        "iteration_index": 0,
        "is_first": 1,
        "is_last": 2,
        "list_item": 3,
        "current_iteration_value": 4,
    }
    RETURN_TYPES = (
        "INT",
        "BOOLEAN",
        "BOOLEAN",
        "COMFY_MATCHTYPE_V3",
        "COMFY_MATCHTYPE_V3",
    )
    INPUT_NAMES = {
        "mode_num_iterations": "mode.num_iterations",
        "mode_start_iteration_index": "mode.start_iteration_index",
        "mode_max_iteration": "mode.max_iteration",
        "mode_step": "mode.step",
        "mode_list": "mode.list",
    }
    INPUT_SOCKET_TYPES = {
        "mode": "COMFY_DYNAMICCOMBO_V3",
        "mode_num_iterations": "INT",
        "mode_start_iteration_index": "INT",
        "mode_max_iteration": "INT",
        "mode_step": "INT",
        "mode_list": "COMFY_MATCHTYPE_V3",
        "cache_iterations": "BOOLEAN",
        "parent_iteration": "INT",
        "initial_iteration_value": "COMFY_MATCHTYPE_V3",
    }
    DYNAMIC_INPUTS = {
        "mode": Combo(
            {
                "simple": {"mode_num_iterations": Number(default=4, low=0)},
                "For": {
                    "mode_start_iteration_index": Number(default=0),
                    "mode_max_iteration": Number(default=4, high=18446744073709551615),
                    "mode_step": Number(default=1, low=1),
                },
                "List": {"mode_list": Link()},
            }
        )
    }

    def __init__(
        self,
        node_id: str,
        mode: StartLoop_Modes = "simple",
        mode_num_iterations: int | tuple[str, int] | None = None,
        mode_start_iteration_index: int | tuple[str, int] | None = None,
        mode_max_iteration: int | tuple[str, int] | None = None,
        mode_step: int | tuple[str, int] | None = None,
        mode_list: tuple[str, int] | None = None,
        cache_iterations: bool | tuple[str, int] = False,
        parent_iteration: tuple[str, int] | None = None,
        initial_iteration_value: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.mode = mode
        if mode_num_iterations is not None:
            self.mode_num_iterations = mode_num_iterations
        if mode_start_iteration_index is not None:
            self.mode_start_iteration_index = mode_start_iteration_index
        if mode_max_iteration is not None:
            self.mode_max_iteration = mode_max_iteration
        if mode_step is not None:
            self.mode_step = mode_step
        if mode_list is not None:
            self.mode_list = mode_list
        self.cache_iterations = cache_iterations
        if parent_iteration is not None:
            self.parent_iteration = parent_iteration
        if initial_iteration_value is not None:
            self.initial_iteration_value = initial_iteration_value
        self.__validate()

    def __validate(self) -> None:
        validate_bool("cache_iterations", self.cache_iterations)
        validate_dynamic(self)
