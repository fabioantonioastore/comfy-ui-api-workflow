from typing import Literal

from ..base import BaseNode
from ..dynamic import Combo, DynamicInputs, Number, validate_dynamic

GLSLShader_Size_Modes = Literal["from_input", "custom"]


class GLSLShader(DynamicInputs, BaseNode):
    __slots__ = (
        "fragment_shader",
        "size_mode",
        "size_mode_width",
        "size_mode_height",
        "images_image0",
        "images_image1",
        "images_image2",
        "images_image3",
        "images_image4",
        "floats_u_float0",
        "floats_u_float1",
        "floats_u_float2",
        "floats_u_float3",
        "floats_u_float4",
        "floats_u_float5",
        "floats_u_float6",
        "floats_u_float7",
        "floats_u_float8",
        "floats_u_float9",
        "floats_u_float10",
        "floats_u_float11",
        "floats_u_float12",
        "floats_u_float13",
        "floats_u_float14",
        "floats_u_float15",
        "floats_u_float16",
        "floats_u_float17",
        "floats_u_float18",
        "floats_u_float19",
        "ints_u_int0",
        "ints_u_int1",
        "ints_u_int2",
        "ints_u_int3",
        "ints_u_int4",
        "ints_u_int5",
        "ints_u_int6",
        "ints_u_int7",
        "ints_u_int8",
        "ints_u_int9",
        "ints_u_int10",
        "ints_u_int11",
        "ints_u_int12",
        "ints_u_int13",
        "ints_u_int14",
        "ints_u_int15",
        "ints_u_int16",
        "ints_u_int17",
        "ints_u_int18",
        "ints_u_int19",
        "bools_u_bool0",
        "bools_u_bool1",
        "bools_u_bool2",
        "bools_u_bool3",
        "bools_u_bool4",
        "bools_u_bool5",
        "bools_u_bool6",
        "bools_u_bool7",
        "bools_u_bool8",
        "bools_u_bool9",
        "curves_u_curve0",
        "curves_u_curve1",
        "curves_u_curve2",
        "curves_u_curve3",
    )
    CATEGORY = "image/shader"
    OUTPUTS = {"IMAGE0": 0, "IMAGE1": 1, "IMAGE2": 2, "IMAGE3": 3}
    RETURN_TYPES = (
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "IMAGE",
    )
    INPUT_NAMES = {
        "size_mode_width": "size_mode.width",
        "size_mode_height": "size_mode.height",
        "images_image0": "images.image0",
        "images_image1": "images.image1",
        "images_image2": "images.image2",
        "images_image3": "images.image3",
        "images_image4": "images.image4",
        "floats_u_float0": "floats.u_float0",
        "floats_u_float1": "floats.u_float1",
        "floats_u_float2": "floats.u_float2",
        "floats_u_float3": "floats.u_float3",
        "floats_u_float4": "floats.u_float4",
        "floats_u_float5": "floats.u_float5",
        "floats_u_float6": "floats.u_float6",
        "floats_u_float7": "floats.u_float7",
        "floats_u_float8": "floats.u_float8",
        "floats_u_float9": "floats.u_float9",
        "floats_u_float10": "floats.u_float10",
        "floats_u_float11": "floats.u_float11",
        "floats_u_float12": "floats.u_float12",
        "floats_u_float13": "floats.u_float13",
        "floats_u_float14": "floats.u_float14",
        "floats_u_float15": "floats.u_float15",
        "floats_u_float16": "floats.u_float16",
        "floats_u_float17": "floats.u_float17",
        "floats_u_float18": "floats.u_float18",
        "floats_u_float19": "floats.u_float19",
        "ints_u_int0": "ints.u_int0",
        "ints_u_int1": "ints.u_int1",
        "ints_u_int2": "ints.u_int2",
        "ints_u_int3": "ints.u_int3",
        "ints_u_int4": "ints.u_int4",
        "ints_u_int5": "ints.u_int5",
        "ints_u_int6": "ints.u_int6",
        "ints_u_int7": "ints.u_int7",
        "ints_u_int8": "ints.u_int8",
        "ints_u_int9": "ints.u_int9",
        "ints_u_int10": "ints.u_int10",
        "ints_u_int11": "ints.u_int11",
        "ints_u_int12": "ints.u_int12",
        "ints_u_int13": "ints.u_int13",
        "ints_u_int14": "ints.u_int14",
        "ints_u_int15": "ints.u_int15",
        "ints_u_int16": "ints.u_int16",
        "ints_u_int17": "ints.u_int17",
        "ints_u_int18": "ints.u_int18",
        "ints_u_int19": "ints.u_int19",
        "bools_u_bool0": "bools.u_bool0",
        "bools_u_bool1": "bools.u_bool1",
        "bools_u_bool2": "bools.u_bool2",
        "bools_u_bool3": "bools.u_bool3",
        "bools_u_bool4": "bools.u_bool4",
        "bools_u_bool5": "bools.u_bool5",
        "bools_u_bool6": "bools.u_bool6",
        "bools_u_bool7": "bools.u_bool7",
        "bools_u_bool8": "bools.u_bool8",
        "bools_u_bool9": "bools.u_bool9",
        "curves_u_curve0": "curves.u_curve0",
        "curves_u_curve1": "curves.u_curve1",
        "curves_u_curve2": "curves.u_curve2",
        "curves_u_curve3": "curves.u_curve3",
    }
    INPUT_SOCKET_TYPES = {
        "fragment_shader": "STRING",
        "size_mode": "COMFY_DYNAMICCOMBO_V3",
        "size_mode_width": "INT",
        "size_mode_height": "INT",
        "images_image0": "IMAGE",
        "images_image1": "IMAGE",
        "images_image2": "IMAGE",
        "images_image3": "IMAGE",
        "images_image4": "IMAGE",
        "floats_u_float0": "FLOAT",
        "floats_u_float1": "FLOAT",
        "floats_u_float2": "FLOAT",
        "floats_u_float3": "FLOAT",
        "floats_u_float4": "FLOAT",
        "floats_u_float5": "FLOAT",
        "floats_u_float6": "FLOAT",
        "floats_u_float7": "FLOAT",
        "floats_u_float8": "FLOAT",
        "floats_u_float9": "FLOAT",
        "floats_u_float10": "FLOAT",
        "floats_u_float11": "FLOAT",
        "floats_u_float12": "FLOAT",
        "floats_u_float13": "FLOAT",
        "floats_u_float14": "FLOAT",
        "floats_u_float15": "FLOAT",
        "floats_u_float16": "FLOAT",
        "floats_u_float17": "FLOAT",
        "floats_u_float18": "FLOAT",
        "floats_u_float19": "FLOAT",
        "ints_u_int0": "INT",
        "ints_u_int1": "INT",
        "ints_u_int2": "INT",
        "ints_u_int3": "INT",
        "ints_u_int4": "INT",
        "ints_u_int5": "INT",
        "ints_u_int6": "INT",
        "ints_u_int7": "INT",
        "ints_u_int8": "INT",
        "ints_u_int9": "INT",
        "ints_u_int10": "INT",
        "ints_u_int11": "INT",
        "ints_u_int12": "INT",
        "ints_u_int13": "INT",
        "ints_u_int14": "INT",
        "ints_u_int15": "INT",
        "ints_u_int16": "INT",
        "ints_u_int17": "INT",
        "ints_u_int18": "INT",
        "ints_u_int19": "INT",
        "bools_u_bool0": "BOOLEAN",
        "bools_u_bool1": "BOOLEAN",
        "bools_u_bool2": "BOOLEAN",
        "bools_u_bool3": "BOOLEAN",
        "bools_u_bool4": "BOOLEAN",
        "bools_u_bool5": "BOOLEAN",
        "bools_u_bool6": "BOOLEAN",
        "bools_u_bool7": "BOOLEAN",
        "bools_u_bool8": "BOOLEAN",
        "bools_u_bool9": "BOOLEAN",
        "curves_u_curve0": "CURVE",
        "curves_u_curve1": "CURVE",
        "curves_u_curve2": "CURVE",
        "curves_u_curve3": "CURVE",
    }
    DYNAMIC_INPUTS = {
        "size_mode": Combo(
            {
                "from_input": {},
                "custom": {
                    "size_mode_width": Number(default=512, low=1, high=16384),
                    "size_mode_height": Number(default=512, low=1, high=16384),
                },
            }
        )
    }

    def __init__(
        self,
        node_id: str,
        images_image0: tuple[str, int],
        fragment_shader: (
            str | tuple[str, int]
        ) = "#version 300 es\nprecision highp float;\n\nuniform sampler2D u_image0;\nuniform vec2 u_resolution;\n\nin vec2 v_texCoord;\nlayout(location = 0) out vec4 fragColor0;\n\nvoid main() {\n    fragColor0 = texture(u_image0, v_texCoord);\n}\n",
        size_mode: GLSLShader_Size_Modes = "from_input",
        size_mode_width: int | tuple[str, int] | None = None,
        size_mode_height: int | tuple[str, int] | None = None,
        images_image1: tuple[str, int] | None = None,
        images_image2: tuple[str, int] | None = None,
        images_image3: tuple[str, int] | None = None,
        images_image4: tuple[str, int] | None = None,
        floats_u_float0: tuple[str, int] | None = None,
        floats_u_float1: tuple[str, int] | None = None,
        floats_u_float2: tuple[str, int] | None = None,
        floats_u_float3: tuple[str, int] | None = None,
        floats_u_float4: tuple[str, int] | None = None,
        floats_u_float5: tuple[str, int] | None = None,
        floats_u_float6: tuple[str, int] | None = None,
        floats_u_float7: tuple[str, int] | None = None,
        floats_u_float8: tuple[str, int] | None = None,
        floats_u_float9: tuple[str, int] | None = None,
        floats_u_float10: tuple[str, int] | None = None,
        floats_u_float11: tuple[str, int] | None = None,
        floats_u_float12: tuple[str, int] | None = None,
        floats_u_float13: tuple[str, int] | None = None,
        floats_u_float14: tuple[str, int] | None = None,
        floats_u_float15: tuple[str, int] | None = None,
        floats_u_float16: tuple[str, int] | None = None,
        floats_u_float17: tuple[str, int] | None = None,
        floats_u_float18: tuple[str, int] | None = None,
        floats_u_float19: tuple[str, int] | None = None,
        ints_u_int0: tuple[str, int] | None = None,
        ints_u_int1: tuple[str, int] | None = None,
        ints_u_int2: tuple[str, int] | None = None,
        ints_u_int3: tuple[str, int] | None = None,
        ints_u_int4: tuple[str, int] | None = None,
        ints_u_int5: tuple[str, int] | None = None,
        ints_u_int6: tuple[str, int] | None = None,
        ints_u_int7: tuple[str, int] | None = None,
        ints_u_int8: tuple[str, int] | None = None,
        ints_u_int9: tuple[str, int] | None = None,
        ints_u_int10: tuple[str, int] | None = None,
        ints_u_int11: tuple[str, int] | None = None,
        ints_u_int12: tuple[str, int] | None = None,
        ints_u_int13: tuple[str, int] | None = None,
        ints_u_int14: tuple[str, int] | None = None,
        ints_u_int15: tuple[str, int] | None = None,
        ints_u_int16: tuple[str, int] | None = None,
        ints_u_int17: tuple[str, int] | None = None,
        ints_u_int18: tuple[str, int] | None = None,
        ints_u_int19: tuple[str, int] | None = None,
        bools_u_bool0: tuple[str, int] | None = None,
        bools_u_bool1: tuple[str, int] | None = None,
        bools_u_bool2: tuple[str, int] | None = None,
        bools_u_bool3: tuple[str, int] | None = None,
        bools_u_bool4: tuple[str, int] | None = None,
        bools_u_bool5: tuple[str, int] | None = None,
        bools_u_bool6: tuple[str, int] | None = None,
        bools_u_bool7: tuple[str, int] | None = None,
        bools_u_bool8: tuple[str, int] | None = None,
        bools_u_bool9: tuple[str, int] | None = None,
        curves_u_curve0: tuple[str, int] | None = None,
        curves_u_curve1: tuple[str, int] | None = None,
        curves_u_curve2: tuple[str, int] | None = None,
        curves_u_curve3: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.fragment_shader = fragment_shader
        self.size_mode = size_mode
        if size_mode_width is not None:
            self.size_mode_width = size_mode_width
        if size_mode_height is not None:
            self.size_mode_height = size_mode_height
        self.images_image0 = images_image0
        if images_image1 is not None:
            self.images_image1 = images_image1
        if images_image2 is not None:
            self.images_image2 = images_image2
        if images_image3 is not None:
            self.images_image3 = images_image3
        if images_image4 is not None:
            self.images_image4 = images_image4
        if floats_u_float0 is not None:
            self.floats_u_float0 = floats_u_float0
        if floats_u_float1 is not None:
            self.floats_u_float1 = floats_u_float1
        if floats_u_float2 is not None:
            self.floats_u_float2 = floats_u_float2
        if floats_u_float3 is not None:
            self.floats_u_float3 = floats_u_float3
        if floats_u_float4 is not None:
            self.floats_u_float4 = floats_u_float4
        if floats_u_float5 is not None:
            self.floats_u_float5 = floats_u_float5
        if floats_u_float6 is not None:
            self.floats_u_float6 = floats_u_float6
        if floats_u_float7 is not None:
            self.floats_u_float7 = floats_u_float7
        if floats_u_float8 is not None:
            self.floats_u_float8 = floats_u_float8
        if floats_u_float9 is not None:
            self.floats_u_float9 = floats_u_float9
        if floats_u_float10 is not None:
            self.floats_u_float10 = floats_u_float10
        if floats_u_float11 is not None:
            self.floats_u_float11 = floats_u_float11
        if floats_u_float12 is not None:
            self.floats_u_float12 = floats_u_float12
        if floats_u_float13 is not None:
            self.floats_u_float13 = floats_u_float13
        if floats_u_float14 is not None:
            self.floats_u_float14 = floats_u_float14
        if floats_u_float15 is not None:
            self.floats_u_float15 = floats_u_float15
        if floats_u_float16 is not None:
            self.floats_u_float16 = floats_u_float16
        if floats_u_float17 is not None:
            self.floats_u_float17 = floats_u_float17
        if floats_u_float18 is not None:
            self.floats_u_float18 = floats_u_float18
        if floats_u_float19 is not None:
            self.floats_u_float19 = floats_u_float19
        if ints_u_int0 is not None:
            self.ints_u_int0 = ints_u_int0
        if ints_u_int1 is not None:
            self.ints_u_int1 = ints_u_int1
        if ints_u_int2 is not None:
            self.ints_u_int2 = ints_u_int2
        if ints_u_int3 is not None:
            self.ints_u_int3 = ints_u_int3
        if ints_u_int4 is not None:
            self.ints_u_int4 = ints_u_int4
        if ints_u_int5 is not None:
            self.ints_u_int5 = ints_u_int5
        if ints_u_int6 is not None:
            self.ints_u_int6 = ints_u_int6
        if ints_u_int7 is not None:
            self.ints_u_int7 = ints_u_int7
        if ints_u_int8 is not None:
            self.ints_u_int8 = ints_u_int8
        if ints_u_int9 is not None:
            self.ints_u_int9 = ints_u_int9
        if ints_u_int10 is not None:
            self.ints_u_int10 = ints_u_int10
        if ints_u_int11 is not None:
            self.ints_u_int11 = ints_u_int11
        if ints_u_int12 is not None:
            self.ints_u_int12 = ints_u_int12
        if ints_u_int13 is not None:
            self.ints_u_int13 = ints_u_int13
        if ints_u_int14 is not None:
            self.ints_u_int14 = ints_u_int14
        if ints_u_int15 is not None:
            self.ints_u_int15 = ints_u_int15
        if ints_u_int16 is not None:
            self.ints_u_int16 = ints_u_int16
        if ints_u_int17 is not None:
            self.ints_u_int17 = ints_u_int17
        if ints_u_int18 is not None:
            self.ints_u_int18 = ints_u_int18
        if ints_u_int19 is not None:
            self.ints_u_int19 = ints_u_int19
        if bools_u_bool0 is not None:
            self.bools_u_bool0 = bools_u_bool0
        if bools_u_bool1 is not None:
            self.bools_u_bool1 = bools_u_bool1
        if bools_u_bool2 is not None:
            self.bools_u_bool2 = bools_u_bool2
        if bools_u_bool3 is not None:
            self.bools_u_bool3 = bools_u_bool3
        if bools_u_bool4 is not None:
            self.bools_u_bool4 = bools_u_bool4
        if bools_u_bool5 is not None:
            self.bools_u_bool5 = bools_u_bool5
        if bools_u_bool6 is not None:
            self.bools_u_bool6 = bools_u_bool6
        if bools_u_bool7 is not None:
            self.bools_u_bool7 = bools_u_bool7
        if bools_u_bool8 is not None:
            self.bools_u_bool8 = bools_u_bool8
        if bools_u_bool9 is not None:
            self.bools_u_bool9 = bools_u_bool9
        if curves_u_curve0 is not None:
            self.curves_u_curve0 = curves_u_curve0
        if curves_u_curve1 is not None:
            self.curves_u_curve1 = curves_u_curve1
        if curves_u_curve2 is not None:
            self.curves_u_curve2 = curves_u_curve2
        if curves_u_curve3 is not None:
            self.curves_u_curve3 = curves_u_curve3
        self.__validate()

    def __validate(self) -> None:
        validate_dynamic(self)
