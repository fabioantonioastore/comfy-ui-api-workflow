from typing import Literal, get_args

from ...base import BaseNode
from ...validation import validate_choice, validate_range

RecraftStyleV3RealisticImage_Substyles = Literal[
    "None",
    "b_and_w",
    "enterprise",
    "evening_light",
    "faded_nostalgia",
    "forest_life",
    "hard_flash",
    "hdr",
    "motion_blur",
    "mystic_naturalism",
    "natural_light",
    "natural_tones",
    "organic_calm",
    "real_life_glow",
    "retro_realism",
    "retro_snapshot",
    "studio_portrait",
    "urban_drama",
    "village_realism",
    "warm_folk",
]
RECRAFTSTYLEV3REALISTICIMAGE_SUBSTYLES = frozenset(
    get_args(RecraftStyleV3RealisticImage_Substyles)
)


class RecraftStyleV3RealisticImage(BaseNode):
    __slots__ = ("substyle",)
    CATEGORY = "partner/image/Recraft"
    OUTPUTS = {"recraft_style": 0}
    RETURN_TYPES = ("RECRAFT_V3_STYLE",)
    INPUT_SOCKET_TYPES = {"substyle": "COMBO"}

    def __init__(
        self,
        node_id: str,
        substyle: RecraftStyleV3RealisticImage_Substyles | tuple[str, int],
    ) -> None:
        super().__init__(node_id=node_id)
        self.substyle = substyle
        self.__validate()

    def __validate(self) -> None:
        validate_choice(
            "substyle", self.substyle, RECRAFTSTYLEV3REALISTICIMAGE_SUBSTYLES
        )


RecraftStyleV3DigitalIllustration_Substyles = Literal[
    "None",
    "2d_art_poster",
    "2d_art_poster_2",
    "antiquarian",
    "bold_fantasy",
    "child_book",
    "child_books",
    "cover",
    "crosshatch",
    "digital_engraving",
    "engraving_color",
    "expressionism",
    "freehand_details",
    "grain",
    "grain_20",
    "graphic_intensity",
    "hand_drawn",
    "hand_drawn_outline",
    "handmade_3d",
    "hard_comics",
    "infantile_sketch",
    "long_shadow",
    "modern_folk",
    "multicolor",
    "neon_calm",
    "noir",
    "nostalgic_pastel",
    "outline_details",
    "pastel_gradient",
    "pastel_sketch",
    "pixel_art",
    "plastic",
    "pop_art",
    "pop_renaissance",
    "seamless",
    "street_art",
    "tablet_sketch",
    "urban_glow",
    "urban_sketching",
    "vanilla_dreams",
    "young_adult_book",
    "young_adult_book_2",
]
RECRAFTSTYLEV3DIGITALILLUSTRATION_SUBSTYLES = frozenset(
    get_args(RecraftStyleV3DigitalIllustration_Substyles)
)


class RecraftStyleV3DigitalIllustration(BaseNode):
    __slots__ = ("substyle",)
    CATEGORY = "partner/image/Recraft"
    OUTPUTS = {"recraft_style": 0}
    RETURN_TYPES = ("RECRAFT_V3_STYLE",)
    INPUT_SOCKET_TYPES = {"substyle": "COMBO"}

    def __init__(
        self,
        node_id: str,
        substyle: RecraftStyleV3DigitalIllustration_Substyles | tuple[str, int],
    ) -> None:
        super().__init__(node_id=node_id)
        self.substyle = substyle
        self.__validate()

    def __validate(self) -> None:
        validate_choice(
            "substyle", self.substyle, RECRAFTSTYLEV3DIGITALILLUSTRATION_SUBSTYLES
        )


RecraftStyleV3LogoRaster_Substyles = Literal[
    "emblem_graffiti", "emblem_pop_art", "emblem_punk", "emblem_stamp", "emblem_vintage"
]
RECRAFTSTYLEV3LOGORASTER_SUBSTYLES = frozenset(
    get_args(RecraftStyleV3LogoRaster_Substyles)
)


class RecraftStyleV3LogoRaster(BaseNode):
    __slots__ = ("substyle",)
    CATEGORY = "partner/image/Recraft"
    OUTPUTS = {"recraft_style": 0}
    RETURN_TYPES = ("RECRAFT_V3_STYLE",)
    INPUT_SOCKET_TYPES = {"substyle": "COMBO"}

    def __init__(
        self,
        node_id: str,
        substyle: RecraftStyleV3LogoRaster_Substyles | tuple[str, int],
    ) -> None:
        super().__init__(node_id=node_id)
        self.substyle = substyle
        self.__validate()

    def __validate(self) -> None:
        validate_choice("substyle", self.substyle, RECRAFTSTYLEV3LOGORASTER_SUBSTYLES)


class RecraftStyleV3InfiniteStyleLibrary(BaseNode):
    __slots__ = ("style_id",)
    CATEGORY = "partner/image/Recraft"
    OUTPUTS = {"recraft_style": 0}
    RETURN_TYPES = ("RECRAFT_V3_STYLE",)
    INPUT_SOCKET_TYPES = {"style_id": "STRING"}

    def __init__(self, node_id: str, style_id: str | tuple[str, int] = "") -> None:
        super().__init__(node_id=node_id)
        self.style_id = style_id


class RecraftColorRGB(BaseNode):
    __slots__ = ("r", "g", "b", "recraft_color")
    CATEGORY = "partner/image/Recraft"
    OUTPUTS = {"recraft_color": 0}
    RETURN_TYPES = ("RECRAFT_COLOR",)
    INPUT_SOCKET_TYPES = {
        "r": "INT",
        "g": "INT",
        "b": "INT",
        "recraft_color": "RECRAFT_COLOR",
    }

    def __init__(
        self,
        node_id: str,
        r: int | tuple[str, int] = 0,
        g: int | tuple[str, int] = 0,
        b: int | tuple[str, int] = 0,
        recraft_color: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.r = r
        self.g = g
        self.b = b
        if recraft_color is not None:
            self.recraft_color = recraft_color
        self.__validate()

    def __validate(self) -> None:
        validate_range("r", self.r, 0, 255)
        validate_range("g", self.g, 0, 255)
        validate_range("b", self.b, 0, 255)


class RecraftControls(BaseNode):
    __slots__ = ("colors", "background_color")
    CATEGORY = "partner/image/Recraft"
    OUTPUTS = {"recraft_controls": 0}
    RETURN_TYPES = ("RECRAFT_CONTROLS",)
    INPUT_SOCKET_TYPES = {
        "colors": "RECRAFT_COLOR",
        "background_color": "RECRAFT_COLOR",
    }

    def __init__(
        self,
        node_id: str,
        colors: tuple[str, int] | None = None,
        background_color: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        if colors is not None:
            self.colors = colors
        if background_color is not None:
            self.background_color = background_color
