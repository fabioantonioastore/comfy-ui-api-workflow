from ...base import BaseNode


class CLIPTextEncodeHiDream(BaseNode):
    __slots__ = ("clip", "clip_l", "clip_g", "t5xxl", "llama")
    CATEGORY = "model/conditioning/hidream"
    OUTPUTS = {"CONDITIONING": 0}
    RETURN_TYPES = ("CONDITIONING",)
    INPUT_SOCKET_TYPES = {
        "clip": "CLIP",
        "clip_l": "STRING",
        "clip_g": "STRING",
        "t5xxl": "STRING",
        "llama": "STRING",
    }

    def __init__(
        self,
        node_id: str,
        clip: tuple[str, int],
        clip_l: str | tuple[str, int],
        clip_g: str | tuple[str, int],
        t5xxl: str | tuple[str, int],
        llama: str | tuple[str, int],
    ) -> None:
        super().__init__(node_id=node_id)
        self.clip = clip
        self.clip_l = clip_l
        self.clip_g = clip_g
        self.t5xxl = t5xxl
        self.llama = llama


class HiDreamO1ReferenceImages(BaseNode):
    __slots__ = (
        "positive",
        "negative",
        "images_image_1",
        "images_image_2",
        "images_image_3",
        "images_image_4",
        "images_image_5",
        "images_image_6",
        "images_image_7",
        "images_image_8",
        "images_image_9",
        "images_image_10",
        "images_image_11",
        "images_image_12",
        "images_image_13",
        "images_image_14",
        "images_image_15",
        "images_image_16",
        "images_image_17",
        "images_image_18",
        "images_image_19",
        "images_image_20",
        "images_image_21",
        "images_image_22",
        "images_image_23",
        "images_image_24",
        "images_image_25",
        "images_image_26",
        "images_image_27",
        "images_image_28",
        "images_image_29",
        "images_image_30",
        "images_image_31",
        "images_image_32",
        "images_image_33",
        "images_image_34",
        "images_image_35",
        "images_image_36",
        "images_image_37",
        "images_image_38",
        "images_image_39",
        "images_image_40",
        "images_image_41",
        "images_image_42",
        "images_image_43",
        "images_image_44",
        "images_image_45",
        "images_image_46",
        "images_image_47",
        "images_image_48",
        "images_image_49",
        "images_image_50",
        "images_image_51",
        "images_image_52",
        "images_image_53",
        "images_image_54",
        "images_image_55",
        "images_image_56",
        "images_image_57",
        "images_image_58",
        "images_image_59",
        "images_image_60",
        "images_image_61",
        "images_image_62",
        "images_image_63",
        "images_image_64",
        "images_image_65",
        "images_image_66",
        "images_image_67",
        "images_image_68",
        "images_image_69",
        "images_image_70",
        "images_image_71",
        "images_image_72",
        "images_image_73",
        "images_image_74",
        "images_image_75",
        "images_image_76",
        "images_image_77",
        "images_image_78",
        "images_image_79",
        "images_image_80",
        "images_image_81",
        "images_image_82",
        "images_image_83",
        "images_image_84",
        "images_image_85",
        "images_image_86",
        "images_image_87",
        "images_image_88",
        "images_image_89",
        "images_image_90",
        "images_image_91",
        "images_image_92",
        "images_image_93",
        "images_image_94",
        "images_image_95",
        "images_image_96",
        "images_image_97",
        "images_image_98",
        "images_image_99",
        "images_image_100",
    )
    CATEGORY = "model/conditioning/hidream"
    OUTPUTS = {"positive": 0, "negative": 1}
    RETURN_TYPES = (
        "CONDITIONING",
        "CONDITIONING",
    )
    INPUT_NAMES = {
        "images_image_1": "images.image_1",
        "images_image_2": "images.image_2",
        "images_image_3": "images.image_3",
        "images_image_4": "images.image_4",
        "images_image_5": "images.image_5",
        "images_image_6": "images.image_6",
        "images_image_7": "images.image_7",
        "images_image_8": "images.image_8",
        "images_image_9": "images.image_9",
        "images_image_10": "images.image_10",
        "images_image_11": "images.image_11",
        "images_image_12": "images.image_12",
        "images_image_13": "images.image_13",
        "images_image_14": "images.image_14",
        "images_image_15": "images.image_15",
        "images_image_16": "images.image_16",
        "images_image_17": "images.image_17",
        "images_image_18": "images.image_18",
        "images_image_19": "images.image_19",
        "images_image_20": "images.image_20",
        "images_image_21": "images.image_21",
        "images_image_22": "images.image_22",
        "images_image_23": "images.image_23",
        "images_image_24": "images.image_24",
        "images_image_25": "images.image_25",
        "images_image_26": "images.image_26",
        "images_image_27": "images.image_27",
        "images_image_28": "images.image_28",
        "images_image_29": "images.image_29",
        "images_image_30": "images.image_30",
        "images_image_31": "images.image_31",
        "images_image_32": "images.image_32",
        "images_image_33": "images.image_33",
        "images_image_34": "images.image_34",
        "images_image_35": "images.image_35",
        "images_image_36": "images.image_36",
        "images_image_37": "images.image_37",
        "images_image_38": "images.image_38",
        "images_image_39": "images.image_39",
        "images_image_40": "images.image_40",
        "images_image_41": "images.image_41",
        "images_image_42": "images.image_42",
        "images_image_43": "images.image_43",
        "images_image_44": "images.image_44",
        "images_image_45": "images.image_45",
        "images_image_46": "images.image_46",
        "images_image_47": "images.image_47",
        "images_image_48": "images.image_48",
        "images_image_49": "images.image_49",
        "images_image_50": "images.image_50",
        "images_image_51": "images.image_51",
        "images_image_52": "images.image_52",
        "images_image_53": "images.image_53",
        "images_image_54": "images.image_54",
        "images_image_55": "images.image_55",
        "images_image_56": "images.image_56",
        "images_image_57": "images.image_57",
        "images_image_58": "images.image_58",
        "images_image_59": "images.image_59",
        "images_image_60": "images.image_60",
        "images_image_61": "images.image_61",
        "images_image_62": "images.image_62",
        "images_image_63": "images.image_63",
        "images_image_64": "images.image_64",
        "images_image_65": "images.image_65",
        "images_image_66": "images.image_66",
        "images_image_67": "images.image_67",
        "images_image_68": "images.image_68",
        "images_image_69": "images.image_69",
        "images_image_70": "images.image_70",
        "images_image_71": "images.image_71",
        "images_image_72": "images.image_72",
        "images_image_73": "images.image_73",
        "images_image_74": "images.image_74",
        "images_image_75": "images.image_75",
        "images_image_76": "images.image_76",
        "images_image_77": "images.image_77",
        "images_image_78": "images.image_78",
        "images_image_79": "images.image_79",
        "images_image_80": "images.image_80",
        "images_image_81": "images.image_81",
        "images_image_82": "images.image_82",
        "images_image_83": "images.image_83",
        "images_image_84": "images.image_84",
        "images_image_85": "images.image_85",
        "images_image_86": "images.image_86",
        "images_image_87": "images.image_87",
        "images_image_88": "images.image_88",
        "images_image_89": "images.image_89",
        "images_image_90": "images.image_90",
        "images_image_91": "images.image_91",
        "images_image_92": "images.image_92",
        "images_image_93": "images.image_93",
        "images_image_94": "images.image_94",
        "images_image_95": "images.image_95",
        "images_image_96": "images.image_96",
        "images_image_97": "images.image_97",
        "images_image_98": "images.image_98",
        "images_image_99": "images.image_99",
        "images_image_100": "images.image_100",
    }
    INPUT_SOCKET_TYPES = {
        "positive": "CONDITIONING",
        "negative": "CONDITIONING",
        "images_image_1": "IMAGE",
        "images_image_2": "IMAGE",
        "images_image_3": "IMAGE",
        "images_image_4": "IMAGE",
        "images_image_5": "IMAGE",
        "images_image_6": "IMAGE",
        "images_image_7": "IMAGE",
        "images_image_8": "IMAGE",
        "images_image_9": "IMAGE",
        "images_image_10": "IMAGE",
        "images_image_11": "IMAGE",
        "images_image_12": "IMAGE",
        "images_image_13": "IMAGE",
        "images_image_14": "IMAGE",
        "images_image_15": "IMAGE",
        "images_image_16": "IMAGE",
        "images_image_17": "IMAGE",
        "images_image_18": "IMAGE",
        "images_image_19": "IMAGE",
        "images_image_20": "IMAGE",
        "images_image_21": "IMAGE",
        "images_image_22": "IMAGE",
        "images_image_23": "IMAGE",
        "images_image_24": "IMAGE",
        "images_image_25": "IMAGE",
        "images_image_26": "IMAGE",
        "images_image_27": "IMAGE",
        "images_image_28": "IMAGE",
        "images_image_29": "IMAGE",
        "images_image_30": "IMAGE",
        "images_image_31": "IMAGE",
        "images_image_32": "IMAGE",
        "images_image_33": "IMAGE",
        "images_image_34": "IMAGE",
        "images_image_35": "IMAGE",
        "images_image_36": "IMAGE",
        "images_image_37": "IMAGE",
        "images_image_38": "IMAGE",
        "images_image_39": "IMAGE",
        "images_image_40": "IMAGE",
        "images_image_41": "IMAGE",
        "images_image_42": "IMAGE",
        "images_image_43": "IMAGE",
        "images_image_44": "IMAGE",
        "images_image_45": "IMAGE",
        "images_image_46": "IMAGE",
        "images_image_47": "IMAGE",
        "images_image_48": "IMAGE",
        "images_image_49": "IMAGE",
        "images_image_50": "IMAGE",
        "images_image_51": "IMAGE",
        "images_image_52": "IMAGE",
        "images_image_53": "IMAGE",
        "images_image_54": "IMAGE",
        "images_image_55": "IMAGE",
        "images_image_56": "IMAGE",
        "images_image_57": "IMAGE",
        "images_image_58": "IMAGE",
        "images_image_59": "IMAGE",
        "images_image_60": "IMAGE",
        "images_image_61": "IMAGE",
        "images_image_62": "IMAGE",
        "images_image_63": "IMAGE",
        "images_image_64": "IMAGE",
        "images_image_65": "IMAGE",
        "images_image_66": "IMAGE",
        "images_image_67": "IMAGE",
        "images_image_68": "IMAGE",
        "images_image_69": "IMAGE",
        "images_image_70": "IMAGE",
        "images_image_71": "IMAGE",
        "images_image_72": "IMAGE",
        "images_image_73": "IMAGE",
        "images_image_74": "IMAGE",
        "images_image_75": "IMAGE",
        "images_image_76": "IMAGE",
        "images_image_77": "IMAGE",
        "images_image_78": "IMAGE",
        "images_image_79": "IMAGE",
        "images_image_80": "IMAGE",
        "images_image_81": "IMAGE",
        "images_image_82": "IMAGE",
        "images_image_83": "IMAGE",
        "images_image_84": "IMAGE",
        "images_image_85": "IMAGE",
        "images_image_86": "IMAGE",
        "images_image_87": "IMAGE",
        "images_image_88": "IMAGE",
        "images_image_89": "IMAGE",
        "images_image_90": "IMAGE",
        "images_image_91": "IMAGE",
        "images_image_92": "IMAGE",
        "images_image_93": "IMAGE",
        "images_image_94": "IMAGE",
        "images_image_95": "IMAGE",
        "images_image_96": "IMAGE",
        "images_image_97": "IMAGE",
        "images_image_98": "IMAGE",
        "images_image_99": "IMAGE",
        "images_image_100": "IMAGE",
    }

    def __init__(
        self,
        node_id: str,
        positive: tuple[str, int],
        negative: tuple[str, int],
        images_image_1: tuple[str, int] | None = None,
        images_image_2: tuple[str, int] | None = None,
        images_image_3: tuple[str, int] | None = None,
        images_image_4: tuple[str, int] | None = None,
        images_image_5: tuple[str, int] | None = None,
        images_image_6: tuple[str, int] | None = None,
        images_image_7: tuple[str, int] | None = None,
        images_image_8: tuple[str, int] | None = None,
        images_image_9: tuple[str, int] | None = None,
        images_image_10: tuple[str, int] | None = None,
        images_image_11: tuple[str, int] | None = None,
        images_image_12: tuple[str, int] | None = None,
        images_image_13: tuple[str, int] | None = None,
        images_image_14: tuple[str, int] | None = None,
        images_image_15: tuple[str, int] | None = None,
        images_image_16: tuple[str, int] | None = None,
        images_image_17: tuple[str, int] | None = None,
        images_image_18: tuple[str, int] | None = None,
        images_image_19: tuple[str, int] | None = None,
        images_image_20: tuple[str, int] | None = None,
        images_image_21: tuple[str, int] | None = None,
        images_image_22: tuple[str, int] | None = None,
        images_image_23: tuple[str, int] | None = None,
        images_image_24: tuple[str, int] | None = None,
        images_image_25: tuple[str, int] | None = None,
        images_image_26: tuple[str, int] | None = None,
        images_image_27: tuple[str, int] | None = None,
        images_image_28: tuple[str, int] | None = None,
        images_image_29: tuple[str, int] | None = None,
        images_image_30: tuple[str, int] | None = None,
        images_image_31: tuple[str, int] | None = None,
        images_image_32: tuple[str, int] | None = None,
        images_image_33: tuple[str, int] | None = None,
        images_image_34: tuple[str, int] | None = None,
        images_image_35: tuple[str, int] | None = None,
        images_image_36: tuple[str, int] | None = None,
        images_image_37: tuple[str, int] | None = None,
        images_image_38: tuple[str, int] | None = None,
        images_image_39: tuple[str, int] | None = None,
        images_image_40: tuple[str, int] | None = None,
        images_image_41: tuple[str, int] | None = None,
        images_image_42: tuple[str, int] | None = None,
        images_image_43: tuple[str, int] | None = None,
        images_image_44: tuple[str, int] | None = None,
        images_image_45: tuple[str, int] | None = None,
        images_image_46: tuple[str, int] | None = None,
        images_image_47: tuple[str, int] | None = None,
        images_image_48: tuple[str, int] | None = None,
        images_image_49: tuple[str, int] | None = None,
        images_image_50: tuple[str, int] | None = None,
        images_image_51: tuple[str, int] | None = None,
        images_image_52: tuple[str, int] | None = None,
        images_image_53: tuple[str, int] | None = None,
        images_image_54: tuple[str, int] | None = None,
        images_image_55: tuple[str, int] | None = None,
        images_image_56: tuple[str, int] | None = None,
        images_image_57: tuple[str, int] | None = None,
        images_image_58: tuple[str, int] | None = None,
        images_image_59: tuple[str, int] | None = None,
        images_image_60: tuple[str, int] | None = None,
        images_image_61: tuple[str, int] | None = None,
        images_image_62: tuple[str, int] | None = None,
        images_image_63: tuple[str, int] | None = None,
        images_image_64: tuple[str, int] | None = None,
        images_image_65: tuple[str, int] | None = None,
        images_image_66: tuple[str, int] | None = None,
        images_image_67: tuple[str, int] | None = None,
        images_image_68: tuple[str, int] | None = None,
        images_image_69: tuple[str, int] | None = None,
        images_image_70: tuple[str, int] | None = None,
        images_image_71: tuple[str, int] | None = None,
        images_image_72: tuple[str, int] | None = None,
        images_image_73: tuple[str, int] | None = None,
        images_image_74: tuple[str, int] | None = None,
        images_image_75: tuple[str, int] | None = None,
        images_image_76: tuple[str, int] | None = None,
        images_image_77: tuple[str, int] | None = None,
        images_image_78: tuple[str, int] | None = None,
        images_image_79: tuple[str, int] | None = None,
        images_image_80: tuple[str, int] | None = None,
        images_image_81: tuple[str, int] | None = None,
        images_image_82: tuple[str, int] | None = None,
        images_image_83: tuple[str, int] | None = None,
        images_image_84: tuple[str, int] | None = None,
        images_image_85: tuple[str, int] | None = None,
        images_image_86: tuple[str, int] | None = None,
        images_image_87: tuple[str, int] | None = None,
        images_image_88: tuple[str, int] | None = None,
        images_image_89: tuple[str, int] | None = None,
        images_image_90: tuple[str, int] | None = None,
        images_image_91: tuple[str, int] | None = None,
        images_image_92: tuple[str, int] | None = None,
        images_image_93: tuple[str, int] | None = None,
        images_image_94: tuple[str, int] | None = None,
        images_image_95: tuple[str, int] | None = None,
        images_image_96: tuple[str, int] | None = None,
        images_image_97: tuple[str, int] | None = None,
        images_image_98: tuple[str, int] | None = None,
        images_image_99: tuple[str, int] | None = None,
        images_image_100: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.positive = positive
        self.negative = negative
        if images_image_1 is not None:
            self.images_image_1 = images_image_1
        if images_image_2 is not None:
            self.images_image_2 = images_image_2
        if images_image_3 is not None:
            self.images_image_3 = images_image_3
        if images_image_4 is not None:
            self.images_image_4 = images_image_4
        if images_image_5 is not None:
            self.images_image_5 = images_image_5
        if images_image_6 is not None:
            self.images_image_6 = images_image_6
        if images_image_7 is not None:
            self.images_image_7 = images_image_7
        if images_image_8 is not None:
            self.images_image_8 = images_image_8
        if images_image_9 is not None:
            self.images_image_9 = images_image_9
        if images_image_10 is not None:
            self.images_image_10 = images_image_10
        if images_image_11 is not None:
            self.images_image_11 = images_image_11
        if images_image_12 is not None:
            self.images_image_12 = images_image_12
        if images_image_13 is not None:
            self.images_image_13 = images_image_13
        if images_image_14 is not None:
            self.images_image_14 = images_image_14
        if images_image_15 is not None:
            self.images_image_15 = images_image_15
        if images_image_16 is not None:
            self.images_image_16 = images_image_16
        if images_image_17 is not None:
            self.images_image_17 = images_image_17
        if images_image_18 is not None:
            self.images_image_18 = images_image_18
        if images_image_19 is not None:
            self.images_image_19 = images_image_19
        if images_image_20 is not None:
            self.images_image_20 = images_image_20
        if images_image_21 is not None:
            self.images_image_21 = images_image_21
        if images_image_22 is not None:
            self.images_image_22 = images_image_22
        if images_image_23 is not None:
            self.images_image_23 = images_image_23
        if images_image_24 is not None:
            self.images_image_24 = images_image_24
        if images_image_25 is not None:
            self.images_image_25 = images_image_25
        if images_image_26 is not None:
            self.images_image_26 = images_image_26
        if images_image_27 is not None:
            self.images_image_27 = images_image_27
        if images_image_28 is not None:
            self.images_image_28 = images_image_28
        if images_image_29 is not None:
            self.images_image_29 = images_image_29
        if images_image_30 is not None:
            self.images_image_30 = images_image_30
        if images_image_31 is not None:
            self.images_image_31 = images_image_31
        if images_image_32 is not None:
            self.images_image_32 = images_image_32
        if images_image_33 is not None:
            self.images_image_33 = images_image_33
        if images_image_34 is not None:
            self.images_image_34 = images_image_34
        if images_image_35 is not None:
            self.images_image_35 = images_image_35
        if images_image_36 is not None:
            self.images_image_36 = images_image_36
        if images_image_37 is not None:
            self.images_image_37 = images_image_37
        if images_image_38 is not None:
            self.images_image_38 = images_image_38
        if images_image_39 is not None:
            self.images_image_39 = images_image_39
        if images_image_40 is not None:
            self.images_image_40 = images_image_40
        if images_image_41 is not None:
            self.images_image_41 = images_image_41
        if images_image_42 is not None:
            self.images_image_42 = images_image_42
        if images_image_43 is not None:
            self.images_image_43 = images_image_43
        if images_image_44 is not None:
            self.images_image_44 = images_image_44
        if images_image_45 is not None:
            self.images_image_45 = images_image_45
        if images_image_46 is not None:
            self.images_image_46 = images_image_46
        if images_image_47 is not None:
            self.images_image_47 = images_image_47
        if images_image_48 is not None:
            self.images_image_48 = images_image_48
        if images_image_49 is not None:
            self.images_image_49 = images_image_49
        if images_image_50 is not None:
            self.images_image_50 = images_image_50
        if images_image_51 is not None:
            self.images_image_51 = images_image_51
        if images_image_52 is not None:
            self.images_image_52 = images_image_52
        if images_image_53 is not None:
            self.images_image_53 = images_image_53
        if images_image_54 is not None:
            self.images_image_54 = images_image_54
        if images_image_55 is not None:
            self.images_image_55 = images_image_55
        if images_image_56 is not None:
            self.images_image_56 = images_image_56
        if images_image_57 is not None:
            self.images_image_57 = images_image_57
        if images_image_58 is not None:
            self.images_image_58 = images_image_58
        if images_image_59 is not None:
            self.images_image_59 = images_image_59
        if images_image_60 is not None:
            self.images_image_60 = images_image_60
        if images_image_61 is not None:
            self.images_image_61 = images_image_61
        if images_image_62 is not None:
            self.images_image_62 = images_image_62
        if images_image_63 is not None:
            self.images_image_63 = images_image_63
        if images_image_64 is not None:
            self.images_image_64 = images_image_64
        if images_image_65 is not None:
            self.images_image_65 = images_image_65
        if images_image_66 is not None:
            self.images_image_66 = images_image_66
        if images_image_67 is not None:
            self.images_image_67 = images_image_67
        if images_image_68 is not None:
            self.images_image_68 = images_image_68
        if images_image_69 is not None:
            self.images_image_69 = images_image_69
        if images_image_70 is not None:
            self.images_image_70 = images_image_70
        if images_image_71 is not None:
            self.images_image_71 = images_image_71
        if images_image_72 is not None:
            self.images_image_72 = images_image_72
        if images_image_73 is not None:
            self.images_image_73 = images_image_73
        if images_image_74 is not None:
            self.images_image_74 = images_image_74
        if images_image_75 is not None:
            self.images_image_75 = images_image_75
        if images_image_76 is not None:
            self.images_image_76 = images_image_76
        if images_image_77 is not None:
            self.images_image_77 = images_image_77
        if images_image_78 is not None:
            self.images_image_78 = images_image_78
        if images_image_79 is not None:
            self.images_image_79 = images_image_79
        if images_image_80 is not None:
            self.images_image_80 = images_image_80
        if images_image_81 is not None:
            self.images_image_81 = images_image_81
        if images_image_82 is not None:
            self.images_image_82 = images_image_82
        if images_image_83 is not None:
            self.images_image_83 = images_image_83
        if images_image_84 is not None:
            self.images_image_84 = images_image_84
        if images_image_85 is not None:
            self.images_image_85 = images_image_85
        if images_image_86 is not None:
            self.images_image_86 = images_image_86
        if images_image_87 is not None:
            self.images_image_87 = images_image_87
        if images_image_88 is not None:
            self.images_image_88 = images_image_88
        if images_image_89 is not None:
            self.images_image_89 = images_image_89
        if images_image_90 is not None:
            self.images_image_90 = images_image_90
        if images_image_91 is not None:
            self.images_image_91 = images_image_91
        if images_image_92 is not None:
            self.images_image_92 = images_image_92
        if images_image_93 is not None:
            self.images_image_93 = images_image_93
        if images_image_94 is not None:
            self.images_image_94 = images_image_94
        if images_image_95 is not None:
            self.images_image_95 = images_image_95
        if images_image_96 is not None:
            self.images_image_96 = images_image_96
        if images_image_97 is not None:
            self.images_image_97 = images_image_97
        if images_image_98 is not None:
            self.images_image_98 = images_image_98
        if images_image_99 is not None:
            self.images_image_99 = images_image_99
        if images_image_100 is not None:
            self.images_image_100 = images_image_100
