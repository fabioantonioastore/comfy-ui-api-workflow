from typing import Literal, get_args

from ..base import BaseNode
from ..validation import (
    validate_bool,
    validate_choice,
    validate_not_empty,
    validate_range,
)


class SDPoseKeypointExtractor(BaseNode):
    __slots__ = ("model", "vae", "image", "batch_size", "bboxes")
    CATEGORY = "image/detection"
    OUTPUTS = {"keypoints": 0}
    RETURN_TYPES = ("POSE_KEYPOINT",)
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "vae": "VAE",
        "image": "IMAGE",
        "batch_size": "INT",
        "bboxes": "BOUNDING_BOX",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        vae: tuple[str, int],
        image: tuple[str, int],
        batch_size: int | tuple[str, int] = 16,
        bboxes: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.vae = vae
        self.image = image
        self.batch_size = batch_size
        if bboxes is not None:
            self.bboxes = bboxes
        self.__validate()

    def __validate(self) -> None:
        validate_range("batch_size", self.batch_size, 1, 10_000)


class SDPoseDrawKeypoints(BaseNode):
    __slots__ = (
        "keypoints",
        "draw_body",
        "draw_hands",
        "draw_face",
        "draw_feet",
        "stick_width",
        "face_point_size",
        "score_threshold",
        "draw_head",
    )
    CATEGORY = "image/detection"
    OUTPUTS = {"IMAGE": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_SOCKET_TYPES = {
        "keypoints": "POSE_KEYPOINT",
        "draw_body": "BOOLEAN",
        "draw_hands": "BOOLEAN",
        "draw_face": "BOOLEAN",
        "draw_feet": "BOOLEAN",
        "stick_width": "INT",
        "face_point_size": "INT",
        "score_threshold": "FLOAT",
        "draw_head": "BOOLEAN",
    }

    def __init__(
        self,
        node_id: str,
        keypoints: tuple[str, int],
        draw_body: bool | tuple[str, int] = True,
        draw_hands: bool | tuple[str, int] = True,
        draw_face: bool | tuple[str, int] = True,
        draw_feet: bool | tuple[str, int] = False,
        stick_width: int | tuple[str, int] = 4,
        face_point_size: int | tuple[str, int] = 3,
        score_threshold: float | tuple[str, int] = 0.3,
        draw_head: bool | tuple[str, int] = True,
    ) -> None:
        super().__init__(node_id=node_id)
        self.keypoints = keypoints
        self.draw_body = draw_body
        self.draw_hands = draw_hands
        self.draw_face = draw_face
        self.draw_feet = draw_feet
        self.stick_width = stick_width
        self.face_point_size = face_point_size
        self.score_threshold = score_threshold
        self.draw_head = draw_head
        self.__validate()

    def __validate(self) -> None:
        validate_bool("draw_body", self.draw_body)
        validate_bool("draw_hands", self.draw_hands)
        validate_bool("draw_face", self.draw_face)
        validate_bool("draw_feet", self.draw_feet)
        validate_range("stick_width", self.stick_width, 1, 10)
        validate_range("face_point_size", self.face_point_size, 1, 10)
        validate_range("score_threshold", self.score_threshold, 0.0, 1.0)
        validate_bool("draw_head", self.draw_head)


class SDPoseFaceBBoxes(BaseNode):
    __slots__ = ("keypoints", "scale", "force_square")
    CATEGORY = "image/detection"
    OUTPUTS = {"bboxes": 0}
    RETURN_TYPES = ("BOUNDING_BOX",)
    INPUT_SOCKET_TYPES = {
        "keypoints": "POSE_KEYPOINT",
        "scale": "FLOAT",
        "force_square": "BOOLEAN",
    }

    def __init__(
        self,
        node_id: str,
        keypoints: tuple[str, int],
        scale: float | tuple[str, int] = 1.5,
        force_square: bool | tuple[str, int] = True,
    ) -> None:
        super().__init__(node_id=node_id)
        self.keypoints = keypoints
        self.scale = scale
        self.force_square = force_square
        self.__validate()

    def __validate(self) -> None:
        validate_range("scale", self.scale, 1.0, 10.0)
        validate_bool("force_square", self.force_square)


RTDETR_detect_Class_Names = Literal[
    "all",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]
RTDETR_DETECT_CLASS_NAMES = frozenset(get_args(RTDETR_detect_Class_Names))


class RTDETR_detect(BaseNode):
    __slots__ = ("model", "image", "threshold", "class_name", "max_detections")
    CATEGORY = "image/detection"
    OUTPUTS = {"bboxes": 0}
    RETURN_TYPES = ("BOUNDING_BOX",)
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "image": "IMAGE",
        "threshold": "FLOAT",
        "class_name": "COMBO",
        "max_detections": "INT",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        image: tuple[str, int],
        threshold: float | tuple[str, int] = 0.5,
        class_name: RTDETR_detect_Class_Names | tuple[str, int] = "all",
        max_detections: int | tuple[str, int] = 100,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.image = image
        self.threshold = threshold
        self.class_name = class_name
        self.max_detections = max_detections
        self.__validate()

    def __validate(self) -> None:
        validate_range("threshold", self.threshold)
        validate_choice("class_name", self.class_name, RTDETR_DETECT_CLASS_NAMES)
        validate_range("max_detections", self.max_detections)


class DrawBBoxes(BaseNode):
    __slots__ = ("bboxes", "image")
    CATEGORY = "image/detection"
    OUTPUTS = {"out_image": 0}
    RETURN_TYPES = ("IMAGE",)
    INPUT_SOCKET_TYPES = {"bboxes": "BOUNDING_BOX", "image": "IMAGE"}

    def __init__(
        self,
        node_id: str,
        bboxes: tuple[str, int],
        image: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.bboxes = bboxes
        if image is not None:
            self.image = image


class SAM3_Detect(BaseNode):
    __slots__ = (
        "model",
        "image",
        "threshold",
        "refine_iterations",
        "individual_masks",
        "conditioning",
        "bboxes",
        "positive_coords",
        "negative_coords",
    )
    CATEGORY = "image/detection"
    OUTPUTS = {"masks": 0, "bboxes": 1}
    RETURN_TYPES = ("MASK", "BOUNDING_BOX")
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "image": "IMAGE",
        "threshold": "FLOAT",
        "refine_iterations": "INT",
        "individual_masks": "BOOLEAN",
        "conditioning": "CONDITIONING",
        "bboxes": "BOUNDING_BOX",
        "positive_coords": "STRING",
        "negative_coords": "STRING",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        image: tuple[str, int],
        threshold: float | tuple[str, int] = 0.5,
        refine_iterations: int | tuple[str, int] = 2,
        individual_masks: bool | tuple[str, int] = False,
        conditioning: tuple[str, int] | None = None,
        bboxes: tuple[str, int] | None = None,
        positive_coords: tuple[str, int] | None = None,
        negative_coords: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.image = image
        self.threshold = threshold
        self.refine_iterations = refine_iterations
        self.individual_masks = individual_masks
        if conditioning is not None:
            self.conditioning = conditioning
        if bboxes is not None:
            self.bboxes = bboxes
        if positive_coords is not None:
            self.positive_coords = positive_coords
        if negative_coords is not None:
            self.negative_coords = negative_coords
        self.__validate()

    def __validate(self) -> None:
        validate_range("threshold", self.threshold, 0.0, 1.0)
        validate_range("refine_iterations", self.refine_iterations, 0, 5)
        validate_bool("individual_masks", self.individual_masks)


class SAM3_VideoTrack(BaseNode):
    __slots__ = (
        "images",
        "model",
        "detection_threshold",
        "max_objects",
        "detect_interval",
        "initial_mask",
        "conditioning",
    )
    CATEGORY = "image/detection"
    OUTPUTS = {"track_data": 0}
    RETURN_TYPES = ("SAM3_TRACK_DATA",)
    INPUT_SOCKET_TYPES = {
        "images": "IMAGE",
        "model": "MODEL",
        "detection_threshold": "FLOAT",
        "max_objects": "INT",
        "detect_interval": "INT",
        "initial_mask": "MASK",
        "conditioning": "CONDITIONING",
    }

    def __init__(
        self,
        node_id: str,
        images: tuple[str, int],
        model: tuple[str, int],
        detection_threshold: float | tuple[str, int] = 0.5,
        max_objects: int | tuple[str, int] = 4,
        detect_interval: int | tuple[str, int] = 1,
        initial_mask: tuple[str, int] | None = None,
        conditioning: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.images = images
        self.model = model
        self.detection_threshold = detection_threshold
        self.max_objects = max_objects
        self.detect_interval = detect_interval
        if initial_mask is not None:
            self.initial_mask = initial_mask
        if conditioning is not None:
            self.conditioning = conditioning
        self.__validate()

    def __validate(self) -> None:
        validate_range("detection_threshold", self.detection_threshold, 0.0, 1.0)
        validate_range("max_objects", self.max_objects, 0, 64)
        validate_range("detect_interval", self.detect_interval, 1)


class SAM3_TrackPreview(BaseNode):
    __slots__ = ("track_data", "opacity", "fps", "images")
    CATEGORY = "image/detection"
    INPUT_SOCKET_TYPES = {
        "track_data": "SAM3_TRACK_DATA",
        "opacity": "FLOAT",
        "fps": "FLOAT",
        "images": "IMAGE",
    }

    def __init__(
        self,
        node_id: str,
        track_data: tuple[str, int],
        opacity: float | tuple[str, int] = 0.5,
        fps: float | tuple[str, int] = 24.0,
        images: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.track_data = track_data
        self.opacity = opacity
        self.fps = fps
        if images is not None:
            self.images = images
        self.__validate()

    def __validate(self) -> None:
        validate_range("opacity", self.opacity, 0.0, 1.0)
        validate_range("fps", self.fps, 1.0, 120.0)


class SAM3_TrackToMask(BaseNode):
    __slots__ = ("track_data", "object_indices")
    CATEGORY = "image/detection"
    OUTPUTS = {"masks": 0}
    RETURN_TYPES = ("MASK",)
    INPUT_SOCKET_TYPES = {"track_data": "SAM3_TRACK_DATA", "object_indices": "STRING"}

    def __init__(
        self,
        node_id: str,
        track_data: tuple[str, int],
        object_indices: str | tuple[str, int] = "",
    ) -> None:
        super().__init__(node_id=node_id)
        self.track_data = track_data
        self.object_indices = object_indices


MediaPipeFaceLandmarker_Detector_Variants = Literal["short", "full", "both"]
MediaPipeFaceLandmarker_Missing_Frame_Fallbacks = Literal[
    "empty", "previous", "interpolate"
]
MEDIAPIPEFACELANDMARKER_DETECTOR_VARIANTS = frozenset(
    get_args(MediaPipeFaceLandmarker_Detector_Variants)
)
MEDIAPIPEFACELANDMARKER_MISSING_FRAME_FALLBACKS = frozenset(
    get_args(MediaPipeFaceLandmarker_Missing_Frame_Fallbacks)
)


class MediaPipeFaceLandmarker(BaseNode):
    __slots__ = (
        "face_detection_model",
        "image",
        "detector_variant",
        "num_faces",
        "min_confidence",
        "missing_frame_fallback",
    )
    CATEGORY = "image/detection"
    OUTPUTS = {"face_landmarks": 0, "bboxes": 1}
    RETURN_TYPES = ("FACE_LANDMARKS", "BOUNDING_BOX")
    INPUT_SOCKET_TYPES = {
        "face_detection_model": "FACE_DETECTION_MODEL",
        "image": "IMAGE",
        "detector_variant": "COMBO",
        "num_faces": "INT",
        "min_confidence": "FLOAT",
        "missing_frame_fallback": "COMBO",
    }

    def __init__(
        self,
        node_id: str,
        face_detection_model: tuple[str, int],
        image: tuple[str, int],
        detector_variant: (
            MediaPipeFaceLandmarker_Detector_Variants | tuple[str, int]
        ) = "short",
        num_faces: int | tuple[str, int] = 1,
        min_confidence: float | tuple[str, int] = 0.5,
        missing_frame_fallback: (
            MediaPipeFaceLandmarker_Missing_Frame_Fallbacks | tuple[str, int]
        ) = "empty",
    ) -> None:
        super().__init__(node_id=node_id)
        self.face_detection_model = face_detection_model
        self.image = image
        self.detector_variant = detector_variant
        self.num_faces = num_faces
        self.min_confidence = min_confidence
        self.missing_frame_fallback = missing_frame_fallback
        self.__validate()

    def __validate(self) -> None:
        validate_choice(
            "detector_variant",
            self.detector_variant,
            MEDIAPIPEFACELANDMARKER_DETECTOR_VARIANTS,
        )
        validate_range("num_faces", self.num_faces, 0, 16)
        validate_range("min_confidence", self.min_confidence, 0.0, 1.0)
        validate_choice(
            "missing_frame_fallback",
            self.missing_frame_fallback,
            MEDIAPIPEFACELANDMARKER_MISSING_FRAME_FALLBACKS,
        )


class SAM3DBody_Loader(BaseNode):
    __slots__ = ("model_file",)
    CATEGORY = "image/detection"
    OUTPUTS = {"sam3d_body_model": 0}
    RETURN_TYPES = ("SAM3D_BODY_MODEL",)
    INPUT_SOCKET_TYPES = {"model_file": "COMBO"}

    def __init__(self, node_id: str, model_file: str | tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.model_file = model_file
        self.__validate()

    def __validate(self) -> None:
        validate_not_empty("model_file", self.model_file)


class SAM3DBody_Predict(BaseNode):
    __slots__ = (
        "sam3d_body_model",
        "image",
        "fov",
        "run_hand_refinement",
        "batch_size",
        "track_data",
        "bboxes",
    )
    CATEGORY = "image/detection"
    OUTPUTS = {"mhr_pose_data": 0}
    RETURN_TYPES = ("MHR_POSE_DATA",)
    INPUT_SOCKET_TYPES = {
        "sam3d_body_model": "SAM3D_BODY_MODEL",
        "image": "IMAGE",
        "run_hand_refinement": "BOOLEAN",
        "fov": "FLOAT",
        "batch_size": "INT",
        "track_data": "SAM3_TRACK_DATA",
        "bboxes": "BOUNDING_BOX",
    }

    def __init__(
        self,
        node_id: str,
        sam3d_body_model: tuple[str, int],
        image: tuple[str, int],
        fov: float | tuple[str, int],
        run_hand_refinement: bool | tuple[str, int] = True,
        batch_size: int | tuple[str, int] = 64,
        track_data: tuple[str, int] | None = None,
        bboxes: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.sam3d_body_model = sam3d_body_model
        self.image = image
        self.fov = fov
        self.run_hand_refinement = run_hand_refinement
        self.batch_size = batch_size
        if track_data is not None:
            self.track_data = track_data
        if bboxes is not None:
            self.bboxes = bboxes
        self.__validate()

    def __validate(self) -> None:
        validate_range("fov", self.fov)
        validate_bool("run_hand_refinement", self.run_hand_refinement)
        validate_range("batch_size", self.batch_size, 1, 512)


class SAM3DBody_FaceExpression(BaseNode):
    __slots__ = (
        "sam3d_body_model",
        "mhr_pose_data",
        "image",
        "strength",
        "mouth_strength",
        "eye_strength",
        "brow_strength",
        "input_threshold",
        "blendshape_smooth_window",
    )
    CATEGORY = "image/detection"
    OUTPUTS = {"mhr_pose_data": 0}
    RETURN_TYPES = ("MHR_POSE_DATA",)
    INPUT_SOCKET_TYPES = {
        "sam3d_body_model": "SAM3D_BODY_MODEL",
        "mhr_pose_data": "MHR_POSE_DATA",
        "image": "IMAGE",
        "strength": "FLOAT",
        "mouth_strength": "FLOAT",
        "eye_strength": "FLOAT",
        "brow_strength": "FLOAT",
        "input_threshold": "FLOAT",
        "blendshape_smooth_window": "INT",
    }

    def __init__(
        self,
        node_id: str,
        sam3d_body_model: tuple[str, int],
        mhr_pose_data: tuple[str, int],
        image: tuple[str, int],
        strength: float | tuple[str, int] = 1.0,
        mouth_strength: float | tuple[str, int] = 1.0,
        eye_strength: float | tuple[str, int] = 2.0,
        brow_strength: float | tuple[str, int] = 2.0,
        input_threshold: float | tuple[str, int] = 0.02,
        blendshape_smooth_window: int | tuple[str, int] = 7,
    ) -> None:
        super().__init__(node_id=node_id)
        self.sam3d_body_model = sam3d_body_model
        self.mhr_pose_data = mhr_pose_data
        self.image = image
        self.strength = strength
        self.mouth_strength = mouth_strength
        self.eye_strength = eye_strength
        self.brow_strength = brow_strength
        self.input_threshold = input_threshold
        self.blendshape_smooth_window = blendshape_smooth_window
        self.__validate()

    def __validate(self) -> None:
        validate_range("strength", self.strength, 0.0, 4.0)
        validate_range("mouth_strength", self.mouth_strength, 0.0, 4.0)
        validate_range("eye_strength", self.eye_strength, 0.0, 4.0)
        validate_range("brow_strength", self.brow_strength, 0.0, 4.0)
        validate_range("input_threshold", self.input_threshold, 0.0, 0.5)
        validate_range("blendshape_smooth_window", self.blendshape_smooth_window, 1, 31)


SAM3DBody_Smooth_Methods = Literal["gaussian", "savgol"]
SAM3DBODY_SMOOTH_METHODS = frozenset(get_args(SAM3DBody_Smooth_Methods))


class SAM3DBody_Smooth(BaseNode):
    __slots__ = (
        "mhr_pose_data",
        "strength",
        "method",
        "window",
        "rotation_threshold_degrees",
    )
    CATEGORY = "image/detection"
    OUTPUTS = {"mhr_pose_data": 0}
    RETURN_TYPES = ("MHR_POSE_DATA",)
    INPUT_SOCKET_TYPES = {
        "mhr_pose_data": "MHR_POSE_DATA",
        "strength": "FLOAT",
        "method": "COMBO",
        "window": "INT",
        "rotation_threshold_degrees": "FLOAT",
    }

    def __init__(
        self,
        node_id: str,
        mhr_pose_data: tuple[str, int],
        strength: float | tuple[str, int] = 1.0,
        method: SAM3DBody_Smooth_Methods | tuple[str, int] = "savgol",
        window: int | tuple[str, int] = 7,
        rotation_threshold_degrees: float | tuple[str, int] = 30.0,
    ) -> None:
        super().__init__(node_id=node_id)
        self.mhr_pose_data = mhr_pose_data
        self.strength = strength
        self.method = method
        self.window = window
        self.rotation_threshold_degrees = rotation_threshold_degrees
        self.__validate()

    def __validate(self) -> None:
        validate_range("strength", self.strength, 0.0, 1.0)
        validate_choice("method", self.method, SAM3DBODY_SMOOTH_METHODS)
        validate_range("window", self.window, 1, 51)
        validate_range(
            "rotation_threshold_degrees", self.rotation_threshold_degrees, 0.0, 90.0
        )
