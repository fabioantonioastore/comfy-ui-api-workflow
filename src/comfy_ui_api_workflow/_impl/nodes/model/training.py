from typing import Literal, get_args

from ..base import BaseNode
from ..validation import (
    validate_bool,
    validate_choice,
    validate_not_empty,
    validate_range,
)

TrainLoraNode_Optimizers = Literal["AdamW", "Adam", "SGD", "RMSprop"]
TrainLoraNode_Loss_Functions = Literal["MSE", "L1", "Huber", "SmoothL1"]
TrainLoraNode_Training_Dtypes = Literal["bf16", "fp32", "none"]
TrainLoraNode_Lora_Dtypes = Literal["bf16", "fp32"]
TrainLoraNode_Algorithms = Literal["LoRA", "LoHa", "LoKr", "OFT"]
TRAINLORANODE_OPTIMIZERS = frozenset(get_args(TrainLoraNode_Optimizers))
TRAINLORANODE_LOSS_FUNCTIONS = frozenset(get_args(TrainLoraNode_Loss_Functions))
TRAINLORANODE_TRAINING_DTYPES = frozenset(get_args(TrainLoraNode_Training_Dtypes))
TRAINLORANODE_LORA_DTYPES = frozenset(get_args(TrainLoraNode_Lora_Dtypes))
TRAINLORANODE_ALGORITHMS = frozenset(get_args(TrainLoraNode_Algorithms))


class TrainLoraNode(BaseNode):
    __slots__ = (
        "model",
        "latents",
        "positive",
        "batch_size",
        "grad_accumulation_steps",
        "steps",
        "learning_rate",
        "rank",
        "optimizer",
        "loss_function",
        "seed",
        "training_dtype",
        "lora_dtype",
        "quantized_backward",
        "algorithm",
        "gradient_checkpointing",
        "checkpoint_depth",
        "offloading",
        "existing_lora",
        "bucket_mode",
        "bypass_mode",
    )
    CATEGORY = "model/training"
    OUTPUTS = {"lora": 0, "loss_map": 1, "steps": 2}
    RETURN_TYPES = ("LORA_MODEL", "LOSS_MAP", "INT")
    INPUT_SOCKET_TYPES = {
        "model": "MODEL",
        "latents": "LATENT",
        "positive": "CONDITIONING",
        "batch_size": "INT",
        "grad_accumulation_steps": "INT",
        "steps": "INT",
        "learning_rate": "FLOAT",
        "rank": "INT",
        "optimizer": "COMBO",
        "loss_function": "COMBO",
        "seed": "INT",
        "training_dtype": "COMBO",
        "lora_dtype": "COMBO",
        "quantized_backward": "BOOLEAN",
        "algorithm": "COMBO",
        "gradient_checkpointing": "BOOLEAN",
        "checkpoint_depth": "INT",
        "offloading": "BOOLEAN",
        "existing_lora": "COMBO",
        "bucket_mode": "BOOLEAN",
        "bypass_mode": "BOOLEAN",
    }

    def __init__(
        self,
        node_id: str,
        model: tuple[str, int],
        latents: tuple[str, int],
        positive: tuple[str, int],
        batch_size: int | tuple[str, int] = 1,
        grad_accumulation_steps: int | tuple[str, int] = 1,
        steps: int | tuple[str, int] = 16,
        learning_rate: float | tuple[str, int] = 0.0005,
        rank: int | tuple[str, int] = 8,
        optimizer: TrainLoraNode_Optimizers | tuple[str, int] = "AdamW",
        loss_function: TrainLoraNode_Loss_Functions | tuple[str, int] = "MSE",
        seed: int | tuple[str, int] = 0,
        training_dtype: TrainLoraNode_Training_Dtypes | tuple[str, int] = "bf16",
        lora_dtype: TrainLoraNode_Lora_Dtypes | tuple[str, int] = "bf16",
        quantized_backward: bool | tuple[str, int] = False,
        algorithm: TrainLoraNode_Algorithms | tuple[str, int] = "LoRA",
        gradient_checkpointing: bool | tuple[str, int] = True,
        checkpoint_depth: int | tuple[str, int] = 1,
        offloading: bool | tuple[str, int] = False,
        existing_lora: str | tuple[str, int] = "[None]",
        bucket_mode: bool | tuple[str, int] = False,
        bypass_mode: bool | tuple[str, int] = False,
    ) -> None:
        super().__init__(node_id=node_id)
        self.model = model
        self.latents = latents
        self.positive = positive
        self.batch_size = batch_size
        self.grad_accumulation_steps = grad_accumulation_steps
        self.steps = steps
        self.learning_rate = learning_rate
        self.rank = rank
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.seed = seed
        self.training_dtype = training_dtype
        self.lora_dtype = lora_dtype
        self.quantized_backward = quantized_backward
        self.algorithm = algorithm
        self.gradient_checkpointing = gradient_checkpointing
        self.checkpoint_depth = checkpoint_depth
        self.offloading = offloading
        self.existing_lora = existing_lora
        self.bucket_mode = bucket_mode
        self.bypass_mode = bypass_mode
        self.__validate()

    def __validate(self) -> None:
        validate_range("batch_size", self.batch_size, 1, 10_000)
        validate_range("grad_accumulation_steps", self.grad_accumulation_steps, 1, 1024)
        validate_range("steps", self.steps, 1, 100_000)
        validate_range("learning_rate", self.learning_rate, 1e-07, 1.0)
        validate_range("rank", self.rank, 1, 128)
        validate_choice("optimizer", self.optimizer, TRAINLORANODE_OPTIMIZERS)
        validate_choice(
            "loss_function", self.loss_function, TRAINLORANODE_LOSS_FUNCTIONS
        )
        validate_range("seed", self.seed, 0, 18_446_744_073_709_551_615)
        validate_choice(
            "training_dtype", self.training_dtype, TRAINLORANODE_TRAINING_DTYPES
        )
        validate_choice("lora_dtype", self.lora_dtype, TRAINLORANODE_LORA_DTYPES)
        validate_bool("quantized_backward", self.quantized_backward)
        validate_choice("algorithm", self.algorithm, TRAINLORANODE_ALGORITHMS)
        validate_bool("gradient_checkpointing", self.gradient_checkpointing)
        validate_range("checkpoint_depth", self.checkpoint_depth, 1, 5)
        validate_bool("offloading", self.offloading)
        validate_not_empty("existing_lora", self.existing_lora)
        validate_bool("bucket_mode", self.bucket_mode)
        validate_bool("bypass_mode", self.bypass_mode)


class LossGraphNode(BaseNode):
    __slots__ = ("loss", "filename_prefix")
    CATEGORY = "model/training"
    INPUT_SOCKET_TYPES = {"loss": "LOSS_MAP", "filename_prefix": "STRING"}

    def __init__(
        self,
        node_id: str,
        loss: tuple[str, int],
        filename_prefix: str | tuple[str, int] = "loss_graph",
    ) -> None:
        super().__init__(node_id=node_id)
        self.loss = loss
        self.filename_prefix = filename_prefix


class MakeTrainingDataset(BaseNode):
    __slots__ = ("images", "vae", "clip", "texts")
    CATEGORY = "model/training"
    OUTPUTS = {"latents": 0, "conditioning": 1}
    RETURN_TYPES = ("LATENT", "CONDITIONING")
    INPUT_SOCKET_TYPES = {
        "images": "IMAGE",
        "vae": "VAE",
        "clip": "CLIP",
        "texts": "STRING",
    }

    def __init__(
        self,
        node_id: str,
        images: tuple[str, int],
        vae: tuple[str, int],
        clip: tuple[str, int],
        texts: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(node_id=node_id)
        self.images = images
        self.vae = vae
        self.clip = clip
        if texts is not None:
            self.texts = texts


class SaveTrainingDataset(BaseNode):
    __slots__ = ("latents", "conditioning", "folder_name", "shard_size")
    CATEGORY = "model/training"
    INPUT_SOCKET_TYPES = {
        "latents": "LATENT",
        "conditioning": "CONDITIONING",
        "folder_name": "STRING",
        "shard_size": "INT",
    }

    def __init__(
        self,
        node_id: str,
        latents: tuple[str, int],
        conditioning: tuple[str, int],
        folder_name: str | tuple[str, int] = "training_dataset",
        shard_size: int | tuple[str, int] = 1000,
    ) -> None:
        super().__init__(node_id=node_id)
        self.latents = latents
        self.conditioning = conditioning
        self.folder_name = folder_name
        self.shard_size = shard_size
        self.__validate()

    def __validate(self) -> None:
        validate_range("shard_size", self.shard_size, 1, 100_000)


class LoadTrainingDataset(BaseNode):
    __slots__ = ("folder_name",)
    CATEGORY = "model/training"
    OUTPUTS = {"latents": 0, "conditioning": 1}
    RETURN_TYPES = ("LATENT", "CONDITIONING")
    INPUT_SOCKET_TYPES = {"folder_name": "COMBO"}

    def __init__(self, node_id: str, folder_name: str | tuple[str, int]) -> None:
        super().__init__(node_id=node_id)
        self.folder_name = folder_name
        self.__validate()

    def __validate(self) -> None:
        validate_not_empty("folder_name", self.folder_name)


class ResolutionBucket(BaseNode):
    __slots__ = ("latents", "conditioning")
    CATEGORY = "model/training"
    OUTPUTS = {"latents": 0, "conditioning": 1}
    RETURN_TYPES = ("LATENT", "CONDITIONING")
    INPUT_SOCKET_TYPES = {"latents": "LATENT", "conditioning": "CONDITIONING"}

    def __init__(
        self, node_id: str, latents: tuple[str, int], conditioning: tuple[str, int]
    ) -> None:
        super().__init__(node_id=node_id)
        self.latents = latents
        self.conditioning = conditioning
