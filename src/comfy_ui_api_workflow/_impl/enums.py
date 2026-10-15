from enum import Enum


# Behaves like StrEnum, which only exists from Python 3.11 (the project supports 3.10).
class OutputTypes(str, Enum):
    MODEL = "MODEL"
    CONDITIONING = "CONDITIONING"
    LATENT = "LATENT"
    CLIP = "CLIP"
    VAE = "VAE"

    def __str__(self) -> str:
        return self.value
