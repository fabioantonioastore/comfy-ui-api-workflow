from ..lazy import submodules_of

__getattr__ = submodules_of(__name__)
