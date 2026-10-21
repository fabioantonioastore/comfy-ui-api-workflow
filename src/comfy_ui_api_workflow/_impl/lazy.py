import importlib
import importlib.abc
import importlib.util
import sys
from collections.abc import Iterable, Sequence
from importlib.machinery import ModuleSpec
from types import ModuleType


class _PublicModules(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Imports package.<name>... as the module package._impl.<name>... it stands for."""

    def __init__(self, package: str, names: Iterable[str]) -> None:
        self.prefixes = tuple(f"{package}.{name}" for name in names)
        self.package = package

    def find_spec(
        self, fullname: str, path: Sequence[str] | None = None, target: object = None
    ) -> ModuleSpec | None:
        if not any(
            fullname == prefix or fullname.startswith(f"{prefix}.")
            for prefix in self.prefixes
        ):
            return None
        private = f"{self.package}._impl{fullname.removeprefix(self.package)}"
        spec = importlib.util.find_spec(private)
        if spec is None:
            return None
        return importlib.util.spec_from_loader(
            fullname, self, is_package=spec.submodule_search_locations is not None
        )

    def create_module(self, spec: ModuleSpec) -> ModuleType:
        # The public name gets the module already loaded under _impl, so each
        # class exists once whichever path imported it.
        private = f"{self.package}._impl{spec.name.removeprefix(self.package)}"
        return importlib.import_module(private)

    def exec_module(self, module: ModuleType) -> None:
        pass


def public_modules(package: str, names: Iterable[str]) -> None:
    """Lets package.<name> be imported by its path, for each name the root exports."""
    finder = _PublicModules(package, names)
    if not any(
        isinstance(other, _PublicModules) and other.package == package
        for other in sys.meta_path
    ):
        # First in line, or the path finder would load a second copy of each
        # submodule from the __path__ of the module the public name stands for.
        sys.meta_path.insert(0, finder)
