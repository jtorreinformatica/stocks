"""
Pattern auto-discovery module.

Automatically imports all PatternDetector subclasses from this package.
To add a new pattern, simply create a new .py file in this directory
with a class that inherits from PatternDetector.
"""

import importlib
import pkgutil
from pathlib import Path

from .base import PatternDetector, PatternMatch  # noqa: F401

_all_detectors: list[type[PatternDetector]] | None = None


def get_all_detectors() -> list[type[PatternDetector]]:
    """
    Discover and return all PatternDetector subclasses in this package.
    Results are cached after the first call.
    """
    global _all_detectors
    if _all_detectors is not None:
        return _all_detectors

    package_dir = Path(__file__).parent

    # Import every module in this package
    for _importer, module_name, _is_pkg in pkgutil.iter_modules([str(package_dir)]):
        if module_name == "base":
            continue
        importlib.import_module(f".{module_name}", package=__name__)

    # Collect all concrete subclasses
    def _collect_subclasses(cls):
        result = []
        for sub in cls.__subclasses__():
            if not getattr(sub, "__abstractmethods__", set()):
                result.append(sub)
            result.extend(_collect_subclasses(sub))
        return result

    _all_detectors = _collect_subclasses(PatternDetector)
    return _all_detectors


def get_detector_by_name(name: str) -> PatternDetector | None:
    """Get an instantiated detector by its display name."""
    for detector_cls in get_all_detectors():
        instance = detector_cls()
        if instance.name == name:
            return instance
    return None


def get_all_detector_names() -> list[str]:
    """Return the display names of all available pattern detectors."""
    return [cls().name for cls in get_all_detectors()]
