"""Compatibility layer for Runtime Security.

Canonical implementation:
    raglab.control.runtime_security

Do not add security logic here.
"""

from raglab.control import runtime_security as _canonical
from raglab.control.runtime_security import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_canonical, name)


def __dir__():
    return sorted(
        set(globals())
        | set(dir(_canonical))
    )