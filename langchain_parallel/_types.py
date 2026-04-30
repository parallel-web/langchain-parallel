"""Back-compat shim for the renamed :mod:`langchain_parallel.types` module.

The settings classes (``ExcerptSettings``, ``FetchPolicy``, ``FullContentSettings``,
``SourcePolicy``) live in :mod:`langchain_parallel.types` as of 0.4.1. This shim
re-exports them so any code that imported from ``langchain_parallel._types``
keeps working; new code should import from ``langchain_parallel.types`` (or
the package root) directly.
"""

from langchain_parallel.types import (
    ExcerptSettings,
    FetchPolicy,
    FullContentSettings,
    SourcePolicy,
)

__all__ = [
    "ExcerptSettings",
    "FetchPolicy",
    "FullContentSettings",
    "SourcePolicy",
]
