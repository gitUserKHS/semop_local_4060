"""Backward-compatible domain contract imports.

The ownership lives in :mod:`semop.kernel.contracts` so foundational composition
code does not import the domain package and trigger its adapter registry.
"""

from ..contracts import DomainInstance, TypedDomainAdapter

__all__ = ["DomainInstance", "TypedDomainAdapter"]
