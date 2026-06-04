"""Backward-compatible module alias for :mod:`selective_rag_rl.policies.bandit`."""

from importlib import import_module as _import_module
import sys as _sys

_impl = _import_module("selective_rag_rl.policies.bandit")
_sys.modules[__name__] = _impl
