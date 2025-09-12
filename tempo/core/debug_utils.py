"""Debug utilities for Tempo, including traceback collection.

This module is designed to avoid circular imports by providing debugging
functionality that can be imported by other core modules.
"""

import traceback

# Global flag to control creation traceback collection
_collect_creation_tracebacks: bool = False

DISABLED_MSG = ["TRACEBACKS DISABLED"]


def set_collect_creation_tracebacks(enabled: bool) -> None:
    """Enable or disable collection of creation tracebacks for debugging."""
    global _collect_creation_tracebacks
    _collect_creation_tracebacks = enabled


def get_creation_traceback() -> list[str]:
    """Get creation traceback if enabled, static constant otherwise."""
    if _collect_creation_tracebacks:
        return traceback.format_stack()[:-1]
    return DISABLED_MSG


def collect_traceback_if_enabled() -> list[str]:
    """Collect traceback only if collection is enabled, otherwise return empty list."""
    if _collect_creation_tracebacks:
        return traceback.format_stack()[:-1]
    return []
