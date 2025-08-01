from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Set, Type

from tempo.core import tensor_ops as top
from tempo.core.compilation_ctx import CompilationCtx
from tempo.utils import logger

log = logger.get_logger(__name__)


# @dataclass
# class MatchResult:
#    """Result of a match operation, containing information needed for replacement."""
#    def __init__(self, **kwargs):
#        self.kwargs = kwargs
#
#    def __getitem__(self, key: str):
#        #return getattr(self, key)
#        #return self.kwargs[key]
#
#
#    def get(self, key: str, default=None):
#        #return getattr(self, key, default)
#        return self.kwargs.get(key, default)


class MatchReplacer(ABC):
    """Abstract base class for match and replace operations in algebraic optimization."""

    @classmethod
    def transform_name(cls) -> str:
        """The name of this transformation. Defaults to class name with 'Optimization' removed."""
        class_name = cls.__name__
        if class_name.endswith("Optimization"):
            return class_name[: -len("Optimization")]
        return class_name

    @abstractmethod
    def match(self, ctx: CompilationCtx, op: top.TensorOp) -> Optional[Any]:
        """
        Check if this optimization can be applied to the given operation.

        Args:
            ctx: Compilation context
            op: Operation to check

        Returns:
            MatchResult if the optimization can be applied, None otherwise
        """
        pass

    @abstractmethod
    def replace(self, ctx: CompilationCtx, op: top.TensorOp, match_result: Any) -> None:
        """
        Apply the optimization transformation.

        Args:
            ctx: Compilation context
            op: Operation to transform
            match_result: Result from the match phase

        Returns:
            True if the transformation was applied successfully, False otherwise
        """
        pass


class MatchReplacerRegistry:
    """Registry for match and replace operations organized by operation type."""

    def __init__(self) -> None:
        self._registry: Dict[Type[top.TensorOp], list[MatchReplacer]] = {}

    def register(self, op_type: Type[top.TensorOp], replacer: MatchReplacer) -> None:
        """Register a match replacer for a specific operation type."""
        if op_type not in self._registry:
            self._registry[op_type] = []
        self._registry[op_type].append(replacer)

    def get_op_match_replacers(self, op: top.TensorOp) -> list[MatchReplacer]:
        """Get all registered replacers for an operation."""
        results = self._registry.get(type(op), [])

        # names = [r.transform_name for r in results]
        # log.info("Found %s match replacers for %s: %s", len(results), type(op), names)

        return results

    def get_all_registered_optimizations(self) -> Set[Type[MatchReplacer]]:
        """Get all registered optimizations."""
        return {type(o) for r in self._registry.values() for o in r}
