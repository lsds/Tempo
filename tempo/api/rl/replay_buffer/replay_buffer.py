from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from tempo.api.recurrent_tensor import MaybeRecurrentTensor, RecurrentTensor
from tempo.api.rl.replay_buffer import udfs
from tempo.api.rl.replay_buffer.replay_buffer_desc import ReplayBufferDesc
from tempo.api.rl.replay_buffer.replay_buffer_registry import ReplayBufferRegistry
from tempo.core.domain import DomainLike
from tempo.core.dtype import DataType
from tempo.core.global_objects import get_active_exec_cfg
from tempo.core.shape import Shape
from tempo.core.symbolic_tensor import SymbolicTensor


@dataclass(frozen=True)
class ReplayBuffer:
    _replay_buffer_desc: ReplayBufferDesc

    @staticmethod
    def make(
        max_size: int,
        # TODO do we actually need to pass in the shapes and dtypes ahead of time?
        item_shapes: list[Shape],
        item_dtypes: list[DataType],
        storage_type: str = "deque",
        **kwargs: Any,
    ) -> ReplayBuffer:
        """Creates and register a replay buffer in the context replay buffer registry.

        Args:
            max_size (int): Maximum size of the replay buffer.
            item_shapes (List[Shape]): Shapes of the items in the replay buffer.
            item_dtypes (List[DataType]): Data types of the items in the replay buffer.
            **kwargs: Additional arguments for the replay buffer.

        Returns:
            buffer: A symbolic object that may be used as a replay buffer.

        """

        exec_cfg = get_active_exec_cfg()

        desc_ = ReplayBufferRegistry.get_replay_buffer_description(
            storage_type, max_size, item_shapes, item_dtypes, exec_cfg, **kwargs
        )

        return ReplayBuffer(desc_)

    def insert(
        self,
        data: tuple[MaybeRecurrentTensor, ...],
        domain: DomainLike = None,
        # do_insert: Optional[MaybeRecurrentTensor] = None, #TODO optional
        **kwargs: Any,
    ) -> RecurrentTensor:
        desc = udfs.get_insert_udf_desc(self._replay_buffer_desc)

        raised_data = [RecurrentTensor.lift(d)._underlying for d in data]

        # assert that the type of data is correct
        for i, d in enumerate(raised_data):
            # assert with error message

            if d.shape != self._replay_buffer_desc.ctx.item_shapes[i].as_static():
                raise ValueError(
                    f"Shape of data at index {i} does not match the expected shape."
                    f"Expected: {self._replay_buffer_desc.ctx.item_shapes[i]}, got: {d.shape}"
                )
            if d.dtype != self._replay_buffer_desc.ctx.item_dtypes[i]:
                raise ValueError(
                    f"Data type of data at index {i} does not match the expected data type."
                    f"Expected: {self._replay_buffer_desc.ctx.item_dtypes[i]}, got: {d.dtype}"
                )

        # if do_insert is not None:
        #    raised_do_insert = RecurrentTensor.lift_to_rt(do_insert)._underlying
        #    raised_data.append(raised_do_insert)

        # TODO symbolic?
        (insertion_token,) = SymbolicTensor.udf(desc, raised_data, domain=domain)
        return RecurrentTensor(insertion_token)

    def sample(
        self,
        # TODO sample expression of some kind. Possibly a RecurrentTensor itself
        domain: DomainLike,
        token: MaybeRecurrentTensor | None = None,
        **kwargs: Any,
    ) -> Sequence[RecurrentTensor]:
        desc = udfs.get_sample_udf_desc(self._replay_buffer_desc)

        inputs = []
        if token is not None:
            raised_token = RecurrentTensor.lift(token)._underlying
            inputs.append(raised_token)

        symbolic_items = SymbolicTensor.udf(desc, inputs, domain=domain)
        return [RecurrentTensor(rt) for rt in symbolic_items]
