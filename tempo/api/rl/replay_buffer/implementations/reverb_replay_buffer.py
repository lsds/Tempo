# from typing import Any, Dict, Sequence
#
# from tempo.api.rl.replay_buffer.replay_buffer_registry import ReplayBufferRegistry
# from tempo.api.rl.replay_buffer.runtime_replay_buffer_interface import (
#    ReplayBufferCtx,
#    RuntimeReplayBufferInterface,
# )
# from tempo.core.dtype import DataType, dtypes
# from tempo.runtime.backends.backend import Backend
#
# try:
#    import reverb
#    import tensorflow as tf
#
#    # TODO when we have a tf backend, move this there
#    # Map TensorFlow data types to Tempo data types
#    TF_TO_TEMPO_DTYPES_DICT: Dict[tf.DType, DataType] = {
#        tf.bool: dtypes.bool_,
#        tf.float16: dtypes.float16,
#        tf.float32: dtypes.float32,
#        tf.float64: dtypes.float64,
#        tf.uint8: dtypes.uint8,
#        tf.uint16: dtypes.uint16,
#        tf.uint32: dtypes.uint32,
#        tf.uint64: dtypes.uint64,
#        tf.int8: dtypes.int8,
#        tf.int16: dtypes.int16,
#        tf.int32: dtypes.int32,
#        tf.int64: dtypes.int64,
#    }
#
#    # Inverse mapping from Tempo data types back to TensorFlow data types
#    Tempo_TO_TF_DTYPES: Dict[DataType, tf.DType] = {
#        v: k for k, v in TF_TO_TEMPO_DTYPES_DICT.items()
#    }
#
#    # TODO make this more configurable
#    # TODO (alan): Use Trajectroy dataset instead of sample and insert
#    class ReverbReplayBuffer(RuntimeReplayBufferInterface):
#        DEFAULT_TABLE_NAME = "default"
#        DEFAULT_SAMPLER = (
#            reverb.selectors.Uniform()
#        )  # .Prioritized(priority_exponent=1)
#        DEFAULT_REMOVER = reverb.selectors.Fifo()
#        Default_RATE_LIMITER = reverb.rate_limiters.MinSize(1)
#
#        def __init__(self, ctx: ReplayBufferCtx):
#            super().__init__()
#
#            self.ctx = ctx
#            self.backend = Backend.get_backend(self.ctx.exec_cfg.backend)
#
#            self.table_name = ReverbReplayBuffer.DEFAULT_TABLE_NAME
#
#            self.server = reverb.Server(
#                tables=[
#                    reverb.Table(
#                        name=self.table_name,
#                        sampler=ReverbReplayBuffer.DEFAULT_SAMPLER,
#                        remover=ReverbReplayBuffer.DEFAULT_REMOVER,
#                        max_size=ctx.max_size,
#                        rate_limiter=ReverbReplayBuffer.Default_RATE_LIMITER,
#                    )
#                ]
#            )
#            self.client = reverb.Client(f"localhost:{self.server.port}")
#
#            converted_dtypes = [Tempo_TO_TF_DTYPES[dtype] for dtype in ctx.item_dtypes]
#            converted_shapes = [
#                tf.TensorShape(list(shape.as_static()._shape))
#                for shape in ctx.item_shapes
#            ]
#
#            # Reuse trajectory writer for inserts
#            self.writer = self.client.trajectory_writer(
#                num_keep_alive_refs=1, validate_items=False
#            )
#
#            # Reuse dataset for sampling
#            self.dataset = reverb.TimestepDataset(
#                server_address=f"localhost:{self.server.port}",
#                table=self.table_name,
#                dtypes=converted_dtypes,
#                shapes=converted_shapes,
#                max_in_flight_samples_per_worker=1 * 3,  # For single-sample fetches
#            )
#            self.iterator = iter(self.dataset)
#
#            self.batch_dataset = None
#            self.batch_iterator = None
#
#        def insert(self, data: Sequence[Any]) -> None:
#            self.writer.append(data)
#            self.writer.create_item(
#                table=self.table_name,
#                priority=1.0,
#                trajectory=self.writer.history,
#            )
#            self.writer.flush()
#
#        def insert_batched(self, data: Sequence[Any]) -> None:
#            bs = data[0].shape[0]  # type: ignore
#
#            for i in range(bs):
#                item = {j: d[i] for j, d in enumerate(data)}
#
#                self.writer.append(item)
#            self.writer.create_item(
#                table=self.table_name,
#                priority=1.0,
#                trajectory={i: self.writer.history[i][:] for i in range(len(data))}, # type: ignore
#            )
#            self.writer.flush()
#
#        def sample(self) -> Sequence[Any]:
#            sample = next(self.iterator)
#            data = sample.data  # This will be a tuple of tensors
#            return data
#
#        def sample_batched(self, num_samples: int) -> Sequence[Any]:
#            if self.batch_iterator is None:
#                batched_dataset = reverb.TimestepDataset(
#                    server_address=f"localhost:{self.server.port}",
#                    table=self.table_name,
#                    dtypes=self.ctx.item_dtypes,
#                    shapes=self.ctx.item_shapes,
#                    max_in_flight_samples_per_worker=num_samples * 3,
#                )
#                self.batch_dataset = batched_dataset.batch(num_samples)
#                self.batch_iterator = iter(self.batch_dataset)
#            sample = next(self.batch_iterator)
#            data = sample.data  # This will be a tuple of batched tensors
#            return data
#
#            return
#
#        # def insert(self, data: Tuple[Any]) -> bool:
#        #    self.writer_client.insert(data, priorities={self.table_name: 1.0})
#        #    return True
#
#        # def insert_many(self, data: Tuple[Tuple[Any]]) -> bool:
#        #    # https://github.com/google-deepmind/reverb/issues/78
#        #    # no way to batch insert as of now
#        #    for d in data:
#        #        self.insert(d)
#        #    return True
#
#        # def sample(self) -> Tuple[Any]:
#        #    samples = self.reader_client.sample(self.table_name, num_samples=1)
#        #    result: Tuple[Any] = list(samples)[0][0].data
#        #    return result
#
#        # def sample_many(self, num_samples: int) -> List[Tuple[Any]]:
#        #    samples = self.reader_client.sample(self.table_name, num_samples=num_samples)
#        #    result = [sample[0].data for sample in samples]
#        #    return result
#
#    ReplayBufferRegistry.register_replay_buffer("reverb", ReverbReplayBuffer)
#
# except ImportError as e:
#    print(f"Failed to register Reverb replay buffer: {e}")
#
