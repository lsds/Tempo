from tempo.core import index_expr as ie
from tempo.core.configs import ExecutionConfig
from tempo.core.datatypes import BackendTensorT, TensorId
from tempo.core.device import DeviceGroup
from tempo.core.dl_backend import DLBackend
from tempo.core.domain import Domain
from tempo.core.dtype import DataType
from tempo.core.shape import Shape
from tempo.runtime.tensor_store.tensor_store import RuntimeTensor
from tempo.utils import isl as isl_utils


class EvalSymbolRuntimeTensor(RuntimeTensor[BackendTensorT]):  # [BackendTensorT]
    def __init__(
        self,
        exec_cfg: ExecutionConfig,
        tensor_id: TensorId,
        shape: Shape,
        dtype: DataType,
        dev: DeviceGroup,
        domain: Domain,
        symbol: ie.Symbol,
        bound_defs: dict[ie.Symbol, ie.IntIndexValueLike],
    ) -> None:
        super().__init__(tensor_id)
        self.exec_cfg = exec_cfg
        self.backend = DLBackend.get_backend(exec_cfg.backend)
        self.dev = self.backend.to_backend_device_obj(dev)

        self.symbol = symbol
        self.dtype = dtype

        bend_dtype = self.backend.to_backend_datatype(dtype)

        self.symbol_idx = domain.find_variable_index(symbol)

        symbol_max_val = self._get_max_val(symbol, bound_defs)
        self.symbol_max_val = symbol_max_val + 1
        self._data_list = [
            self.backend.fast_int_lift(i, device=self.dev, dtype=bend_dtype)
            for i in range(symbol_max_val + 1)
        ]

        ## NOTE: we pre-populate the tensor store at compile time.
        # dom_max_vals: List[int] = [self._get_max_val(dim, bound_defs) for dim in domain.variables]
        # dom_max_vals = [v + 1 for v in dom_max_vals]

        # print(f"Computed max vals: {dom_max_vals} for domain: {domain}")

        # symbol_idx = domain.find_variable_index(symbol)
        ## Iterate the product of all possible values for each dimension.
        # for point in product(*[range(max_val) for max_val in dom_max_vals]):
        #    point_tuple = tuple(point)
        #    symbol_val = point_tuple[symbol_idx]
        #    self._data_dict[point_tuple] = self.backend.fast_int_lift(
        #        symbol_val, device=self.dev, dtype=bend_dtype
        #    )

    def all_int_fast_path(self, item: tuple[int | slice]) -> BackendTensorT:
        return self._data_list[item[self.symbol_idx]]  # type: ignore

    def __getitem__(self, item: tuple[int | slice]) -> BackendTensorT:
        return self._data_list[item[self.symbol_idx]]  # type: ignore

    def _get_max_val(
        self, dim: ie.Symbol, bound_defs: dict[ie.Symbol, ie.IntIndexValueLike]
    ) -> int:
        static_bound_defs = {k: v for k, v in bound_defs.items() if isinstance(v, int)}
        b = dim.as_bound()
        if b in bound_defs:
            b_val = bound_defs[b]
            if isinstance(b_val, int):
                return b_val
            else:
                b_val = isl_utils.int_index_val_max(  # type: ignore
                    dim, known_symbols=static_bound_defs
                )
                if b_val is None:
                    raise ValueError(f"Could not compute max value for {dim}.")

                assert isinstance(b_val, ie.ConstInt), "TODO: support D0 as max val?"
                return b_val.const
        else:
            return 1

    def all_int_fast_path_set(self, item: tuple[int, ...], value: BackendTensorT) -> None: ...

    def __setitem__(self, item: tuple[int | slice, ...], value: BackendTensorT) -> None: ...

    def mem_usage_bytes(self) -> int:
        return self.symbol_max_val * self.dtype.repr_bytes

    # Do not throw away values
    def flush(self) -> None: ...

    def deallocate_point(self, item: tuple[int | slice, ...]) -> None: ...

    def offload_point(self, item: tuple[int | slice, ...]) -> None: ...

    def fetch_point(self, item: tuple[int | slice, ...]) -> None: ...

    def deallocate_block(self, block: tuple[int | slice, ...]) -> None: ...

    def offload_block(self, block: tuple[int | slice, ...]) -> None: ...

    def fetch_block(self, block: tuple[int | slice, ...]) -> None: ...
