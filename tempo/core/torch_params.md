# Information about parameters passed to `torch.jit.trace()` and `torch.compile()`

## `torch.jit.trace()` parameters

### func

A function which will be run with `example_inputs`. The function can only take arguments and return
values which are tensors.

In `dataflow_graph.py`, this is `self.execute`.

### example_inputs

Optional tuple of inputs which will be passed to the function while tracing.

`None` (default)

Either this or `example_kwarg_inputs` must be specified.

The resulting trace can be run wtih inputs of different types and shapes, assuming traced operations
support those types and shapes.

### check_trace

`True` (default) checks if same inputs run through the traced code produce the same outputs.

`False` if your network contains non-deterministic operations or you are sure the network is correct
despite a checker failure.

### check_inputs

Optional list of tuples of input arguments to check the trace against expected behaviour.

If unspecified, original example_inputs are used for checking.

### check_tolerance

Optional float denoting floating-point compaison tolerance to use in checking procedure. Can also
relax checker strictness if results diverge numerically for a known reason, such as operator fusion.

### strict

`True` (default) runs the tracer in strict mode.

`False` means the tracer will record mutable container types (lists/dicts), and you must be sure
that the container being used is a constant structure, and not used as controk flow (if, for)
conditions.

### example_kwarg_inputs

An optional pack (dict) of keyword arguments of example inputs that will be passed to the function
while tracing.

`None` (default)

Either this or `example_inputs` must be specified.

The keys of the dict must match with the traced function's arguments name (raises runtime
exception if keys do not match).

## `torch.compile()` parameters

### model

The model is the function which we aim to optimise with `torch.compile()`.

In `dataflow_graph.py`, this is the result of calling `torch.jit.trace()`.

### fullgraph

`False` (default) attempts to discover any compileable regions in the function.

`True` requires that the entire function being optimised is capturable in a
single graph, otherwise an error is thrown.

### dynamic

`None` (default) means torch will automatically detect if dynamism has occurred and compile
a more dynamic kernel upon recompile.

`True` means that torch attempts to make a kernel which is as dynamic as possible to avoid
recompilation when sizes change. Comes with the potential for overspecialization.

`False` means that torch will never create dynamic kernels, and instead always specialise.

### backend

You can use the below in a `.py` file to list the non-experimental backends available:

```
import torch

res = torch._dynamo.list_backends()

print(res)
```

`'inductor'` (default) uses the TorchInductor backend. Balances performance and overhead.

`'cudagraphs'` enables use of CUDA graphs with AOT Autograd. CUDA Graphs provide a way to define
workflows as graphs rather than single operations. They may reduce overhead by launching multiple
GPU operations through a single CPU operation.

Note: running TorchInductor on GPU requires Triton to be installed (below installs for CUDA 11.7):

    pip install torchtriton --extra-index-url "https://download.pytorch.org/whl/nightly/cu117"

`'onnxrt'` uses ONNX Runtime for training on CPU/GPU. This converts the PyTorch model into ONNX,
which then allows for porting models between different frameworks.

If ONNX is not already installed, use this command to install it:

    pip install onnx onnxscript onnxruntime

`'openxla'` is a `aot-autograd` backend of torch.compile. `aot-autograd` will attempt to save some
states for potential backward. `torch.no_grad` will help `aot-autograd` understand that it is being
executed in a inference context.

In the pytorch/xla 2.0 release, PyTorch/XLA provided an experimental backend for the TorchDynamo for
both inference and training. The way that XLA bridge works is that Dynamo will provide a TorchFX
graph when it recognizes a model pattern and PyTorch/XLA will use existing Lazy Tensor technology
to compile the FX graph and return the compiled function. User will likely see better inference
performance by putting the inference execution in a `torch.no_grad` context.

`'openxla_eval'` backend can be used directly without torch.no_grad, since openxla_eval is not an
`aot-autograd` backend and only works for inference.

`'tvm'` uses Apache TVM for inference optimisations. TVM is  an open deep learning compiler stack
for CPU, GPU, and specialized accelerators.

### mode

`'max-autotune'` leverages Triton-based matrix multiplications and enables CUDA graphs (asynchronous
execution model which reduces time taken to dispatch work on the GPU and improves GPU runtime,
according to NVIDIA).

`'default'` uses neither of the above optimisations

`'reduce-overhead'` is recommended for small batch sizes, at the cost of higher GPU memory usage.
There are certain circumstances where CUDA graphs are not available (see [here](https://pytorch.org/docs/stable/generated/torch.compile.html) for more info)

`'max-autotune-no-cudagraphs'`, as the name suggests, does not use CUDA graphs.

To display the configurations which each mode sets, you can run the below in a `.py` file:

```
import torch

res = torch._inductor.list_mode_options()

print(res)
```

### options

There are many, many options which can be used when calling `torch.compile()`.

To display all of these options, you can run this code in a `.py` file:

```
import torch

res = torch._inductor.list_options()

print(res)
```

Or, simply reference this list:

- 'TYPE_CHECKING'
- 'debug'
- 'debug_check_inf_and_nan'
- 'disable_progress'
- 'verbose_progress'
- 'fx_graph_cache'
- 'cpp_wrapper'
- 'dce'
- 'static_weight_shapes'
- 'size_asserts'
- 'nan_asserts'
- 'pick_loop_orders'
- 'inplace_buffers'
- 'allow_buffer_reuse'
- 'memory_planning'
- 'memory_pool'
- 'benchmark_harness'
- 'epilogue_fusion'
- 'epilogue_fusion_first'
- 'pattern_matcher'
- 'post_grad_custom_pre_pass'
- 'post_grad_custom_post_pass'
- 'pre_grad_custom_pass'
- 'split_cat_fx_passes'
- 'efficient_conv_bn_eval_fx_passes'
- 'group_fusion'
- 'batch_fusion'
- 'pre_grad_fusion_options'
- 'post_grad_fusion_options'
- 'reorder_for_locality'
- 'dynamic_scale_rblock'
- 'force_fuse_int_mm_with_mul'
- 'use_mixed_mm'
- 'force_mixed_mm'
- 'reorder_for_compute_comm_overlap'
- 'reorder_for_compute_comm_overlap_passes'
- 'estimate_op_runtime'
- 'intra_node_bw'
- 'inter_node_bw'
- 'max_autotune'
- 'max_autotune_pointwise'
- 'max_autotune_gemm'
- 'max_autotune_gemm_backends'
- 'unbacked_symint_fallback'
- 'search_autotune_cache'
- 'save_args'
- 'autotune_in_subproc'
- 'autotune_multi_device'
- 'coordinate_descent_tuning'
- 'coordinate_descent_check_all_directions'
- 'coordinate_descent_search_radius'
- 'layout_optimization'
- 'force_layout_optimization'
- 'keep_output_stride'
- 'warn_mix_layout'
- 'realize_reads_threshold'
- 'realize_bytes_threshold'
- 'realize_acc_reads_threshold'
- 'fallback_random'
- 'implicit_fallbacks'
- 'aggressive_fusion'
- 'debug_fusion'
- 'benchmark_fusion'
- 'enabled_metric_tables'
- 'max_fusion_size'
- 'max_pointwise_cat_inputs'
- 'unroll_reductions_threshold'
- 'comment_origin'
- 'conv_1x1_as_mm'
- 'split_reductions'
- 'benchmark_kernel'
- 'constant_and_index_propagation'
- 'always_keep_tensor_constants'
- 'assert_indirect_indexing'
- 'joint_graph_constant_folding'
- 'debug_index_asserts'
- 'is_nightly_or_source'
- 'developer_warnings'
- 'worker_start_method'
- 'compile_threads'
- 'global_cache_dir'
- 'kernel_name_max_ops'
- 'shape_padding'
- 'permute_fusion'
- 'profiler_mark_wrapper_call'
- 'generate_intermediate_hooks'
- 'debug_ir_traceback'
- '_raise_error_for_testing'
- '_profile_var'
- 'profile_bandwidth'
- 'profile_bandwidth_regex'
- 'profile_bandwidth_output'
- 'disable_cpp_codegen'
- 'freezing'
- 'freezing_discard_parameters'
- 'cpp.threads'
- 'cpp.no_redundant_loops'
- 'cpp.dynamic_threads'
- 'cpp.simdlen'
- 'cpp.min_chunk_size'
- 'cpp.cxx'
- 'cpp.enable_kernel_profile'
- 'cpp.weight_prepack'
- 'cpp.inject_relu_bug_TESTING_ONLY'
- 'cpp.inject_log1p_bug_TESTING_ONLY'
- 'cpp.vec_isa_ok'
- 'cpp.descriptive_names'
- 'cpp.max_horizontal_fusion_size'
- 'cpp.fallback_scatter_reduce_sum'
- 'cpp.enable_unsafe_math_opt_flag'
- 'triton.cudagraphs'
- 'triton.cudagraph_trees'
- 'triton.slow_path_cudagraph_asserts'
- 'triton.cudagraph_trees_history_recording'
- 'triton.fast_path_cudagraph_asserts'
- 'triton.skip_cudagraph_warmup'
- 'triton.debug_sync_graph'
- 'triton.debug_sync_kernel'
- 'triton.dense_indexing'
- 'triton.max_tiles'
- 'triton.autotune_pointwise'
- 'triton.autotune_cublasLt'
- 'triton.tiling_prevents_pointwise_fusion'
- 'triton.tiling_prevents_reduction_fusion'
- 'triton.unique_kernel_names'
- 'triton.descriptive_names'
- 'triton.persistent_reductions'
- 'triton.divisible_by_16'
- 'triton.max_block'
- 'triton.store_cubin'
- 'triton.spill_threshold'
- 'triton.inject_relu_bug_TESTING_ONLY'
- 'aot_inductor.output_path'
- 'aot_inductor.debug_compile'
- 'aot_inductor.abi_compatible'
- 'aot_inductor.serialized_in_spec'
- 'aot_inductor.serialized_out_spec'
- 'cuda.arch'
- 'cuda.version'
- 'cuda.compile_opt_level'
- 'cuda.enable_cuda_lto'
- 'cuda.enable_ptxas_info'
- 'cuda.enable_debug_info'
- 'cuda.use_fast_math'
- 'cuda.cutlass_dir'
- 'cuda.cutlass_max_profiling_configs'
- 'cuda.cuda_cxx'
- 'cuda.cutlass_only_evt_capable_ops'
- 'trace.enabled'
- 'trace.debug_dir'
- 'trace.debug_log'
- 'trace.info_log'
- 'trace.fx_graph'
- 'trace.fx_graph_transformed'
- 'trace.ir_pre_fusion'
- 'trace.ir_post_fusion'
- 'trace.output_code'
- 'trace.graph_diagram'
- 'trace.draw_orig_fx_graph'
- 'trace.dot_graph_shape'
- 'trace.compile_profile'
- 'trace.upload_tar'
- '_save_config_ignore'
