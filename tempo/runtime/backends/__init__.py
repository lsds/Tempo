backend_modules = [
    "tempo.runtime.backends.pytorch.pytorch_backend",
    "tempo.runtime.backends.jax.jax_backend",
    "tempo.runtime.backends.numpy.numpy_backend",
    # "tempo.runtime.backends.pytorch.pytorch_backend_with_jax_codegen",
]

for module_name in backend_modules:
    try:
        __import__(f"{module_name}", globals(), locals(), ["*"], 0)
    except ImportError as e:
        print(f"Could not import {module_name}: {e}")
