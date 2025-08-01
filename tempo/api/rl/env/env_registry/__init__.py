env_modules = [
    "gymnasium_envs",
    "brax_envs",
    "jumanji_envs",
    "trivial_envs",
    "torch_catch_envs",
    # 'gymnax_envs',
    "cule_envs",
]

for module_name in env_modules:
    try:
        __import__(
            f"tempo.api.rl.env.env_registry.{module_name}",
            globals(),
            locals(),
            ["*"],
            0,
        )
    except ImportError as e:
        print(f"Could not import {module_name}: {e}")
