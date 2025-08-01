replay_modules = [
    "deque_replay_buffer",
    "reverb_replay_buffer",
]

for module_name in replay_modules:
    try:
        __import__(
            f"tempo.api.rl.replay_buffer.implementations.{module_name}",
            globals(),
            locals(),
            ["*"],
            0,
        )
    except ImportError as e:
        print(f"Could not import {module_name}: {e}")
