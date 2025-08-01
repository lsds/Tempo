import gym as gym_old
import gymnasium as gym

from tempo.core.dtype import DataType, dtypes
from tempo.core.shape import StaticShape


def space_to_shape(space: gym.spaces.Space) -> StaticShape:
    if isinstance(space, gym.spaces.Discrete):
        return StaticShape(())
    elif isinstance(space, (gym.spaces.Box, gym_old.spaces.box.Box)):
        return StaticShape(tuple(space.shape))
    elif isinstance(space, gym.spaces.MultiBinary):
        return StaticShape(tuple(space.shape))
    elif isinstance(space, gym.spaces.MultiDiscrete):
        return StaticShape(tuple(space.shape))
    # elif isinstance(space, gym.spaces.Tuple):
    #    return tuple(space_to_shape(s) for s in space.spaces)
    # elif isinstance(space, gym.spaces.Dict):
    #    return {k: space_to_shape(s) for k, s in space.spaces.items()}
    else:
        raise NotImplementedError(f"Unsupported space type: {type(space)}")


def space_to_dtype(space: gym.spaces.Space) -> DataType:
    np_dtype = space.dtype
    assert np_dtype is not None
    return dtypes.from_np(np_dtype)
