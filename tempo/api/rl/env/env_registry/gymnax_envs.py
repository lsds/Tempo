# def _gymnax_env_builder(name: str, ne: Optional[int], **kwargs: Any) -> gym.Env:
#    gymnax_kwargs = {**kwargs}
#    gymnax_kwargs.update({"disable_env_checker": True, "autoreset": False})
#
#    max_steps = kwargs.get("max_episode_steps", None)
#    for k in [
#        "autoreset",
#        "auto_reset",
#        "disable_env_checker",
#        "max_episode_steps",
#        "episode_length",
#    ]:
#        if k in gymnax_kwargs:
#            log.warning(f"{k} is not supported for brax environments")
#            gymnax_kwargs.pop(k)
#
#    import gymnax
#    from gymnax.wrappers import GymnaxToGymWrapper, GymnaxToVectorGymWrapper
#
#    # TODO better way to handle max_episode_steps
#    env, params = gymnax.make(name, **gymnax_kwargs)
#    if max_steps is not None:
#        params.max_steps_in_episode = max_steps
#    # TODO: as far as I can tell, no way to specify the platform to use....
#    env = (
#        GymnaxToVectorGymWrapper(env, num_envs=ne, params=params)
#        if ne
#        else GymnaxToGymWrapper(env, params=params)
#    )
#
#    return env


# if env_set == "gymnax":
#            import jax
#
#            from_backend = lambda x: jax.numpy.from_dlpack(x)
#            wrappers = [
#                partial(
#                    ToBackendTensorTWrapper,
#                    to_backend_tensor=to_backend,
#                    from_backend_tensor=from_backend,
#                ),
#                *(wrappers or []),
#            ]
#            if max_episode_steps is not None:
#                kwargs["episode_length"] = max_episode_steps
