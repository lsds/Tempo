import setuptools


def read_requirements(file: str) -> list[str]:
    """Returns content of given requirements file."""
    return [line for line in open(file) if not (line.startswith("#") or line.startswith("--"))]


install_requires = read_requirements("./requirements/requirements.txt")
setuptools.setup(
    # ext_modules=cythonize([
    #    "tempo/core/schedule/execution_schedule.pyx",
    #    #"tempo/core/symbol_dict.py",
    #    ], language_level="3"),
    # "src/**/*.py", language_level="3"
    install_requires=install_requires,
    extras_require={
        "envs": read_requirements("./requirements/requirements-envs.txt"),
        "dev": read_requirements("./requirements/requirements-dev.txt"),
        "examples": read_requirements("./requirements/requirements-examples.txt"),
        "llm": read_requirements("./requirements/requirements-llm.txt"),
    },
)
