from typing import List, Type


def get_all_subclasses(cls: Type) -> List[Type]:
    """Recursively get all subclasses of a given class.

    Args:
        cls (Type): The class to get the subclasses of.

    Returns:
        List[Type]: A list of all subclasses of the given class.

    """
    subclasses = cls.__subclasses__()
    all_subclasses = []
    for subclass in subclasses:
        all_subclasses.append(subclass)
        all_subclasses.extend(get_all_subclasses(subclass))
    return all_subclasses
