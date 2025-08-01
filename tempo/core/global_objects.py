import threading
from typing import Any, Dict, List, Mapping, Optional, Tuple

from tempo.core import index_expr as ie
from tempo.core.configs import ExecutionConfig
from tempo.core.dependence_graph import PDG
from tempo.core.domain import Domain, DomainLike
from tempo.core.op_tags import GROUP_TAG, NO_DEDUP_TAG, REGION_TAG

active_domain = threading.local()
active_dg_loc_storage = threading.local()
active_exec_config_storage = threading.local()
active_tags: List[Tuple[str, Any]] = []


def set_active_domain(domain: DomainLike) -> None:
    global active_domain
    dom = Domain.from_(domain, none_is_empty=True)

    active_domain.domain = dom


def get_active_domain() -> Domain:
    global active_domain

    if not hasattr(active_domain, "domain"):
        raise RuntimeError("No active domain found")

    return active_domain.domain  # type: ignore


def get_active_domain_or_empty() -> Domain:
    try:
        return get_active_domain()
    except RuntimeError:
        return Domain.empty()


def set_active_dg(dg: Optional[PDG]) -> None:
    global active_dg_loc_storage
    active_dg_loc_storage.dg = dg


def has_active_dg() -> bool:
    global active_dg_loc_storage
    return hasattr(active_dg_loc_storage, "dg") and active_dg_loc_storage.dg is not None


def get_static_bounds_or_empty() -> Mapping[ie.Symbol, int]:
    if not has_active_dg():
        return {}
    return get_active_dg().static_bounds


def get_dynamic_bounds_or_empty() -> Mapping[ie.Symbol, ie.IntIndexValue]:
    if not has_active_dg():
        return {}
    return get_active_dg().dynamic_bounds


def get_active_dg() -> PDG:
    global active_dg_loc_storage
    if active_dg_loc_storage.dg is None:
        raise RuntimeError("No active RDG found")
    return active_dg_loc_storage.dg  # type: ignore


def set_active_config(config: Optional[ExecutionConfig]) -> None:
    global active_exec_config_storage
    active_exec_config_storage.config = config


def get_active_exec_cfg() -> ExecutionConfig:
    global active_exec_config_storage
    if (
        not hasattr(active_exec_config_storage, "config")
        or active_exec_config_storage.config is None
    ):
        return ExecutionConfig.default()
    return active_exec_config_storage.config  # type: ignore


def get_active_tags() -> Mapping[str, Any]:
    dict_tags: Dict[str, Any] = {}

    for tag_type, tag in active_tags:
        existing_tags = ()
        if tag_type in dict_tags:
            existing_tags = dict_tags[tag_type]
        dict_tags[tag_type] = existing_tags + (tag,)
    return dict_tags


class TagCtxManager(object):
    def __init__(self, tag_type: str, tag: Optional[str] = None) -> None:
        self.tag_type = tag_type
        self.tag = tag

    def __enter__(self) -> None:
        active_tags.append((self.tag_type, self.tag))

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        active_tags.pop()


class RegionTagCtxManager(TagCtxManager):
    def __init__(self, tag: str) -> None:
        super().__init__(REGION_TAG, tag)


class NoDedupCtxManager(TagCtxManager):
    def __init__(self) -> None:
        super().__init__(NO_DEDUP_TAG)


class GroupTagCtxManager(TagCtxManager):
    def __init__(self, tag: str) -> None:
        super().__init__(GROUP_TAG, tag)


class DomainCtxManager(object):
    def __init__(self, domain: DomainLike) -> None:
        self.domain = domain

    def __enter__(self) -> None:
        set_active_domain(self.domain)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        set_active_domain(None)
