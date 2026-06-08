import os
from typing import Any, Generator

import huggingface_hub
import pytest
from huggingface_hub import _CACHED_NO_EXIST, try_to_load_from_cache
from huggingface_hub.utils import EntryNotFoundError
from transformers.utils import hub


def _try_cache(repo_id: str, filename: str, kwargs: dict[str, Any]) -> str | object | None:
    return try_to_load_from_cache(
        repo_id,
        filename,
        cache_dir=kwargs.get("cache_dir"),
        revision=kwargs.get("revision"),
        repo_type=kwargs.get("repo_type"),
    )


def _cache_first_hf_hub_download(repo_id: str, filename: str, **kwargs: Any) -> str:
    if kwargs.get("force_download"):
        return huggingface_hub.hf_hub_download(repo_id, filename, **kwargs)

    subfolder = kwargs.get("subfolder")
    full_filename = f"{subfolder}/{filename}" if subfolder else filename
    cached = _try_cache(repo_id, full_filename, kwargs)
    if cached is _CACHED_NO_EXIST:
        # `transformers` treats `EntryNotFoundError` like an online 404; without this mapping, a
        # file cached as missing surfaces as a fatal connection error when the Hub is unreachable.
        raise EntryNotFoundError(f"{full_filename} is cached as non-existent in {repo_id}.")
    if cached is not None:
        return cached
    return huggingface_hub.hf_hub_download(repo_id, filename, **kwargs)


def _cache_first_snapshot_download(repo_id: str, **kwargs: Any) -> str:
    allow_patterns = kwargs.get("allow_patterns")
    if kwargs.get("force_download") or not allow_patterns:
        return huggingface_hub.snapshot_download(repo_id, **kwargs)

    # `transformers` passes exact filenames as patterns and re-resolves each file from the cache
    # afterwards: knowing the state of every file (`.no_exist` entries are non-None) is enough to
    # resolve the snapshot offline, preserving the returned-path contract without any network call.
    if all(_try_cache(repo_id, filename, kwargs) is not None for filename in allow_patterns):
        return huggingface_hub.snapshot_download(repo_id, **{**kwargs, "local_files_only": True})
    return huggingface_hub.snapshot_download(repo_id, **kwargs)


@pytest.fixture(autouse=True, scope="session")
def _hf_cache_first() -> Generator[None, None, None]:
    """Serve Hugging Face Hub files from the local cache instead of the network.

    CI runners share egress IPs across tenants, so anonymous Hub calls randomly hit HTTP 429 no
    matter how few requests we make. With the cache restored by `actions/cache`, the test suite
    needs no network at all; files missing from the cache still fall back to a real download.
    """
    if os.environ.get("CI") != "true":
        yield
        return

    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(hub, "hf_hub_download", _cache_first_hf_hub_download)
        patcher.setattr(hub, "snapshot_download", _cache_first_snapshot_download)
        yield
