"""Check atomic publication and preservation of an existing native cache."""

from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("seed_autotune", ROOT / "scripts/seed-autotune.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
PRESET = ROOT / "autotune/b300-tp4-static5.json"
CONCURRENT_STARTS = 8


def test_concurrent_start_publishes_one_complete_cache(tmp_path):
    with ThreadPoolExecutor(max_workers=CONCURRENT_STARTS) as executor:
        results = list(executor.map(lambda _: MODULE.seed(tmp_path, PRESET), range(CONCURRENT_STARTS)))
    assert sum(results) == 1
    cache = tmp_path / MODULE.NATIVE_NAMESPACE / "autotune_configs.json"
    assert cache.read_bytes() == PRESET.read_bytes()
    assert list(cache.parent.iterdir()) == [cache]
    cache.write_bytes(b"existing user tuning")
    assert not MODULE.seed(tmp_path, PRESET)
    assert cache.read_bytes() == b"existing user tuning"


def test_corrupt_preset_does_not_publish(tmp_path):
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_bytes(b"{}")
    cache_root = tmp_path / "cache"
    with pytest.raises(ValueError, match="checksum mismatch"):
        MODULE.seed(cache_root, corrupt)
    assert not cache_root.exists()
