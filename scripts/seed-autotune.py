"""Seed the measured configuration's native cache without replacing user tuning."""

import hashlib
import logging
import os
from pathlib import Path
import tempfile

PRESET = Path("/opt/tinfoil/autotune/b300-tp4-static5.json")
PRESET_SHA256 = "9c83b7daebe5d8e258bccefef3e6984e3eedbd9d02aa006e8ef360a29f52afe1"
CONFIG_HASH = "76da1e93f493c957e4d61f325d5e4ce41264a35ee8164944f852f0b10f9810f3"
NATIVE_NAMESPACE = Path("flashinfer_autotune_cache/0.7.0.post1/103a") / CONFIG_HASH
LOGGER = logging.getLogger("tinfoil.autotune")


def seed(cache_root: Path, preset: Path = PRESET) -> bool:
    data = preset.read_bytes()
    if hashlib.sha256(data).hexdigest() != PRESET_SHA256:
        raise ValueError("FlashInfer preset checksum mismatch")
    destination = cache_root / NATIVE_NAMESPACE / "autotune_configs.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return False
    with tempfile.NamedTemporaryFile(dir=destination.parent) as temporary:
        temporary.write(data)
        temporary.flush()
        try:
            # A completed file becomes visible at once, including concurrent starts.
            os.link(temporary.name, destination)
        except FileExistsError:
            return False
    LOGGER.info("Seeded measured B300 TP4 FlashInfer cache: %s", destination)
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    seed(Path(os.environ.get("VLLM_CACHE_ROOT", "/root/.cache/vllm")))
