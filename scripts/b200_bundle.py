"""Verify and install configuration-specific native tuning and kernel caches."""

import hashlib
import json
import logging
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile

SCHEMA = "tinfoil-kernel-bundle-v1"
POLICY_RECORD = "flashinfer.profiling_policy.v1"
IDENTITY_FIELDS = {"gpu", "model", "configuration", "precision", "software", "measurement_policy"}
REQUIRED_POLICY_FIELDS = {"timer", "timer_boundary", "warmup", "repeat", "l2_cache_policy", "cuda_graph_profile_replays"}
LOGGER = logging.getLogger("b200_bundle")


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def identity_key(identity):
    if set(identity) != IDENTITY_FIELDS or any(not value for value in identity.values()):
        raise ValueError("The bundle compatibility identity is incomplete")
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def checked_path(root, relative):
    name = PurePosixPath(relative)
    if not name.parts or name.is_absolute() or any(part in (".", "..") for part in name.parts):
        raise ValueError("Bundle paths must be contained relative paths")
    if name.as_posix() != relative:
        raise ValueError("Bundle paths must be normalized")
    current = root
    if current.is_symlink():
        raise ValueError("Bundle roots cannot be symbolic links")
    for part in name.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("Bundle paths cannot contain symbolic links")
    return current


def validate_tuning(path, measurement_policy):
    cache = json.loads(path.read_text())
    entries = {key: value for key, value in cache.items() if not key.startswith("_")}
    records = cache.get("_records", {}).get(POLICY_RECORD, {})
    if not entries or not cache.get("_metadata") or set(entries) != set(records):
        raise ValueError("Every native shape choice needs its original measurement provenance")
    retained = {}
    retained_policy = measurement_policy.get("retained_records")
    for key, value in entries.items():
        record = records[key]
        raw = record.get("policy", [])
        if len(raw) % 2 or len(set(raw[::2])) != len(raw[::2]):
            raise ValueError("Invalid native measurement policy")
        policy = dict(zip(raw[::2], raw[1::2], strict=True))
        if record.get("config") != value:
            raise ValueError("A native choice differs from its measured policy")
        if not REQUIRED_POLICY_FIELDS <= policy.keys():
            if retained_policy is None or policy != retained_policy["policy"]:
                raise ValueError("A native choice has no qualified measurement provenance")
            retained[key] = record
        elif policy not in measurement_policy["profiles"]:
            raise ValueError("A native choice differs from its measured policy")
    if retained_policy is not None:
        canonical = json.dumps({"metadata": cache["_metadata"], "records": retained},
                               sort_keys=True, separators=(",", ":")).encode()
        if (len(retained) != retained_policy["entries"]
                or hashlib.sha256(canonical).hexdigest() != retained_policy["sha256"]):
            raise ValueError("The retained native records differ from the selected source")
    return len(entries)


def verify(bundle, identity):
    manifest = json.loads((bundle / "manifest.json").read_text())
    if (manifest.get("schema") != SCHEMA or manifest.get("identity") != identity
            or manifest.get("identity_key") != identity_key(identity)):
        raise ValueError("The kernel bundle is incompatible with this runtime configuration")
    files = manifest["files"]
    if not files or manifest["tuning_path"] not in files:
        raise ValueError("The native tuning cache is missing from the bundle")
    for name, expected in files.items():
        path = checked_path(bundle / "files", name)
        if (not path.is_file() or path.stat().st_size != expected["bytes"]
                or digest(path) != expected["sha256"]):
            raise ValueError("A kernel bundle artifact failed verification: " + name)
    count = validate_tuning(checked_path(bundle / "files", manifest["tuning_path"]),
                            identity["measurement_policy"])
    if count != manifest["tuning_entries"]:
        raise ValueError("The native shape inventory changed")
    return manifest


def install(bundle, destination, identity):
    manifest = verify(bundle, identity)
    targets = {name: checked_path(destination, name) for name in manifest["files"]}
    for name, target in targets.items():
        expected = manifest["files"][name]
        if target.is_file() and digest(target) == expected["sha256"]:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".bundle-", delete=False) as temporary:
            pending = Path(temporary.name)
            try:
                with checked_path(bundle / "files", name).open("rb") as source:
                    shutil.copyfileobj(source, temporary)
                temporary.flush()
                if digest(pending) != expected["sha256"]:
                    raise ValueError("The bundle source changed while installing")
                os.replace(pending, target)
            finally:
                pending.unlink(missing_ok=True)
    LOGGER.info("Loaded kernel bundle %s: %d shape choices, %d artifacts, %d bytes",
                manifest["identity_key"], manifest["tuning_entries"], len(manifest["files"]),
                sum(row["bytes"] for row in manifest["files"].values()))
    return manifest
