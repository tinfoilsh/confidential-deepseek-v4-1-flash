"""Compare CPU-built GPU payloads with the qualified archive, without executing them."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

from elftools.elf.elffile import ELFFile

REFERENCE_SHA256 = "6537581894af3b1423124ef371eab9b16fe370ce03511f7365354255e0c748db"
CUTE_CACHE_DIRECTORY = "/mxfp8_gemm_sm100a_cute_dsl/"
TRITON_CACHE_DIRECTORY = "/.triton/cache/"


def gpu_sections(binary, cute=False):
    elf = ELFFile(io.BytesIO(binary))
    sections = {section.name: section.data() for section in elf.iter_sections()
                if (section.name == ".lrodata" if cute else section.name.startswith(".text."))}
    if not sections:
        raise ValueError("The object has no expected GPU payload sections")
    return sections


def compare(reference, artifacts):
    with reference.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != REFERENCE_SHA256:
            raise ValueError("The reference archive checksum differs")
    with tarfile.open(reference) as archive:
        cute = {Path(member.name).name: archive.extractfile(member).read()
                for member in archive.getmembers()
                if member.isfile() and CUTE_CACHE_DIRECTORY in member.name
                and member.name.endswith(".o")}
        triton = {member.name: archive.extractfile(member).read()
                  for member in archive.getmembers()
                  if member.isfile() and TRITON_CACHE_DIRECTORY in member.name
                  and member.name.endswith(".cubin")}
    actual_cute = {path.name: path for path in (artifacts / "mxfp8").glob("*.o")}
    if actual_cute.keys() != cute.keys():
        raise ValueError("The rebuilt CuTe inventory differs from the reference")
    rows = []
    for name, expected in cute.items():
        actual = actual_cute[name].read_bytes()
        rows.append({"file": "mxfp8/" + name,
                     "gpu_payload_identical": gpu_sections(actual, True) == gpu_sections(expected, True),
                     "whole_object_identical": actual == expected})
    actual_triton = list((artifacts / "triton").rglob("*.cubin"))
    if not actual_triton:
        raise ValueError("No rebuilt Triton kernels were supplied")
    for path in sorted(actual_triton):
        actual = path.read_bytes()
        matches = [name for name, expected in triton.items()
                   if Path(name).name == path.name and gpu_sections(actual) == gpu_sections(expected)]
        rows.append({"file": str(path.relative_to(artifacts)), "matching_references": matches,
                     "gpu_payload_identical": bool(matches),
                     "whole_object_identical": any(actual == triton[name] for name in matches)})
    return {"reference_sha256": REFERENCE_SHA256, "gpu_execution": False,
            "cute_compared": len(cute), "triton_compared": len(actual_triton),
            "triton_reference_inventory": len(triton),
            "matching_gpu_payloads": sum(row["gpu_payload_identical"] for row in rows),
            "matching_whole_objects": sum(row["whole_object_identical"] for row in rows),
            "files": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    arguments = parser.parse_args()
    report = compare(arguments.reference, arguments.artifacts)
    arguments.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not all(row["gpu_payload_identical"] for row in report["files"]):
        raise SystemExit("Some generated GPU payloads differ; see the comparison report")


if __name__ == "__main__":
    main()
