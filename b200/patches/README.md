# B200 inference patch series

`Dockerfile.b200` applies this entire series in filename order, after the shared
`patches/` series. The patches are review boundaries; deploy the complete series.

| Patch | Review focus |
| --- | --- |
| `0001-cc-kernel-timing.patch` | GPU timestamps, timing provenance, and the precompiled timer library. |
| `0002-mxfp8-compiled-cache.patch` | Persistent MXFP8 kernel caching with source-aware cache keys. |
| `0003-pinned-kernel-replay.patch` | Exact recorded-choice verification and disabled startup profiling. |
| `0004-sampling-metadata-mirrors.patch` | Explicit CPU/GPU metadata copies for sampling processors. |
| `0005-native-prefill-graphs.patch` | Compact native prefill capture selection and capture logging. |
| `0006-target-preparation-graphs.patch` | Attention preparation, buffer ownership, and replay eligibility. |
| `0007-target-verification-graphs.patch` | Output projection, sampling, speculative verification, and result ownership. |
| `0008-target-graph-integration.patch` | Capture, selection, replay, and result consumption in the runner. |

The graph definitions in patches 6–7 are connected to the runner by patch 8.
Patches 1 and 3 both modify the autotuner; patches 5 and 8 both modify graph
capture. Their hunks are based on the preceding patches in the series.
