# Layout, batching, and accelerator evaluation

Reviewed against Mojo `1.0.0` on macOS Apple silicon on 2026-08-22.

## Decision

Keep the released numerical core CPU-first and standalone. Do not add MAX,
`LayoutTensor`, or GPU kernels yet. Add a batched public API only after a real
multi-agent consumer and a benchmark justify independent workspace lanes.

| Candidate | Mojo 1.0.0 standalone result | Decision |
|---|---|---|
| `LayoutTensor` | Not shipped in the standalone Mojo package | Defer; current flat contiguous buffers already expose every used axis |
| CPU `parallelize` | Not exported by the standalone package | Keep the deterministic serial kernel |
| GPU `DeviceContext` | Host API moved behind MAX | Defer to an optional MAX backend |
| Sequential batch wrapper | Reuses one scratch arena but exposes no new parallelism | Do not add a misleading batch API |
| Independent batch lanes | Requires one workspace per lane and a new memory/performance contract | Prototype only when a multi-agent workload exists |

## Evidence

The pinned compiler rejects standalone imports for `std.layout.LayoutTensor`,
`std.algorithm.functional.parallelize`, and `std.gpu.host.DeviceContext`. This
matches the [Mojo 1.0.0 release notes](https://mojolang.org/releases/v1.0.0/),
which moved `layout` and accelerator-related APIs to MAX. The currently
published [GPU requirements](https://docs.modular.com/mojo/requirements/) list
Apple silicon as compatible, but that describes the accelerator stack rather
than this repository's standalone Mojo dependency.

This is an availability decision, not a claim that layouts or GPUs cannot help.
The dense contractions have parallel work over states, actions, static states,
and independent episodes. They become plausible accelerator candidates only
when their arithmetic amortizes device setup, transfers, kernel launches, and
the synchronization between message-passing iterations.

## Promotion gate for a future backend

An optional CPU-batch or MAX backend must:

1. retain the existing flat public tensor contracts or provide an explicit
   adapter;
2. match the frozen Float32 parity fixtures and convergence traces;
3. compare batch sizes `1, 4, 16, 64` at small, medium, and paper-sized models;
4. report compilation, warm-up, steady-state latency, episode throughput,
   baseline RSS, incremental host RSS, and device memory;
5. beat the prepared CPU path on at least one declared production workload;
6. remain optional so `pixi run package` keeps producing a standalone
   `aif_mojo.mojoc`.

On this Mac, Apple GPU work should be evaluated first because it avoids a cloud
round trip. NVIDIA validation can use a disposable cloud runner after the MAX
dependency is explicit; the current CPU roadmap does not need Kaggle or another
GPU service.
