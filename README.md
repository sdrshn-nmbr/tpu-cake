# TPU-cake

[CAKE](https://arxiv.org/pdf/2608.12629) uses AI agents to write, test, and improve GPU kernels. Its implementation is not open source, but since I had experience from recent experiments with analyzing HLO dumps and XProf objects while writing Pallas kernels for TPUs, I thought it would be cool to port the idea to the TPU world. The CAKE paper describes a useful loop that translates well to TPUs:

```text
write a schedule
→ reject invalid work
→ test correctness
→ run it on hardware
→ keep measured improvements
→ turn failures into better checks
```

TPU-cake applies that idea to Pallas and Mosaic. It describes TPU schedules, checks them before execution, and keeps the evidence needed to compare them honestly.

## How MatX inspired

[MatX](https://matx.com/research) follows a similar way of model x software x hardware co-design:

- Estimate compute and data movement before building.
- Make memory, parallelism, and hardware choices explicit.
- Choose useful specialization instead of blindly maximizing fusion.
- Keep the workload and correctness rules outside generated code.
- Use controlled experiments and real hardware measurements.
- Turn repeated failures into checks, tests, or clearer limits.

TPU-cake began by combining CAKE's search loop with this MatX discipline (as seen in repos like https://github.com/MatX-inc/seqax and the public TPU tools already available in JAX, OpenXLA, Tokamax, etc.

## Objects

- A custom xDSL dialect for TPU schedules.
- A high-level distributed-tensor xDSL dialect with explicit pending reductions and collectives.
- Typed shapes, sharding, layout, memory placement, ownership, and lifetime.
- Checks for DMA use, synchronization, operation shapes, and live memory.
- A small Python frontend that emits canonical xDSL.
- Immutable workload, experiment, metric, evidence, and run-receipt contracts.
- Matmul and Thinking Machines Inkling ragged paged attention examples.
- An XProf reader that rejects captures without the intended TPU work.
- Verified distributed matmul lowering through Pallas and Mosaic.
- Replayable full Seqax-forward lowering through multi-device JAX/XLA with exact
  input/output sharding and real collectives.
- Verified full Seqax-forward physical execution with Pallas contractions and
  JAX/XLA vector and collective operations.
- A physical compute, HBM, ICI, and live-memory cost model.
- Separate timing, trace, and hardware-counter runs.
- A resumable, matched-input tile search with alternating run order and bootstrap confidence intervals.
- A durable experiment ledger and artifact-complete receipt validation.

xDSL is the canonical program and schedule format. Pydantic is used only for external contracts, normalized evidence, and receipts. The first complete searched Pallas target is a distributed BF16 matmul followed by reduce-scatter.

## Use

```bash
uv sync
uv run pytest

uv run tpu-cake verify-schedule examples/matmul.mlir
uv run tpu-cake estimate-physical-cost PHYSICAL_SCHEDULE.xdsl \
  --output PHYSICAL_COST_REPORT.json
uv run tpu-cake verify-physical-cost PHYSICAL_COST_REPORT.json \
  --schedule PHYSICAL_SCHEDULE.xdsl
uv run tpu-cake compare-seqax-silu-fusion \
  contracts/seqax-silu-multiply-fusion-v1.json \
  --output SEQAX_SILU_FUSION_REPORT.json
uv run tpu-cake verify-seqax-silu-fusion SEQAX_SILU_FUSION_REPORT.json \
  --contract contracts/seqax-silu-multiply-fusion-v1.json
uv run tpu-cake render-workload inkling-rpa
uv run tpu-cake experiment inkling-rpa
uv run tpu-cake inspect-profile CAPTURE_ROOT \
  --contract contracts/inkling-steady-decode.toml

uv run tpu-cake run-matmul \
  --output-dir RUN_ROOT/timing \
  --mode timing --mesh-size 8 --m 128 --k 1024 --n 1024

uv run tpu-cake finalize-matmul-run RUN_ROOT
uv run tpu-cake search-matmul SEARCH_CONTRACT.json --output-dir SEARCH_ROOT

uv run tpu-cake verify-rpa-search RPA_SEARCH_ROOT \
  --contract contracts/inkling-fused-rpa-block-search.json
uv run tpu-cake finalize-rpa-run RPA_RUN_ROOT \
  --search-root RPA_SEARCH_ROOT \
  --search-contract contracts/inkling-fused-rpa-block-search.json
uv run tpu-cake verify-rpa-bundle RPA_RUN_ROOT \
  --search-contract contracts/inkling-fused-rpa-block-search.json

uv run tpu-cake run-seqax-surface --output-dir SURFACE_RUN_ROOT
uv run tpu-cake verify-seqax-surface SURFACE_RUN_ROOT
uv run tpu-cake run-seqax-surface-profile \
  --output-dir SEQAX_SURFACE_PROFILE_ROOT/SCENARIO/MODE \
  --surface-root SURFACE_RUN_ROOT --scenario SCENARIO --mode MODE
uv run tpu-cake finalize-seqax-surface-profile SEQAX_SURFACE_PROFILE_ROOT \
  --surface-root SURFACE_RUN_ROOT
uv run tpu-cake verify-seqax-surface-profile SEQAX_SURFACE_PROFILE_ROOT

uv run tpu-cake calibrate-seqax-cost SEQAX_SURFACE_PROFILE_ROOT \
  --contract contracts/seqax-cost-calibration-v1.json \
  --output COST_CALIBRATION_REPORT.json
uv run tpu-cake verify-seqax-cost-calibration COST_CALIBRATION_REPORT.json \
  --profile-root SEQAX_SURFACE_PROFILE_ROOT \
  --contract contracts/seqax-cost-calibration-v1.json

uv run tpu-cake run-seqax-physical-pallas \
  --output-dir SEQAX_PALLAS_RUN_ROOT/timing --mode timing
uv run tpu-cake run-seqax-physical-pallas \
  --output-dir SEQAX_PALLAS_RUN_ROOT/trace --mode trace
uv run tpu-cake run-seqax-physical-pallas \
  --output-dir SEQAX_PALLAS_RUN_ROOT/counters --mode counters
uv run tpu-cake finalize-seqax-physical-pallas SEQAX_PALLAS_RUN_ROOT
uv run tpu-cake verify-seqax-physical-pallas SEQAX_PALLAS_RUN_ROOT
```

## Current boundary

The distributed matmul path is complete through verified TPU execution and bounded tile search. The distributed-tensor dialect represents the pinned Seqax forward algebra, has an independent numerical interpreter, and lowers the complete forward program into physical TPU xDSL. That physical program drives the full dataflow on an eight-device `d=2, t=4` mesh. Its contractions execute through 17 Pallas calls; vector operations and 29 all-gathers plus five reduce-scatters execute through JAX/XLA. A three-phase receipt binds exact inputs, outputs, compiler HLO, raw XPlanes, XProf exports, and periodic hardware-counter evidence.

This full-forward path now lowers declared contraction tiles into Pallas grids and block indexing. A bounded TPU7x search over global split-K, split-N, and split-KN policies retained the full-tile incumbent for the fixed model-256, one-layer, one-token surface; it did not find a useful improvement. Vector operations, collectives, DMA annotations, and resource schedules remain implemented or selected by JAX/XLA rather than owned Mosaic kernels. The result is therefore a verified physical-contraction baseline and a narrow negative search result, not a claim that the complete physical schedule is optimal.

The Seqax workload-surface experiment compares the unwrapped `shard_map` control with the canonical whole-program JIT across three forward shapes. Inputs are placed on their declared TPU shards before timing. Each scenario is resampled independently, and promotion requires a confidence interval above the declared practical threshold without a material regression in any scenario. Saved HLO is runner-captured integrity evidence with exact hashes and structural replay checks; it is not independently signed compiler attestation.

The first Seqax cost calibration replays the accepted three-shape trace and counter bundle, then fits a fixed residual-overhead term plus a layer-count-associated residual term on top of the IR-derived idealized resource floor. It describes that exact profiler-instrumented surface within a declared residual bound, but has no held-out predictive validation. It therefore quantifies an overhead-dominated gap in the advertised-rate floor; neither coefficient has causal attribution, and the fit has no sampling-uncertainty or coefficient-confidence-interval estimate. It is not a production latency model, a calibrated model-size or sequence-length scaling law, or a calibration of MXU, HBM, or ICI rates.

The generic physical cost report consumes the verified `tpu_schedule.kernel` directly. It binds the exact physical schedule hash and accounts for declared MXU tiles and grids, per-operation vector work, explicit DMA, inclusive buffer lifetimes, view aliases, pipeline trip counts and rotation storage, collective traffic, remote-DMA routes, topology links, and resource concurrency. It can therefore distinguish legal physical schedules that share one distributed program. BF16 MXU compute and explicit-HBM values are advertised-rate analytical floors; F16, F32, vector, and special-function work remain unpriced. Remote-DMA time includes exact declared per-link bandwidth and per-device injection/ejection floors. Collective time is a separate ring-equivalent injection-rate scenario because collective plans do not define an exact per-link byte schedule. The report does not claim compiler execution, measured latency, or predictive calibration.

The first explicit fusion comparison replaces each legacy unmaterialized `silu` then `multiply` pair with one typed `silu_multiply` operation across the declared tiny, wider, and deeper Seqax surface. Public replay regenerates both distributed and physical schedules from an external contract and verifies exact producer lineage, declared work, traffic, and memory deltas. It removes 64, 512, and 1,024 declared VMEM bytes per device respectively, without changing declared peak VMEM. JAX/XLA still controls realized vector fusion and materialization, so these are schedule-model savings with no measured winner or predictive validation.

Inkling fused RPA has a complete typed adapter, numerical oracle, TPU execution, bounded block-size search, separate timing/trace/counter captures, and a search-bound receipt. It delegates execution to the pinned upstream RPA Pallas wrapper and covers one fixed decode-only local-shard fixture. It does not yet represent the wrapper's internal Pallas schedule, own the outer multi-device `shard_map`, prove full Inkling serving, or establish a global block-size optimum.

## Ideas and source material

- [MatX research](https://matx.com/research)
- [Seqax](https://github.com/MatX-inc/seqax)
- [Future leakage in block-quantized attention](https://matx.com/research/leaky_quantization)
- [SPIRe](https://matx.com/research/sd) and its [paper](https://arxiv.org/pdf/2504.06419)
- [Speculative decoding with blockwise sparse attention](https://matx.com/research/sd_nsa)
- [Prioritize values over keys](https://matx.com/research/smva)
- [Optimize for inference too](https://matx.com/research/lifetime_llm_cost)
- [Simple and fast Rust deriving](https://matx.com/research/rules_derive)
- [Chip design from the bottom up](https://www.youtube.com/watch?v=oIk3R-sMX5o)
- [How GPT, Claude, and Gemini are trained and served](https://www.youtube.com/watch?v=xmkSf5IS-zw)
- [Reiner Pope on transformer-optimized chips](https://www.youtube.com/watch?v=qvrdCpLPbuQ)
- [Where silicon designed for LLMs is headed](https://www.youtube.com/watch?v=gm3parYIMqA)
- [Hardware-software codesign](https://www.youtube.com/watch?v=zrMYIhmuXEo)
- [CAKE](https://arxiv.org/pdf/2608.12629)
- [JAX Pallas and Mosaic](https://docs.jax.dev/en/latest/pallas/design/design.html)
- [Ragged Paged Attention](https://arxiv.org/pdf/2604.15464)
- [Splash Attention](https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py)
- [Tokamax](https://github.com/openxla/tokamax)
- [Accelerator Agents, MaxKernel, and JAXBench](https://github.com/AI-Hypercomputer/accelerator-agents)
- [XProf](https://openxla.org/xprof/kernel-profiling)
- [TPU accelerator microbenchmarks](https://github.com/AI-Hypercomputer/accelerator-microbenchmarks)
