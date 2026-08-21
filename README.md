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
- Verified full Seqax-forward physical execution with Pallas contractions,
  one explicit full-local Pallas fused-vector implementation, and JAX/XLA
  fallback vector and collective operations.
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
uv run tpu-cake estimate-physical-latency PHYSICAL_SCHEDULE.xdsl \
  --calibration contracts/tpu7x-collective-latency-v1.json \
  --output PHYSICAL_LATENCY_REPORT.json
uv run tpu-cake verify-physical-latency PHYSICAL_LATENCY_REPORT.json \
  --schedule PHYSICAL_SCHEDULE.xdsl \
  --calibration contracts/tpu7x-collective-latency-v1.json
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
PYTHONPATH=/home/sudarshan/inkle/engine/sglang-jax/python \
uv run python -m tpu_cake.rpa_device_main SHARDED_RPA_SURFACE_ROOT \
  --surface-contract contracts/inkling-sharded-rpa-surface.json
uv run tpu-cake verify-inkling-sharded-rpa-surface SHARDED_RPA_SURFACE_ROOT \
  --contract contracts/inkling-sharded-rpa-surface.json

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

uv run tpu-cake capture-seqax-residual-profile-hlo \
  --contract contracts/seqax-residual-all-reduce-profile-v1.json
uv run tpu-cake run-seqax-residual-profile \
  --contract contracts/seqax-residual-all-reduce-profile-v1.json \
  --output-dir SEQAX_RESIDUAL_PROFILE_ROOT
uv run tpu-cake verify-seqax-residual-profile SEQAX_RESIDUAL_PROFILE_ROOT \
  --contract contracts/seqax-residual-all-reduce-profile-v1.json

uv run tpu-cake run-seqax-residual-confirmation \
  --contract contracts/seqax-residual-all-reduce-confirmation-v1.json \
  --output-dir SEQAX_RESIDUAL_CONFIRMATION_ROOT
uv run tpu-cake verify-seqax-residual-confirmation SEQAX_RESIDUAL_CONFIRMATION_ROOT \
  --contract contracts/seqax-residual-all-reduce-confirmation-v1.json
```

## Current boundary

The distributed matmul path is complete through verified TPU execution and bounded tile search. The distributed-tensor dialect represents the pinned Seqax forward algebra, has an independent numerical interpreter, and lowers the complete forward program into physical TPU xDSL. That physical program drives the full dataflow on an eight-device `d=2, t=4` mesh. Its contractions execute through 17 Pallas calls; vector operations and 29 all-gathers plus five reduce-scatters execute through JAX/XLA. A three-phase receipt binds exact inputs, outputs, compiler HLO, raw XPlanes, XProf exports, and periodic hardware-counter evidence.

This full-forward path now lowers declared contraction tiles into Pallas grids and block indexing. A bounded TPU7x search over global split-K, split-N, and split-KN policies retained the full-tile incumbent for the fixed model-256, one-layer, one-token surface; it did not find a useful improvement. The fused SiLU-multiply candidate has one owned full-local Pallas vector implementation; other vector operations, collectives, DMA annotations, and resource schedules remain implemented or selected by JAX/XLA rather than owned Mosaic kernels. The result is therefore a verified physical-contraction baseline and a narrow negative search result, not a claim that the complete physical schedule is optimal.

The standalone distributed-matmul path also has an opt-in Pallas-owned bidirectional reduce-scatter, adapted from the [JAX distributed Pallas remote-DMA pattern](https://docs.jax.dev/en/latest/pallas/tpu/distributed.html). Its typed physical implementation and saved plan bind the ring choice, two-buffer HBM scratch, VMEM accumulator, five DMA semaphores, two capacity semaphores, one scoped startup semaphore, two startup-barrier phases, and the exact `2g+1` half-output remote-copy count. Physical verification and the static resource report charge the VMEM scratch and retain the exact native endpoint bytes; HBM and semaphore counts remain plan-bound because the kernel schema has no corresponding capacity fields. The generic collective byte scenario is still not a per-link traffic schedule for this implementation. The legacy `lax.psum_scatter` plan and hashes remain the default. CPU interpret and disposable TPU probes establish compilation and numerical parity for the opt-in path, not a performance winner or a general collective replacement.

The Seqax workload-surface experiment compares the unwrapped `shard_map` control with the canonical whole-program JIT across three forward shapes. Inputs are placed on their declared TPU shards before timing. Each scenario is resampled independently, and promotion requires a confidence interval above the declared practical threshold without a material regression in any scenario. Saved HLO is runner-captured integrity evidence with exact hashes and structural replay checks; it is not independently signed compiler attestation.

The first Seqax cost calibration replays the accepted three-shape trace and counter bundle, then fits a fixed residual-overhead term plus a layer-count-associated residual term on top of the IR-derived idealized resource floor. It describes that exact profiler-instrumented surface within a declared residual bound, but has no held-out predictive validation. It therefore quantifies an overhead-dominated gap in the advertised-rate floor; neither coefficient has causal attribution, and the fit has no sampling-uncertainty or coefficient-confidence-interval estimate. It is not a production latency model, a calibrated model-size or sequence-length scaling law, or a calibration of MXU, HBM, or ICI rates.

The generic physical cost report consumes the verified `tpu_schedule.kernel` directly. It binds the exact physical schedule hash and accounts for declared MXU tiles and grids, per-operation vector work, explicit DMA, inclusive buffer lifetimes, view aliases, pipeline trip counts and rotation storage, collective traffic, remote-DMA routes, topology links, and resource concurrency. It can therefore distinguish legal physical schedules that share one distributed program. BF16 MXU compute and explicit-HBM values are advertised-rate analytical floors; F16, F32, vector, and special-function work remain unpriced. Remote-DMA time includes exact declared per-link bandwidth and per-device injection/ejection floors. Collective time is a separate ring-equivalent injection-rate scenario because collective plans do not define an exact per-link byte schedule. The report does not claim compiler execution, measured latency, or predictive calibration.

The TPU7x collective-latency overlay replaces that byte-only collective scenario with exact-size paired medians from one bound `d=2, t=4` slice. It is deliberately fail-closed outside the ten measured axis, operation, reducer, dtype, and payload combinations in the model-256, one-layer, one-token schedules. Sum reductions require the exact measured dtype and reducer; all-gather uses an explicit payload-only transport assumption across element types. On that surface, sharded RMS normalization reduces the modeled serialized collective term by 5.88 microseconds, but the matched whole-forward diagnostic regresses by 1.92%. Vector and special-function execution, overlap, and compiler scheduling remain unpriced, so this is a descriptive localization result rather than a latency predictor or promotion. This matches the [JAX Scaling Book sharding model](https://jax-ml.github.io/scaling-book/sharding/): small-buffer collectives can be dominated by fixed dispatch and hop latency, so byte counts alone are not enough to select the boundary.

The optional residual-all-reduce schedule replaces two reduce-scatter, residual, and later all-gather boundaries with two full-width sum all-reduces while retaining a local shard for the next contraction. On the fixed model-256, one-layer, one-token surface it has 15 all-gathers, two all-reduces, and one reduce-scatter, compared with 17, zero, and three for the standard schedule. It moves more ring-equivalent bytes, so its only plausible win is fewer latency-bearing collective boundaries. The profile contract pins exact HLO identities from two matching clean TPU compilation captures; this enables the runner but is not profile or correctness evidence. Its eventual trace/counter receipt is diagnostic only: each schedule must pass the frozen numerical policy independently, and observed XProf rows remain separate from static physical and compiler inventories.

The residual-all-reduce confirmation is a separate unprofiled decision experiment bound to that diagnostic receipt and its exact plan and HLO identities. It keeps both candidates resident, requires both to pass the frozen five-seed numerical contract, and records 32 balanced AB/BA rounds with five synchronized samples per candidate. The candidate wins only when the deterministic 100,000-sample paired-median bootstrap has a 99% lower confidence bound strictly above a 3% practical improvement. The protocol predeclares no early stop or further retry; enforcing that rule across independently chosen output roots is an operational boundary rather than a cryptographically attested one. A passing receipt promotes only the fixed model-256, one-layer, one-token TPU7x workload; it does not establish independent whole-model correctness, larger-shape benefit, memory feasibility, or a general collective rule.

The first explicit fusion comparison replaces each legacy unmaterialized `silu` then `multiply` pair with one typed `silu_multiply` operation across the declared tiny, wider, and deeper Seqax surface. Public replay regenerates both distributed and physical schedules from an external contract and verifies exact producer lineage, declared work, traffic, memory deltas, and the full-local Pallas implementation choice. The fused operation executes through an owned Pallas region in CPU interpret tests, but has not yet passed TPU Mosaic compilation. It removes 64, 512, and 1,024 declared VMEM bytes per device respectively, without changing declared peak VMEM. These are schedule-model savings with no measured winner, physical-memory proof, or predictive validation.

The Seqax BF16 numerical-v5 run is frozen negative portability evidence: its TPU Pallas and control outputs agreed exactly and passed against both CPU references, but one producer-to-ARM CPU comparison missed the v5 replay bound. Numerical-v6 treats all 41 inspected v5 observations as calibration and reserves three new width, batch/sequence, and depth/sequence surfaces with v6-derived seeds. V6 explicitly uses the already-frozen CPU-facing 3u relative-L2 and 8u row-scaled bounds symmetrically for producer and verifier JAX CPU references, while requiring normal and instrumented TPU outputs to pass independently against both references. Each path is still assessed against the declared RMSNorm scale, fixed-order projection oracles, BF16 conversions, and final-output bounds; cross-path checkpoint distances and top-1 agreement remain reporting only. The producer receipt proves only producer-host validation. Portable acceptance requires a separate relocation attestation that binds the archive and receipt hashes, verifier runtime and architecture, regenerated CPU hashes, metrics, and verdict. Its public archive path enforces contract-bound limits of 1 GiB compressed, 10,000 members, 1 GiB per member, and 4 GiB total uncompressed before extraction. This is bounded agreement with two JAX CPU realizations backed by fixed-order checkpoint oracles, not architecture-independent proof of the whole forward pass. The v6 runner is enabled by two matching TPU compile captures; this pinning is compiler-identity evidence only, not numerical acceptance.

Inkling fused RPA has a complete typed adapter, numerical oracle, TPU execution, bounded block-size search, separate timing/trace/counter captures, and a search-bound receipt. The production-shaped surface runner owns the explicit `2x4` outer `shard_map`, global and local ABIs, deterministic global inputs, reconstruction oracle, exact StableHLO identity, five-seed numerical check, repeated cache equality, and synchronized resident-input wall timing. Raw compiler HLO remains receipt-bound because identical TPU compiles did not produce byte-stable text. The local attention call still delegates to the pinned upstream RPA Pallas wrapper, so its internal Pallas schedule is not yet represented by TPU-cake. The pinned surface is limited to Hq32, Hkv16, D128, contexts 128/512/1024/2048, and decode blocks 8/128/8/128; it is not accepted until a fresh bundle passes public relocated replay, and it is not full Inkling serving, a throughput promotion, or a global block-size optimum. The contract also pins the backend Python path and clean Inkling repository revision that provide `sgl_jax`; compiling rejects a missing, dirty, or different backend checkout before creating an output root, while public receipt replay remains path-relocatable.

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
