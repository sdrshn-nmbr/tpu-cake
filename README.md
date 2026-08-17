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

## What exists

- A custom xDSL dialect for TPU schedules.
- A high-level distributed-tensor xDSL dialect with explicit pending reductions and collectives.
- Typed shapes, sharding, layout, memory placement, ownership, and lifetime.
- Checks for DMA use, synchronization, operation shapes, and live memory.
- A small Python frontend that emits canonical xDSL.
- Immutable workload, experiment, metric, evidence, and run-receipt contracts.
- Matmul and Inkling ragged paged attention examples.
- An XProf reader that rejects captures without the intended TPU work.
- Verified distributed matmul lowering through Pallas and Mosaic.
- A physical compute, HBM, ICI, and live-memory cost model.
- Separate timing, trace, and hardware-counter runs.
- A resumable, matched-input tile search with alternating run order and bootstrap confidence intervals.
- A durable experiment ledger and artifact-complete receipt validation.

xDSL is the canonical program and schedule format. Pydantic is used only for external contracts, normalized evidence, and receipts. The first complete lowering target is a distributed BF16 matmul followed by reduce-scatter.

## Try it

```bash
uv sync
uv run pytest

uv run tpu-cake verify-schedule examples/matmul.mlir
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
```

## Layout

- `src/tpu_cake/dialects/`: canonical TPU schedule language and checks.
- `src/tpu_cake/frontend.py`: Python schedule builder.
- `src/tpu_cake/workloads/`: reference workloads and schedules.
- `contracts/`: experiment and profile contracts.
- `examples/`: generated xDSL schedules.
- `evidence/`: normalized profile results.
- `matx/` and `ecosystem/`: pinned research material.

## Current boundary

The distributed matmul path is complete through verified TPU execution and bounded tile search. The distributed-tensor dialect represents the pinned Seqax forward algebra and has an independent numerical interpreter, but physical Pallas lowering remains deliberately narrow and currently accepts the distributed matmul slice.

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

Pinned copies, transcripts, hashes, and exact source URLs are stored under `matx/` and `ecosystem/`.

Refresh them with:

```bash
uv run collect_matx.py
uv run collect_ecosystem.py
```
