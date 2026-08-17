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
- Typed shapes, sharding, layout, memory placement, ownership, and lifetime.
- Checks for DMA use, synchronization, operation shapes, and live memory.
- A small Python frontend that emits canonical xDSL.
- Immutable workload, experiment, metric, evidence, and run-receipt contracts.
- Matmul and Inkling ragged paged attention examples.
- An XProf reader that rejects captures without the intended TPU work.

xDSL is the schedule format. Pydantic is used only for inputs, evidence, and receipts. Pallas and Mosaic are future lowering targets.

## Try it

```bash
uv sync
uv run pytest

uv run tpu-cake verify-schedule examples/matmul.mlir
uv run tpu-cake render-workload inkling-rpa
uv run tpu-cake experiment inkling-rpa
uv run tpu-cake inspect-profile CAPTURE_ROOT \
  --contract contracts/inkling-steady-decode.toml
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

The schedule language, contracts, examples, and profile checks work. Pallas lowering, cost prediction, and automated schedule search are not implemented yet.

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
