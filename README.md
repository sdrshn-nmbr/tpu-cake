# TPU-cake

TPU-cake describes TPU kernel schedules, checks them before execution, and keeps the evidence needed to compare them honestly.

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

## Project rules

- Describe important hardware choices explicitly.
- Reject invalid schedules before using a TPU.
- Estimate cost before compiling.
- Compare candidates with the same inputs and conditions.
- Keep timing traces separate from hardware-counter captures.
- Treat real TPU measurements as the final result.
- Turn repeated failures into checks or regression tests.

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

Refresh the research corpus with:

```bash
uv run collect_matx.py
uv run collect_ecosystem.py
```
