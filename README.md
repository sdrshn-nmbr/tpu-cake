# TPU-native verified kernel search

This repository is the home of a TPU-native, CAKE-style system for generating,
checking, measuring, and improving Pallas kernel schedules. Inkling is an early
workload and source of real serving shapes, but the system is not part of the
Inkling serving engine.

The first checked-in component is the source-backed research corpus below. The
schedule representation, verifier, cost model, and search loop will be added as
independent packages rather than embedded in a model-specific runtime.

This corpus supports the design of an agent-facing TPU kernel system inspired by CAKE. It separates retrieved source material from conclusions drawn from it.

## Corpus layout

- `matx/manifest.json`: hashed inventory of MatX articles, talks, transcripts, papers, and pinned code snapshots.
- `matx/raw/`: original HTML, PDF, video captions, metadata, and code archives.
- `matx/text/`: extracted article text, official transcripts, clean YouTube transcripts, and paper text.
- `ecosystem/manifest.json`: hashed inventory of CAKE, RPA, Pallas, Tokamax, and accelerator-agent material.
- `ecosystem/source/`: pinned files from JAX and Tokamax.
- `ecosystem/code/`: pinned archives of Tokamax and Google Cloud's Accelerator Agents.
- `collect_matx.py` and `collect_ecosystem.py`: repeatable collectors.

The manifests include source URLs, byte counts, and SHA-256 hashes. YouTube video files are not copied; captions, chapters, descriptions, and source URLs are retained.

## MatX primary source set

### Research articles

| Source | Local text | Main relevance |
|---|---|---|
| MatX research index | `matx/text/pages/research-index.md` | Complete public research inventory. |
| MatX One | `matx/text/pages/matx-one.md` | Splittable systolic arrays, SRAM and HBM, flexible shapes, co-located ML and hardware work. |
| Future leakage in block-quantized attention | `matx/text/pages/future-leakage.md` | Numerical optimization must preserve causal behavior; parallel and autoregressive checks can disagree. |
| Speculative decoding with blockwise sparse attention | `matx/text/pages/speculative-decoding-nsa.md` | Change the algorithm so the hardware can reuse data; validate quality with controlled ablations. |
| SPIRe | `matx/text/pages/spire.md` | Predict throughput from compute and memory costs before implementing a full serving system. |
| Prioritize values over keys | `matx/text/pages/prioritize-values.md` | Model design should reflect the asymmetric cost of moving key and value data. |
| Optimize for inference too | `matx/text/pages/lifetime-llm-cost.md` | Training FLOPs alone are the wrong objective when inference dominates lifetime work. |
| seqax | `matx/text/pages/seqax.md` | Explicit math, memory, and parallelism; deterministic data loading; profiling enabled by default. |
| Rust deriving with `macro_rules` | `matx/text/pages/rules-derive.md` | Small, constrained abstractions can improve feedback and avoid large compiler machinery. |

The SPIRe paper is retained as PDF and extracted text. Four pinned seqax snapshots preserve the main, SPIRe, SMVA, and NSA implementations. The GitHub organization inventory captures MatX's public compiler, simulator, model-checker, waveform, and build-system work.

### Supplied talks

| Talk | Primary transcript | What it contributes |
|---|---|---|
| Chip design from the bottom up | `matx/text/transcripts/chip-design-bottom-up.md` | Data movement, scratchpads, systolic arrays, local grain size, and deterministic execution. |
| How GPT, Claude, and Gemini are trained and served | `matx/text/transcripts/llm-training-and-serving.md` | Roofline-style reasoning for batching, model weights, KV caches, network topology, and serving latency. |
| Reiner Pope on transformer-optimized chips | `matx/text/transcripts/stripe-transformer-chips.md` | MatX's iteration loop: mental bounds, resource balances, custom simulators, model experiments, then implementation verification. |
| Where silicon designed for LLMs is headed | `matx/text/transcripts/silicon-for-llms-youtube.md` | SRAM/HBM balance, specialization, interconnect, performance modeling, and prefill/decode differences. |
| Hardware-software codesign | `matx/text/transcripts/hardware-software-codesign-youtube.md` | Choose the right specialization grain, retain flexible compute, and use metrics below the current hardware abstraction. |

Raw WebVTT captions and source metadata are retained for all five talks. The first three also have publisher-provided transcripts.

## Closest public TPU systems

| System | What it already provides | What it does not provide |
|---|---|---|
| Accelerator Agents / MaxKernel | Agent orchestration, Pallas generation, compilation repair, numerical tests, XProf-based profiling, grid-search autotuning, and JAXBench. | A typed TPU schedule representation, pre-compilation schedule verification, a calibrated schedule cost model, and evidence-driven evolution of the language and verifier. |
| JAXBench | Fifty fixed TPU workloads, reference implementations, candidate evaluation, correctness checks, and device-side timing. | Serving-context contracts, exact-model state checks, broad numerical properties, and schedule-level diagnostics. |
| Pallas/Mosaic | Explicit refs, tiles, memory spaces, DMA, semaphores, pipelines, `core_map`, and TPU lowering. | A stable agent-facing schedule contract and localized high-level explanations for many compiler failures. |
| Pallas interpret mode | Bounds checks, race detection, simulated TPU memory spaces, communication, and semaphore execution. | Proof that a candidate compiles, maps efficiently, or behaves identically on a physical TPU. |
| Tokamax | Production kernel patterns, serializable autotuning, argument specifications, cached choices, and TPU7x tuning data. | A general verified schedule language or workload-level correctness authority. |
| RPA | A mature example of ragged tiling, fused cache updates, asynchronous pipelines, and separate decode, prefill, and mixed schedule families. | A general search system; it is one highly developed kernel family. |
| Splash Attention | Reusable TPU attention tiling, online softmax, masking, and pipeline patterns. | Paged-cache serving semantics and Inkling-specific positional behavior. |
| XProf/XPlane | Physical device timing, HLO/LLO attribution, custom-call visibility, and hardware counters. | A guarantee that the intended program was captured or a direct mapping from a finding to a schedule decision. |

The pinned Accelerator Agents archive is the nearest implementation baseline. The local checkout at `/Users/sudarshan/Code/opjax/references/accelerator-agents` was also inspected because it contains MaxKernel and JAXBench in immediately runnable form.

## What MatX contributes to the CAKE design

### 1. Estimate before compiling

MatX starts with lower bounds and resource balances: compute, memory traffic, capacity, and data movement. Reiner Pope describes estimating performance within roughly 30 to 40 percent before implementation, then using a custom performance simulator for the next level of detail. SPIRe applies the same method to speculative decoding by expressing compute and memory in a common unit before measuring a serving stack.

For the TPU system, every candidate should first receive a cheap estimate of:

- HBM bytes;
- VMEM and SMEM footprint;
- MXU and vector work;
- padding waste;
- DMA and collective volume;
- dependency-bound idle time;
- per-core or per-device imbalance.

The estimate ranks candidates. It does not replace physical TPU timing.

### 2. Make the important decisions explicit

Seqax makes math, memory, and parallelism visible. CAKE makes buffers, roles, stages, barriers, and lifetimes visible. Pallas already exposes most TPU mechanisms, but raw Pallas code does not present them as one stable object that a verifier can inspect.

The agent-facing representation should declare:

- logical tensor regions and physical tiles;
- HBM, VMEM, SMEM, and semaphore resources;
- ownership and lifetime;
- DMA producers and consumers;
- MXU and vector operations;
- pipeline stages and waits;
- TensorCore and device assignment;
- remote transfers and collectives;
- accumulation precision, padding, and tail rules.

Mechanical details should be derived during lowering.

### 3. Specialize at the right grain

MatX repeatedly argues for choosing the largest useful recurring grain without eliminating necessary flexibility. This is the kernel-search equivalent of the earlier MoE lesson: a larger fused region is not automatically better.

The system should search distinct schedule families, not reward maximum fusion. A candidate must justify a wider boundary through measured reduction in traffic, launch cost, synchronization, or intermediate storage.

### 4. Use an external workload contract

The mathematical and serving contract must remain outside the generated kernel. It fixes:

- shapes and shape families;
- dtypes and accumulation rules;
- allowed numerical error;
- determinism or invariance properties;
- target hardware and software versions;
- authoritative reference implementation;
- benchmark procedure;
- profile acceptance rules.

MatX's quantized-attention result is a warning: parallel-mode agreement alone can hide an autoregressive failure. Kernel tests need properties that reflect how the operation is used, not only one tensor comparison.

### 5. Treat failures as system input

CAKE's central mechanism is also consistent with MatX's simulator-first workflow. A repeated failure should not remain another prompt example. It should become one of:

- a verifier rule;
- a typed resource or operation;
- a lowering rule;
- a cost-model calibration point;
- a corpus regression test;
- an explicit unsupported capability.

Human review remains the merge gate for changes to the representation, verifier, and lowering.

## Initial architecture implied by the corpus

```text
Workload contract
      |
Typed TPU schedule candidates
      |
Static verifier --------> localized findings
      |
Analytical cost model ---> ranking and bottleneck estimate
      |
Pallas interpret mode ---> bounds, races, DMA and semaphore behavior
      |
Pallas/core_map lowering
      |
Compilation + numerical oracle
      |
Warm TPU benchmark
      |
XProf timing trace + separate hardware-counter trace
      |
Retained candidate, rejected candidate, or harness improvement
```

The first implementation should sit above Pallas. It should descend into Mosaic or XLA only when a required schedule cannot be expressed or a compiler finding cannot be localized.

## Corpus-driven starting workloads

The corpus points to a progression based on verification difficulty and relevance:

1. Dense tiled matrix multiplication with explicit HBM-to-VMEM pipelining.
2. Fused gate, up, and activation work around an existing grouped multiplication.
3. Ragged gather, weighting, and reduction.
4. Ragged grouped multiplication using real routed-token distributions.
5. RPA block-size and pipeline variants for fixed decode shapes.
6. Multi-device token transfer with explicit remote DMA and synchronization.

Each new workload should either fit the existing representation or expose a specific missing capability. It should not trigger a broad redesign without a corpus-backed reason.

## Known gaps in this collection

- MatX does not publish its chip compiler, performance simulator, internal kernel library, or verification infrastructure. Public statements show principles, not implementation details.
- YouTube captions can contain transcription errors. Publisher transcripts take precedence where available.
- Accelerator Agents and Tokamax are moving projects. Their archives are pinned; web links may describe newer states.
- Pallas and Mosaic are experimental and their public APIs change.
- The collection does not yet contain a normalized corpus of production Pallas schedules. RPA, Splash, Tokamax, and JAXBench are the first seeds.

## Refresh

```bash
uv run research/tpu-cake/collect_matx.py
uv run research/tpu-cake/collect_ecosystem.py
```

After refreshing, verify every file against its manifest before using it as evidence.
