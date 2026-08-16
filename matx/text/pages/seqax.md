# Introducing seqax: A Simple and Efficient LLM Research Codebase

May 6, 2024

MatX ML team

We’re excited to announce [seqax](https://github.com/MatX-inc/seqax), a research-focused
LLM codebase that is simple, efficient, and performs well on up to 100
GPUs or TPUs. Everything you need to edit, from the math, to
parallelism, to memory footprint, is all there in 500 lines of JAX
code.

## Introduction

Mature LLM codebases are often weighed down by excessive
configurability, making it hard to understand which code path is taken.
At MatX, we’ve developed **seqax**—a minimalist codebase
designed for small-to-medium scale LLM pretraining research. In seqax,
we avoid most configuration: no if statements, no inheritance. We run
experiments by directly editing the code.

## Key Features

- **Simplicity**: The entire training loop and model are
  500 lines of code in one file.
- **Explicit Math**: All operations for neural network
  layers are explicit, rather than calling external libraries.
- **Explicit Memory Usage**: All tensors that go into a
  model checkpoint on disk are explicit. All tensors that occupy a lot of
  memory, including activations saved for the backward pass, are
  explicit.
- **Explicit Parallelism**: All parallel communication is
  explicit, with sharding expressed using Python type annotations.
- **Scalability**: Performs well on up to 100 GPUs or
  TPUs, making it suitable for small-to-medium scale research. Includes
  robust implementations for multihost partitioning, checkpointing,
  deterministic data loading, and profiling.

## Explicit Parallelism

In seqax, parallelism is made explicit, allowing you to fully
understand and control how computations are distributed across devices.
Sharding notation is expressed using Python type annotations, making the
code both readable and concise.

For example, here’s how multihost Fully Sharded Data Parallel (FSDP)
and Tensor Parallelism (TP) are implemented for a feedforward network.
There are two sharding axes: `d` for FSDP and `t`
for TP. In our annotation style, `F/t` indicates that
dimension `F` is sharded over `t` chips, and
`F/t` is the size per chip.

```
# Pre-FFN RMSNorm
ln2 = shardops.all_gather('M/t/d -> M', layer_weights.ln2)
gx = shardops.all_gather('B/d L M/t -> B/d L M', x)
nx = rms_norm(gx) * ln2

# FFN, using SwiGLU
w_gate = shardops.all_gather('M/d F/t -> M F/t', layer_weights.w_gate)
gate_proj = shardops.einsum_unreduced('B/d L M, M F/t -> B/d L F/t', nx, w_gate)
w_up = shardops.all_gather('M/d F/t -> M F/t', layer_weights.w_up)
up_proj = shardops.einsum_unreduced('B/d L M, M F/t -> B/d L F/t', nx, w_up)
y = jax.nn.swish(gate_proj) * up_proj
w_down = shardops.all_gather('M/d F/t -> M F/t', layer_weights.w_down)
ffn_out = shardops.einsum_unreduced('B/d L F/t, M F/t -> B/d L M', y, w_down)
ffn_out = shardops.psum_scatter('B/d L M -> B/d L M/t', ffn_out)
```

## Scalability Features

Seqax isn’t just a prototyping tool; it includes several features
essential for experiments of any size:

- **Multihost Partitioning**: Efficiently run experiments
  across multiple hosts.
- **Checkpointing**: Save and restore model checkpoints
  using the [Zarr](https://zarr.readthedocs.io/en/stable/)
  format, which supports various compression and chunk size settings.
- **Deterministic Data Loading**: A data loader that
  ensures deterministic behavior, including resuming training at an
  arbitrary step number.
- **Profiling**: Profiling is always enabled during the
  first few training steps to help identify bottlenecks early.

## Get Started

Seqax is open source and available on GitHub.

- **Repository**: <https://github.com/MatX-inc/seqax>
- **Download**: [Zip
  Archive](https://github.com/MatX-inc/seqax/archive/refs/heads/main.zip)

## Conclusion

We’ve been using seqax internally at MatX for several months and are
excited to share it with the community. Stay tuned for our upcoming [research](/research) publications.

If you’re passionate about working on one of the cleanest LLM
codebases and want to be part of a team that values open research,
consider joining us at MatX: <https://matx.com/jobs>.
