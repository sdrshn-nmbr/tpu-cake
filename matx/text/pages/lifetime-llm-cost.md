# Optimize for inference too, not just training FLOPs

January 8, 2025

MatX ML team

Large Language Models (LLMs) have been shaped by the scaling laws, as
established by [Kaplan et al.,
2020](https://arxiv.org/abs/2001.08361) and [Hoffmann et al.,
2022](https://arxiv.org/abs/2203.15556). They guide us to design models that optimize for training cost
but often overlook inference costs. Although training and inference
costs are highly related, they diverge sharply in one critical area:
attention key-value (KV) computation. During training, KV computation is
usually cheaper than the rest of the model, but during inference,
loading the KV cache becomes the dominant expense. This disconnect has
led to models with ~70% efficiency during training but only ~10%
efficiency during inference.

Perhaps a methodology that considers *both* inference and
training cost would lead to more balanced models? This post sketches out
some options in this direction.

## Scaling Laws and Model Size Selection

[Kaplan et al., 2020](https://arxiv.org/abs/2001.08361)
demonstrated that transformer model quality improves predictably with
increasing model parameters NNN and
training tokens DDD. They estimated the
compute cost of training a transformer model at 6ND6ND6ND floating-point operations (FLOPs). Other
resource requirements, such as loading weight matrices from memory, are
amortized by using large training batches. Since training is
compute-bound, the training FLOPs budget effectively determines the
optimal model size and the number of training tokens needed to maximize
model performance.

## Inference and the Memory Bottleneck

During inference, we distinguish between two phases: prefill and
decode.

**Prefill** initializes the KV cache by processing
continuous input text, such as a question or background information. In
this phase, the model processes all input tokens in parallel, similar to
training. As a result, the cost of loading weight matrices is amortized
over the sequence, and computation remains the dominant cost. The
compute cost per token is approximately 2N2N2N FLOPs, since only the forward pass is
performed.

In contrast, the **decode** phase generates text one
token at a time. For each generated token, the model still requires
2N2N2N FLOPs for the forward pass.
Furthermore, for each generated token, the model needs to load all
parameters and the KV cache associated with all previous tokens. While
the loading of model parameters can be amortized by using larger batch
sizes, the KV cache grows both with the **batch size** and
the **sequence length**, making memory bandwidth a
potential bottleneck ([Pope
et al., 2022](https://proceedings.mlsys.org/paper_files/paper/2023/hash/c4be71ab8d24cdfb45e3d06dbfca2780-Abstract-mlsys2023.html)).

Consider decoding using a transformer model with multi-head attention
(MHA) ([Vaswani
et al., 2017](https://papers.nips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)), a batch size of 128, a context length of 8,192, and
using a 1-byte numeric type. A typical model with 8 billion parameters
will require loading 280 GB of KV cache, i.e., 30 times larger than the
model parameters themselves! When running such a model on typical
hardware, the arithmetic logic will spend much of its time waiting for
the KV cache to load.

## Llama 3 Models and Grouped Query Attention

The Llama 3 models ([Llama
Team, 2024](https://arxiv.org/abs/2407.21783)) address the inference memory bottleneck by incorporating
Grouped Query Attention (GQA; [Ainslie et al.,
2023](https://aclanthology.org/2023.emnlp-main.298.pdf)), which significantly reduces the KV cache size by sharing key
and value projections across multiple attention heads. For instance, in
the 8B model, the KV cache size drops from 280 GB with MHA to 69 GB with
GQA. While GQA may cause a slight decrease in model quality compared to
MHA, this trade-off is acceptable given the substantial gain in
inference speed. The use of GQA in Llama 3 models illustrates how
architectural choices can balance compute and memory demands.

The size of the KV cache for different model sizes is shown in the
following table:

|  | KV Cache Size | | |
| --- | --- | --- | --- |
| Model Configuration | 8B Model | 70B Model | 405B Model |
| Transformer with MHA | 280 GB | 1,400 GB | 4,300 GB |
| Llama 3 (with GQA) | 69 GB | 170 GB | 270 GB |

## FLOP-Equivalent

To compare computational FLOPs and memory accesses on equal footing,
we use the **Hardware Operational Intensity (HOI)**:

HOI=Peak Compute Performance (FLOPs/s)Peak Memory Bandwidth (Bytes/s)\text{HOI} = \frac{\text{Peak Compute Performance (FLOPs/s)}}{\text{Peak Memory Bandwidth (Bytes/s)}}HOI=Peak Memory Bandwidth (Bytes/s)Peak Compute Performance (FLOPs/s)​

HOI indicates the hardware’s computational capacity relative to its
memory bandwidth—it tells us how many floating-point operations can be
performed per byte of memory access. By multiplying the KV cache size
(in bytes) by HOI, we obtain the **FLOP-equivalent** cost
required to load the KV cache, allowing us to directly compare memory
operations with compute operations. An important aspect of HOI is that
it tends to remain relatively stable across hardware generations.

For an Nvidia H100 GPU using FP8 arithmetic, HOI is approximately 600
FLOPs/byte. Let’s revisit the Llama 3 8B model, and consider the cost of
decoding the next token in each of 128 sequences with a context length
of 8192:

- Forward pass computation cost: 128×2N≈2 TFLOPs128 \times 2N \approx 2\ \text{TFLOPs}128×2N≈2 TFLOPs
- Cost to load the full KV cache: 69 GB×HOI≈41 TFLOPs69\, \text{GB} \times \text{HOI} \approx 41\ \text{TFLOPs}69GB×HOI≈41 TFLOPs

In this case, the cost of loading the KV cache in FLOP-equivalents is
20 times greater than the compute cost of performing the forward pass.
This indicates that the 8B model is memory-bound during decoding in this
setting.

For the Llama 3 405B model, the ratio of KV cache load cost to
compute cost is approximately 1.5, indicating a closer balance between
compute and memory demands. However, if we increase the context length,
the KV cache size grows proportionally, making decode memory-bound
again.

## Rethinking Model Selection: Total Cost Optimization

Traditional model selection focuses on training FLOPs, defined by the
cost Ctrain=6NDtrainC\_{\text{train}} = 6N D\_{\text{train}}Ctrain​=6NDtrain​, but this approach overlooks the significant
costs associated with inference. To achieve a model of the same quality
that is cheaper over its lifetime, we consider the **total
estimated lifetime cost**, which includes the training, prefill,
and decode phases:

Ctotal=6NDtrain⏟Training cost + 2NDprefill⏟Prefill cost + max⁡(2N,⏟Forward passKV cache size×HOI⏟KV cache loading)×Ddecode⏟Decode costC\_{\text{total}} = \underbrace{6N D\_{\text{train}}}\_{\text{Training cost}} \ +\ \underbrace{2N D\_{\text{prefill}}}\_{\text{Prefill cost}} \ +\ \underbrace{\max( \underbrace{2N,}\_{\text{Forward pass}} \underbrace{\text{KV cache size} \times \text{HOI}}\_{\text{KV cache loading}} ) \times D\_{\text{decode}}}\_{\text{Decode cost}}Ctotal​=Training cost6NDtrain​​​ + Prefill cost2NDprefill​​​ + Decode costmax(Forward pass2N,​​KV cache loadingKV cache size×HOI​​)×Ddecode​​​

- Training cost: The number of FLOPs performed during training,
  including forward and backward passes.
- Prefill cost: The number of FLOPs performed during the prefill
  phase of inference, forward pass only.
- Decode cost: The dominant cost between computation and memory
  accesses during the decode phase, multiplied by the number of decode
  tokens.

Including inference costs in the total model cost alters the optimal
balance between model size and training data. [Sardana et
al., 2023](https://proceedings.mlr.press/v235/sardana24a.html) showed that when inference costs are considered, it
results in recommending smaller models trained on more data compared to
the Chinchilla scaling laws ([Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556)),
which suggest Dtrain≈20ND\_{\text{train}} \approx 20NDtrain​≈20N. Besides training on more tokens, our approximation of the
total computational cost can be used to explore architectural changes to
the transformer model that spend more FLOPs to decrease memory bandwidth
cost during inference.

## Strategies to Spend FLOPs to Save Memory Bandwidth

Our proposed cost metric allows for a comparison of transformer
variants by considering both computational and memory-bandwidth demands.
While memory-efficient attention mechanisms have been extensively
researched, these techniques are designed to save both on FLOPs and on
memory. Such free lunches are hard to find; we propose approaches which
increase FLOPs cost in order to save memory, so long as the total
lifetime cost of the model is improved. Our research agenda includes the
following:

1. **KV Cache Compression Techniques**

Extreme compression techniques for the KV cache, such as Multi-Query
Attention ([Shazeer,
2019](https://arxiv.org/abs/1911.02150)), cross-layer KV cache sharing ([Brandon et al., 2024](https://arxiv.org/abs/2405.12981); [Character
AI](https://research.character.ai/optimizing-inference)), and cross-token KV cache sharing ([Mu et al., 2023](https://arxiv.org/abs/2304.08467)) are often
rejected due to quality degradations when compared on a FLOPs-neutral
basis. However, on a total-cost-neutral basis, these are very likely
massively profitable. Further techniques in this direction are worth
exploring.

2. **Changing Model Shape**

The typical practice for model shape (depth vs width; dmodeld\_{\text{model}}dmodel​ vs dffd\_{\text{ff}}dff​ vs nheadsn\_{\text{heads}}nheads​) is optimized for quality
given a training budget. When optimizing for quality given a
*lifetime* budget, the optimal shape is likely different:

- It likely grows the feedforward network and shrinks the attention
  network, reducing the KV-cache memory bandwidth.
- It likely grows the width (especially dffd\_{\text{ff}}dff​) and decreases the depth. This
  saves memory capacity: while keeping high efficiency, you can run a
  smaller batch size on the same number of chips—or equivalently, run the
  same batch size on a larger number of chips.

3. **Speculative Decoding for Large Batches and
   Contexts**

While Speculative Decoding ([Chen et al., 2023](https://arxiv.org/abs/2302.01318); [Leviathan et al.,
2023](https://icml.cc/virtual/2023/oral/25546)) is traditionally considered a small-batch-size optimization,
it becomes beneficial for large batch sizes when considering KV cache
memory bandwidth ([Chen
et al., 2024](https://www.together.ai/blog/speculative-decoding-for-high-throughput-long-context-inference)). Techniques that incorporate Speculative Decoding
during pretraining, rather than as a post-training optimization, seem
promising.

4. **Enhanced Query-Key Interactions**

Dot-Product Attention is designed to make each query-key interaction
highly computationally efficient, thus keeping training costs low.
However, for inference, where memory fetches are more expensive than
computation in query-key interactions, more computationally complex
query-key interactions may be profitable. This could include
algebraically different interactions ([Kobayashi et
al., 2020](https://aclanthology.org/2020.emnlp-main.574.pdf)), or even deeper neural networks for these
interactions.

By optimizing for a cost metric that aligns more closely with what we
truly care about, we can pursue model designs that pay sufficient
respect to the significant cost of the KV cache. Models which are 1.5x
more expensive to train and 10x cheaper to inference might easily be
possible.
