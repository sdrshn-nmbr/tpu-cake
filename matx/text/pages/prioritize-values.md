# Prioritize values over keys: faster attention with many sparsely accessed value heads

April 08, 2025

Daniel Heinlein, Vaclav Cvicek, Akshay Mishra, Mahdi Nazemi, Sanjit
Neelam, and Reiner Pope

During Transformer decoding, KV cache size and memory bandwidth
requirements can [limit
overall throughput](/research/lifetime_llm_cost#inference-and-the-memory-bottleneck). Multi Query Attention ([Shazeer, 2019](https://arxiv.org/abs/1911.02150))
is a powerful technique to mitigate this, but some models such as the
Llama family have not deployed it due to quality concerns, opting for
the less aggressive Grouped Query Attention instead ([Dubey et al., 2024](https://arxiv.org/abs/2407.21783)).
An alternative approach is to sparsely access the attention values ([Sheng et al., 2023](https://proceedings.mlr.press/v202/sheng23a.html))
but this has not been widely adopted—perhaps because the dense access to
attention keys limits the memory bandwidth savings to “only” a factor of
2. We propose a way to combine Multi Query Attention with sparsification
by using multiple sparsely-accessed value heads and a single
densely-accessed key head. Additionally, we present an approach to
sparsifying the value heads that is more computationally efficient and
can also achieve higher sparsity levels. Our approach achieves the
quality of Grouped Query Attention with the memory bandwidth of Multi
Query Attention, thus reducing memory bandwidth costs by up to a factor
of 8 for common model architectures.

![](../static/smva_tensorsizes.svg)

Figure 1: Comparison of sizes of tensors, the white area of the KV
cache is not loaded. We propose SMVA which uses one K head and many V
heads whose KV cache entries are accessed sparsely.

## Introduction

The key-value (KV) cache is the state that is kept between forward
passes of large language models (LLMs), to allow incremental decoding
without repeated computations. We propose two methods to reduce the
bandwidth demands of loading entries from the KV cache.

- *Multi Value Attention (MVA)* allows for different numbers
  of K heads and V heads and hence, can be used to reduce the number of K
  heads compared to Multi Head Attention (MHA ([Vaswani et al., 2017](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf))).
  Typically, we set the number of K heads to 1 and the number of V heads
  equal to the number of Q heads.
- *Sparse V* decreases the bandwidth usage of loading
  entries from the KV cache by loading only the most impactful entries in
  the V part of the KV cache.

These techniques work very well together. Ultimately, we propose
*Sparse Multi Value Attention (SMVA)*: use one K head, as many V
heads as possible (equal to the number of Q heads), and access the V
heads sparsely.

![](../static/smva_mechanism.svg)

Figure 2: Example of SMVA with 2 Q heads and 2 V heads. The Sparse V
threshold is 20% and V’s with dotted boxes are not loaded.

## Multi Value Attention

We decouple the number of K heads from the number of V heads, partly
because V heads are more important for quality than K heads, and also
because V heads are more easily accessed sparsely. Decoupling the number
of heads is straightforward when the ratio of V heads to K heads is an
integer. We also show how to do it in general, based on an approach
which forms partitions of heads based on the greatest common divisor of
K and V. For implementation details see the pseudocode [below](#implementation).

We differ from GQA ([Ainslie et al., 2023](https://doi.org/10.18653/v1/2023.emnlp-main.298)),
MQA ([Shazeer, 2019](https://arxiv.org/abs/1911.02150)),
and MHA ([Vaswani et al., 2017](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf))
in decoupling the number of K heads and V heads. MVA combines the high
throughput of using fewer K heads, as in MQA, with the quality benefits
of many sparsely accessed (and therefore cheaper) V heads, as in
MHA.

## Sparse V

We sparsify the reads of V by setting a fixed threshold and zeroing
out any attention probabilities lower than that threshold. This is
equivalent to loading only a subset of the entries in the V part of the
KV cache and fixing the others to zero.

We differ from ([Sheng et al., 2023](https://proceedings.mlr.press/v202/sheng23a.html))
in two ways:

- We introduce V-sparsity during training, rather than after
  training. We find this allows us to achieve 4x greater sparsity at the
  same quality levels. We found that enabling sparsity upon reaching 60%
  of the training steps works best but any value in the range 20%-70% also
  works well.
- Rather than inducing sparsity via a top-k computation, we induce
  it by comparing the attention probabilities to a fixed threshold. We
  find this achieves the same sparsity and quality levels, while allowing
  us to replace an expensive O(nlog⁡2(k))\mathcal{O}(n \log^2(k))O(nlog2(k)) sorting network implementation of top-k with a much
  cheaper O(n)\mathcal{O}(n)O(n) threshold
  computation.

Sparse V is incompatible with fused K/V attention implementations
such as Flash Attention ([Dao et al., 2022](https://proceedings.neurips.cc/paper_files/paper/2022/file/67d57c32e20fd0a7a302cb81d36e40d5-Paper-Conference.pdf))
for two reasons. First, the probability-based threshold requires fully
normalized probabilities, defeating the online softmax optimization at
the heart of Flash Attention. Second, the pattern of memory fetches for
V is dependent on the results of the K computation, defeating the
ability to prefetch the K and V tensors in lockstep. Nevertheless, the
memory bandwidth saved by Sparse V far exceeds the memory bandwidth
spent by losing K/V fusion.

## Implementation

We implemented a modified attention mechanism that allows for
flexible reshaping of the Q, K, and V heads to implement our techniques.
The following pseudocode based on our [seqax](https://github.com/MatX-inc/seqax/tree/SMVA) codebase
demonstrates this variation. Sparse V is activated by providing a
non-zero `p_threshold`. For brevity, we have omitted details
such as sharding, RoPE embedding, the causal mask, and batch processing.
Also for simplicity, we fetch the V tensor densely rather than sparsely,
even though it is multiplied by sparse `probs`.

```
q = einsum("Qlen M, M Q K V G D -> Qlen Q K V G D", x, w_q)
k = einsum("Klen M, M K G D -> Klen K G D", x, w_k)
v = einsum("Klen M, M V G D -> Klen V G D", x, w_v)
logits = einsum("Qlen Q K V G D, Klen K G D -> Qlen Klen Q K V G", q, k)
probs = softmax(logits, axis = "Klen")
pt = p_threshold * (p_threshold_steps_fraction * max_steps <= step)
probs = where(probs < pt, 0, probs)
attn_out = einsum("Qlen Klen Q K V G, Klen V G D -> Qlen Q K V G D", probs, v)
y = einsum("Qlen Q K V G D, M Q K V G D -> Qlen M", attn_out, w_o)
```

Algorithm 1: Sparse Multi Value Attention (SMVA).

## Experiments

To evaluate our methods, we trained multiple variants of a 101
million parameter model using the [seqax](https://github.com/MatX-inc/seqax/blob/SMVA/configs/101m150.yaml)
codebase. We trained the model for 150,000 steps (9.8B tokens, batch 32,
sequence length 2048), approximately five times the number of tokens
suggested by the Chinchilla scaling laws ([Hoffmann et al., 2024](https://papers.nips.cc/paper_files/paper/2022/hash/c1e2faff6f588870935f114ebe04a3e5-Abstract-Conference.html),
Table 3). Similarly to Llama 3 ([Dubey et al., 2024](https://arxiv.org/abs/2407.21783)),
we prioritized a longer training duration to optimize the model for
inference.

Our training dataset is the Allen Institute for AI’s version of the
[C4/en](https://huggingface.co/datasets/allenai/c4) dataset,
tokenized using [Llama 2’s
tokenizer](https://huggingface.co/meta-llama/Llama-2-7b-hf). The reported loss is the training loss averaged over the
last 1,024 training steps (67 million tokens).

To minimize noise in our experiments, all experiments visit the
training data in the same random order, and all models are initialized
using the same random seed. Thanks to JAX’s hash-based randomization,
changing to the shape of one weight matrix does not affect the random
initialization of other weight matrices. As a result, changes in loss
even as small as 0.03% are consistently reproducible.

In addition to measuring the loss, we also compute a [*FLOP-equivalent cost*](/research/lifetime_llm_cost) with a
roofline model that estimates the decoding cost as the maximum of the KV
cache fetch time and computation time.

### Experiments with different attention types

We compared different attention mechanisms in the table below. For
each model, we report the loss, FLOPs, and KV cache requirements, all
relative to MHA.

| Attention | Q,K,V Heads | dffd\_{\text{ff}}dff​ | Sparsity | Loss | Sparse FLOPs | Dense FLOPs | KV Size | KV Fetched | [FLOP-Equivalent Cost](/research/lifetime_llm_cost) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MHA | 8,8,8 | 4096 | - | **2.862 (+0.0%)** | - | **1.00** | 1.00 | 1.00 | 1.00 |
| GQA | 8,4,4 | 4437 | - | 2.866 (+0.1%) | - | **1.00** | 0.50 | 0.50 | 0.50 |
| GQA | 8,2,2 | 4608 | - | 2.870 (+0.3%) | - | **1.00** | 0.25 | 0.25 | 0.25 |
| MQA | 8,1,1 | 4694 | - | 2.879 (+0.6%) | - | **1.00** | **0.13** | 0.13 | 0.13 |
| MQA | 15,1,1 | 4096 | - | 2.874 (+0.4%) | - | 1.29 | **0.13** | 0.13 | 0.13 |
| MVA | 8,1,8 | 4395 | - | 2.863 (+0.0%) | - | **1.00** | 0.56 | 0.56 | 0.56 |
| MHA | 8,8,8 | 4096 | Sparse V | 2.868 (+0.2%) | **0.84** | **1.00** | 1.00 | 0.55 | 0.58 |
| MHA | 8,8,8 | 4096 | top-k | 2.868 (+0.2%) | **0.84** | **1.00** | 1.00 | 0.55 | 0.58 |
| MQA | 15,1,1 | 4096 | Sparse V | 2.879 (+0.6%) | 0.99 | 1.29 | **0.13** | 0.12 | **0.12** |
| SMVA | 10,1,10 | 4096 | Sparse V | 2.868 (+0.2%) | 0.88 | 1.08 | 0.69 | 0.12 | 0.17 |
| SMVA | 8,1,8 | 4395 | Sparse V | 2.868 (+0.2%) | **0.84** | **1.00** | 0.56 | **0.11** | 0.15 |

Table 1: Comparison of normalized quality and costs of different
attention mechanisms with fixed parameter count. Our proposed model,
SMVA with 8,1,8 heads beats the quality of GQA with 2 KV heads while
having 40% less FLOP-equivalent cost.

GQA reduces the number of KV heads compared to MHA, decreasing the KV
cache size and memory bandwidth requirements but slightly increasing the
loss. To ensure a fair comparison and maintain a constant number of
parameters across models, we adjusted the feed-forward dimension dffd\_{\text{ff}}dff​ accordingly.

MQA reduces the number of KV heads all the way down to one,
significantly decreasing the KV cache size. However, we observed a
slight deterioration in model quality, as indicated by an increase in
loss.

To recover the original loss, we increased the number of V heads to 8
while keeping the number of K heads at 1, implementing our proposed MVA.
This adjustment restored the model’s quality close to that of MHA but
increased the total KV cache size again.

Finally, we applied Sparse V by accessing the V part of the KV cache
sparsely, considering only entries of the attention probabilities larger
than 0.01. Further details on our sparsity approach are provided
below.

Our two proposed changes (SMVA with 8 Q heads) achieved better
quality than GQA with 2 KV heads (0.2% vs. 0.3%), required less
FLOP-equivalent cost than GQA with 2 KV heads (84% vs. 100%), had a
smaller KV cache than MHA (56% vs. 100%), and required less KV cache
bandwidth than MQA (11% vs. 13%) and hence resulted in 15%
FLOP-equivalent cost in parameter-controlled experiments, which is on
par with MQA (13%).

In summary, among all experiments with loss within 0.2% of MHA, SMVA
(with 8 Q heads) achieved the lowest amount of FLOPs needed for the
forward pass, reduced the KV cache size by half compared to MHA, and had
the smallest KV cache bandwidth requirements. Consequently, it had the
smallest FLOP-equivalent cost.

### Sparse V experiments

To evaluate various Sparse V approaches, we experimented with
different values of the probability threshold pthreshold∈{0.001,0.01,0.1}p\_{\text{threshold}} \in \{0.001, 0.01, 0.1\}pthreshold​∈{0.001,0.01,0.1}
and the threshold activation step proportion pthreshold, steps, fraction∈{0.0,0.3,0.6,0.9}p\_{\text{threshold, steps, fraction}} \in \{0.0, 0.3, 0.6, 0.9\}pthreshold, steps, fraction​∈{0.0,0.3,0.6,0.9}. Across various model sizes and various multiples
of Chinchilla optimal training, we found that setting pthreshold=0.01p\_{\text{threshold}} = 0.01pthreshold​=0.01 and pthreshold, steps, fraction=0.6p\_{\text{threshold, steps, fraction}} = 0.6pthreshold, steps, fraction​=0.6
achieves a loss comparable to MHA (within 0.2%). Therefore, we fix these
parameters throughout this blog post.

We compared our threshold-based Sparse V approach with the top-k
Sparse V method presented in ([Sheng et al., 2023](https://proceedings.mlr.press/v202/sheng23a.html)),
where MHA is augmented post-training by sparsely accessing the V part of
the KV cache using top-k selection.

Setting pthreshold=0.01p\_{\text{threshold}} = 0.01pthreshold​=0.01
allows up to 100 elements of the attention matrix (along the
`Klen` axis) to be nonzero. Therefore, comparing with top-k
where k=100k = 100k=100 offers a direct
comparison.

In our reproductions, when sparsifying post-training with top-k and
k=100k = 100k=100, the loss increased
significantly (by 1.5%). To achieve a loss increase of only 0.2%
compared to MHA, kkk needed to be about
400, resulting in about four times less sparsity than our Sparse V
method at similar quality levels.

However, when sparsification was applied already during part of the
training, MHA with top-k and k=100k = 100k=100
achieved the low loss increase of 0.2%. This indicates that sparsifying
during training is very advantageous. Additionally, sparsifying with a
threshold is simpler and faster than using full top-k selection.

## Conclusion

We have proposed two new methods, Sparse V and Multi Value Attention
(MVA), and combine both into a single method, Sparse Multi Value
Attention (SMVA), to address the bandwidth and memory challenges of KV
caching in Transformer models. Sparse V loads only the most important
values, while MVA decreases the number of K heads, both leading to
savings in memory, compute, and bandwidth. Our experiments show that
these methods maintain similar model quality to Grouped Query Attention
(GQA) but with lower resource demands. These improvements could help
make large language models (LLMs) more efficient and scalable,
especially for real-time applications. Future work can focus on further
optimizing these techniques and exploring their impact on even larger
models.

## Acknowledgments

Research supported with Cloud TPUs from Google’s TPU Research Cloud
(TRC).
