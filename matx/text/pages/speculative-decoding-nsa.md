# Speculative Decoding with Blockwise Sparse Attention

July 22, 2025

Sanjit Neelam, Vaclav Cvicek, Daniel Heinlein, Akshay Mishra, Mahdi
Nazemi, and Gilbert Hendry

Speculative decoding (SD) and blockwise sparse attention both
accelerate LLM decoding, but when combined naively, the KV cache may
lose sparsity during the verification step of SD. We show that forcing
all draft tokens to attend to the same subset of the context restores
sparsity while preserving model quality.

## Introduction

Scaling the context length of LLMs continues to be crucial for
improving their capabilities and unlocking new categories of
applications. However, this leaves a [greater proportion](https://arxiv.org/pdf/2403.14123) of newer
ML accelerators underutilized, since it decreases the [operational
intensity](https://en.wikipedia.org/wiki/Roofline_model#Arithmetic_intensity) of generating tokens.

New attention mechanisms such as [DeepSeek’s Native Sparse
Attention (NSA)](https://arxiv.org/abs/2502.11089v2) and [Moonshot AI’s Mixture of Block
Attention (MoBA)](https://arxiv.org/abs/2502.13189) divide the context into blocks and allow each token
to dynamically attend to just a subset of these blocks. By reducing the
volume of data that needs to be moved from high bandwidth memory (HBM)
into compute cores, these “blockwise sparse” methods aim to increase the
operational intensity of decoding.

Speculative decoding (SD) similarly increases the operational
intensity of decoding, and has been widely implemented to reduce latency
or increase throughput by around 2×. Unfortunately, naively combining
blockwise sparsity with SD by using a blockwise sparse target model
falls short of its potential: when *each* draft token loads a
different set of blocks from HBM, the operational intensity of
verification drops significantly.

In this article, we train NSA models in which *a block* of
tokens attend to the same subset of the context in the “token selection”
attention path. We show that these models have the same language
modeling quality as baseline NSA models, while achieving up to 3.5×
higher operational intensity during the verification step of speculative
decoding.

[Our
implementation of NSA](https://github.com/MatX-inc/seqax/blob/e2c8151532c12569385375f3177f212dc96fa7ac/model.py#L210-L335), [a supporting
notebook](https://github.com/MatX-inc/seqax/blob/NSA/nsa.ipynb), and [code used
to produce figures](https://github.com/MatX-inc/seqax/blob/NSA/blog.ipynb) are available at <https://github.com/MatX-inc/seqax/tree/NSA>.

## Verification may be much slower with a blockwise sparse target model

Speculative decoding uses a cheap draft model to generate kkk candidate tokens that are verified in
parallel by a target model. SD reduces the latency or increases the
throughput of decoding by the [factor](https://matx.com/research/sd#evaluation-metric) Speedup=E[# Tokens Generated per Round of Speculation](k×tdraft+tverify)/tdecode\begin{align\*} \text{Speedup} &= \frac{\mathbb{E}[\text{\# Tokens Generated per Round of Speculation}]}{(k \times t\_\text{draft} + t\_\text{verify}) / t\_\text{decode}} \end{align\*}Speedup​=(k×tdraft​+tverify​)/tdecode​E[# Tokens Generated per Round of Speculation]​​ where a *round of speculation* is the generation and
subsequent verification of each block of kkk draft tokens, tdraftt\_\text{draft}tdraft​ and tdecodet\_\text{decode}tdecode​ are the latencies of a draft
and target model forward pass on a single token, and tverifyt\_\text{verify}tverify​ is the latency of a target
model forward pass on k+1k+1k+1 tokens.

If tdraft≪tverifyt\_\text{draft} \ll t\_\text{verify}tdraft​≪tverify​ and we can change tverifyt\_\text{verify}tverify​ without changing kkk, tdecodet\_\text{decode}tdecode​, or the expected number of
tokens generated, the speedup is inversely proportional to tverifyt\_\text{verify}tverify​. Since verification with a
long context has low operational intensity, tverifyt\_\text{verify}tverify​ is directly proportional to
the number of context tokens ℓctx\ell\_\text{ctx}ℓctx​ that must be loaded. Thus,
reducing ℓctx\ell\_\text{ctx}ℓctx​ by some factor
increases the speedup due to SD by the same factor.

| Context Length | 8192 | 16384 | 32768 | 65536 |
| --- | --- | --- | --- | --- |
| Full Attention | 8192 | 16384 | 32768 | 65536 |
| NSA (verification best case) | 2048 | 2560 | 3584 | 5632 |
| NSA (verification worst case) | 5120 | 5632 | 6656 | 8704 |
| NSA (reduction in memory access volume in best case compared to worst case) | **2.5×** | **2.2×** | **1.9×** | **1.5×** |

Table 1: Memory access volume (in equivalent number of tokens) during
a forward pass (the size of the KV cache is the same when decoding one
token or verifying kkk draft tokens.
Here, k=3k=3k=3). Adapted from [Yuan et al.](https://arxiv.org/abs/2502.11089)’s Table 4.

[Table 1](#table:1) shows the difference in ℓctx\ell\_\text{ctx}ℓctx​ in the best and worst cases
when k=3k = 3k=3. In the best case, each
draft token attends to the same subset of the context, and in the worst
case, each draft token attends to a different subset of the context. In
the limit as the batch size tends to infinity (or equivalently, assuming
that the latency of loading the model’s parameters from HBM is
negligible), [Figure 1](#fig:1) shows that the operational
intensity of verification (and therefore the speedup due to SD) may be
over 3.5× higher in the best case than in the worst case.

![](../static/sd_nsa_1.svg)

Figure 1: Operational intensity of verification as a function of the
number of draft tokens for target models with 3B active parameters, 30
layers, 4 GQA groups, key dimension dk=192d\_k=192dk​=192, and value dimension dv=128d\_v=128dv​=128. In the best case, each draft token
selects the same subset of the context, and in the worst case, each
draft token selects a different subset of the context.

## We can force a block of tokens to attend to the same subset of the context

To *always* achieve the best case operational intensity during
verification, we replace KVt+1slc,KVt+2slc,…,KVt+kslc\text{KV}\_{t + 1}^\text{slc}, \text{KV}\_{t + 2}^\text{slc}, \dots, \text{KV}\_{t + k}^\text{slc}KVt+1slc​,KVt+2slc​,…,KVt+kslc​ with KVtslc\text{KV}\_t^\text{slc}KVtslc​, where KVtslc\text{KV}\_t^\text{slc}KVtslc​ are the keys and
values selected by token ttt. During
training, for each 1≤t≤Qlen1 \leq t \leq \texttt{Qlen}1≤t≤Qlen which is a multiple of k+1k+1k+1, KVtslc\text{KV}\_t^\text{slc}KVtslc​ is reused by the kkk tokens following token ttt. During each verification forward pass, the
query sequence length Qlen\texttt{Qlen}Qlen is
equal to k+1k+1k+1, so the keys and values
selected by the first token are attended to by all subsequent
tokens.

We simulate loading only selected keys and values from HBM using the
attention mask shown in [Figure 2 (left)](#fig:2). Forcing a
block of query tokens to attend to the same subset of the context
corresponds to using an attention mask like the one shown in [Figure 2 (right)](#fig:2). [Figure 3](#fig:3) zooms
in on the last k+1k+1k+1 rows of each mask
(here, k=3k=3k=3) and shows that in this
instance, our modification lets us load 14−4=1014 - 4 = 1014−4=10 fewer KV blocks during verification.

![](../static/sd_nsa_2.svg)

Figure 2: Left is a selected attention mask constructed using
uniformly random importance scores ptslc\mathbf{p}\_t^\text{slc}ptslc​. Right is the
attention mask obtained by applying our modification to left, which
forces blocks of k+1k+1k+1 tokens (here,
k=3k=3k=3) to select the same subset of the
context. As in the selected attention mask in [Yuan et al.](https://arxiv.org/abs/2502.11089)’s Figure 2,
yellow squares indicate which attention scores must be computed.

![](../static/sd_nsa_3.svg)

Figure 3: Above are the last four rows of [Figure 2
(left)](#fig:2) and below are the last four rows of [Figure 2
(right)](#fig:2). Both are examples of a selected attention mask when
performing a forward pass on k+1k+1k+1 tokens
in parallel (here, k=3k=3k=3).

[Table 2](#table:2) shows that for models with 184 million
and 1.2 billion parameters, the cross-entropy on a slice of the [LongCrawl64](https://manifestai.com/articles/longcrawl64/)
validation set is approximately equal for different numbers of draft
tokens kkk. We train all models on 10B
tokens from a slice of the [LongCrawl64](https://manifestai.com/articles/longcrawl64/)
training set, with a sequence length of 2048. The attention sublayer in
each model is our best interpretation of NSA presented by [Yuan et al.](https://arxiv.org/abs/2502.11089), and we use
l=32l=32l=32, d=16d=16d=16, l′=64l'=64l′=64, n=4n=4n=4, and w=128w=128w=128. The feed-forward sublayer in each
model is a dense (SwiGLU) layer rather than a Mixture of Experts (MoE)
layer, and we use multiple RMSNorms in each sublayer. All models are
trained for 19070 steps using the AdamW optimizer with a batch size of
256, β1=0.9,β2=0.95\beta\_1 = 0.9, \beta\_2 = 0.95β1​=0.9,β2​=0.95, and
with a weight\_decay of 0.10.10.1. Over the
first 1907 steps we linearly increase the learning rate from 0 to one of
the learning rates in {6.5e−3,3e−3}\{6.5e^{-3}, 3e^{-3}\}{6.5e−3,3e−3} (using a smaller peak learning rate for larger models),
before decaying it to 1% of the peak learning rate following a cosine
decay schedule.

| Parameters | NSA | k=1k=1k=1 | k=3k=3k=3 | k=7k=7k=7 |
| --- | --- | --- | --- | --- |
| 186M | 2.128 | 2.128 | 2.128 | 2.13 |
| 1177M | 1.794 | 1.794 | 1.793 | 1.794 |

Table 2: Cross-entropy on a slice of the [LongCrawl64](https://manifestai.com/articles/longcrawl64/)
validation set is approximately equal across different numbers of draft
tokens kkk for models with 184 million
and 1.2 billion parameters.

[Figure 4](#fig:4) shows that training loss curves for
models with 1.2 billion parameters are essentially identical.

Figure 4: (Interactive) loss curves show that forcing blocks of k+1k+1k+1 tokens to attend to the same subset of
the context maintains model quality for k∈{0,1,3,7}k \in \{0, 1, 3, 7\}k∈{0,1,3,7}.

## Ablations

[Yuan et al.](https://arxiv.org/abs/2502.11089) always
select the first and last two blocks in the token selection attention
path. Since our training sequence length is 2048 rather than [Yuan et al.](https://arxiv.org/abs/2502.11089)’s 8192, we
select a total of n=4n=4n=4 rather than n=16n=16n=16 blocks. However, these choices together
may limit the maximum possible effect size of our modification, since
the single dynamically-selected block may barely affect model
quality.

Thus, we try a) training models which do not always select the first
and last two blocks. We also try b) training a model without token
selection, to rule out the possibility that this entire attention path
barely affects model quality, and we try a training-free method c) which
applies our modification at test-time.

|  |  | Description | NSA | k=1k=1k=1 | k=3k=3k=3 | k=7k=7k=7 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | b) | No selection | **2.147** |  |  |  |
| 2 |  | Baseline | **2.128** | **2.128** | **2.128** | 2.130 |
| 3 | a) | Free selection | 2.132 | 2.129 | **2.126** | 2.127 |
| 4 | c) | Test-time only | **2.128** | 2.129 | 2.132 | 2.134 |
| 5 | a), c) | Free sel. test-time | **2.132** | **2.132** | 2.133 | 2.135 |

Table 3: Cross-entropy on a slice of the [LongCrawl64](https://manifestai.com/articles/longcrawl64/)
validation set for models with 184 million parameters. All rows use
n=4n=4n=4 blocks.

[Table 3](#table:3) shows that with ablation a), model
quality varies more but does not degrade as kkk increases from 0 to 7. With the other two
ablations, model quality degrades slightly but monotonically as kkk increases from 0 to 7; comparing row 2 with
4 and row 3 with 5 indicates train-test mismatch. While the gap between
no selection and the baseline (n=4n=4n=4) is
only 0.019, this difference is still meaningful: factors such as GPU
non-determinism, initialization seed, and data ordering led to
variations in cross-entropy of just a couple of thousandths.

We also tried d) selecting a total of n=16n=16n=16 blocks, and observed that model quality
is preserved with any combination of a) and d). Indeed, although token
selection has a greater effect on model quality as nnn increases, for large enough nnn the maximum possible effect size of our
modification is smaller than when n=4n=4n=4,
since there is greater overlap between the subsets of the context
attended to by each token.

## Discussion

[Yuan et al.](https://arxiv.org/abs/2502.11089) say that
their Figure 8 (visualization of attention map) inspired their design of
NSA since they observed “nearby keys often showing similar attention
scores”. We further suggest that this figure shows many different
queries attending to the same keys, and is an alternative motivation for
forcing a block of tokens to attend to the same subset of the
context.

Limitations of our results are that we evaluated small models with
short contexts from a single dataset, and that cross-entropy alone is
insufficient for evaluating long-context performance. Our implementation
of NSA uses dense (SwiGLU) layers rather than Mixture of Experts (MoE)
layers to mix information along the model dimension, and there may be an
interaction between blockwise sparse attention and the type of mixer
used.

We have shown that we can force a block of tokens to attend to the
same subset of the context while preserving model quality. Our ablations
rule out the possibility that selected attention does not affect model
quality with our hyperparameters, and show that training with our
modification reduces train-test mismatch (but may not be necessary when
the number of draft tokens kkk is
small).
