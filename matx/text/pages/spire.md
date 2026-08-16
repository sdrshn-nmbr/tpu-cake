# SPIRe: Boosting LLM Inference Throughput with Speculative Decoding

April 08, 2025

Sanjit Neelam, Daniel Heinlein, Vaclav Cvicek, Akshay Mishra, and Reiner
Pope

Speculative decoding (SD) has been shown to reduce the latency of
autoregressive decoding (AD) by 2-3× for small batch sizes. However,
increasing throughput and therefore reducing the cost per token requires
decoding with large batch sizes. Recent work shows that SD can
accelerate decoding with large batch sizes too if the context is
sufficiently long and the draft model’s KV cache is sparse. We introduce
SPIRe, a draft model that combines static sparse attention, pruned
initialization, and feedback memory to increase the modeled throughput
of speculative decoding by over 100% compared to speculation with a much
smaller draft model and by over 35% compared to the strong baseline of
sparse self-speculation. Our approach is particularly effective when
context lengths vary significantly across requests. A PDF version of
this article is available at <https://arxiv.org/abs/2504.06419>.

## Introduction

Speculative decoding uses a cheap draft model to generate candidate
tokens that are verified in parallel by an expensive target model ([Leviathan et al., 2023](https://arxiv.org/abs/2211.17192); [Chen et al., 2023](https://arxiv.org/abs/2302.01318)). Accepted
draft tokens are guaranteed to be samples from the *target*
model’s distribution over the next token by a modified rejection
sampling scheme. If draft tokens are generated quickly enough and have a
high enough acceptance rate, speculative decoding improves both the
latency and throughput of decoding by reducing the number of memory
accesses needed to generate each token[1](#fn1).

Optimizing speculative decoding for higher throughput requires
different strategies than optimizing for lower latency; the latter is
well-documented in the literature ([Xia et al., 2024](https://arxiv.org/abs/2401.07851); [Miao et al., 2024](https://arxiv.org/abs/2305.09781); [Yan et al., 2025](https://arxiv.org/abs/2402.01528)). For
example, low latency can often be achieved with small batch sizes ([Pope et al., 2022](https://arxiv.org/abs/2211.05102)) and small
draft models are preferred in this setting since fetching weights is the
bottleneck. On the other hand, high throughput requires large batch
sizes, which enables the use of larger draft models since in this regime
loading the KV cache is the bottleneck ([Chen et al., 2024](https://arxiv.org/abs/2408.11049v3)). See [Figure 1](#fig:1) and [Figure 4](#fig:4) for
examples of how the speedup due to speculative decoding varies with
draft model architecture, batch size, and context length according to
our performance model [below](#evaluation-metric).

![](../static/sd_1.svg)

Figure 1: Speedup of generating tokens in a batch of size B=64B=64B=64 given a context of length LLL. A *round of speculation* is the
generation and subsequent verification of each block of k=4k=4k=4 draft tokens, the *average generation
length* τ\tauτ is the average number
of tokens generated per round of speculation, and speedup is τ\tauτ divided by how much longer a round of
speculation takes compared to autoregressively decoding one token.

Prior work has shown that the acceptance rate of draft tokens can be
increased using the target model’s intermediate activations and output
logits ([Du et al., 2024](https://arxiv.org/abs/2402.02082);
[Zhou et al., 2024](https://arxiv.org/abs/2310.08461)). Since
a target model’s state can be cached during training at negligible cost,
and since inference rather than training accounts for most of the cost
of serving an LLM, the investment in training a draft model aligned to a
particular target model may be worthwhile. Indeed, we show [below](#cost-analysis) that this is the case for SPIRe under
conservative assumptions.

Our work, along with [Chen et al., 2024](https://arxiv.org/abs/2408.11049v3),
demonstrates that speculative decoding is an effective technique for
reducing the cost of decoding, and our work enables more efficient
exploration of the design space of draft models based on the batch sizes
and context lengths expected in production. Our main contributions are
the following:

- An implementation-agnostic model for evaluating the throughput of
  speculative decoding with different draft models.
- An efficient method for training a Feedback Transformer draft
  model.
- A draft model, SPIRe, that increases the modeled throughput of
  speculative decoding by over 100% compared to [vanilla speculative decoding](#evaluation) and by over 35%
  compared to the strong baseline of MagicDec ([Chen et al., 2024](https://arxiv.org/abs/2408.11049v3)).

## Evaluation Metric

It is problematic to compare different methods of training the draft
model using the throughput achieved by the resulting models in a
production environment. Measured throughput depends on myriad
implementation details such as use of specialized kernels ([Dao et al., 2022](https://arxiv.org/abs/2205.14135)), how
models are partitioned across chips ([Pope et al., 2022](https://arxiv.org/abs/2211.05102)), how
queries are batched ([Daniel
et al., 2023](https://www.anyscale.com/blog/continuous-batching-llm-inference)), and which queries are used for evaluation ([Zhou et al., 2024](https://arxiv.org/abs/2310.08461)). Any of
these variables could be confounders when comparing different draft
model architectures.

To derive a proxy for measured throughput, we first note that the
throughput of speculative decoding is equal to the throughput of vanilla
decoding multiplied by ThroughputMultiplier=E[# Tokens Generated per Round of Speculation]k×tdraft+tverifyttarget\begin{align\*} \begin{array}{c} \text{Throughput} \\ \text{Multiplier} \end{array} &= \frac{\mathbb{E}[\text{\# Tokens Generated per Round of Speculation}]}{\frac{k \times t\_\text{draft} + t\_\text{verify}}{t\_\text{target}}} \end{align\*}ThroughputMultiplier​​=ttarget​k×tdraft​+tverify​​E[# Tokens Generated per Round of Speculation]​​ where kkk is the *maximum
speculation depth*, a *round of speculation* is the
generation and subsequent verification of each block of kkk draft tokens, tdraftt\_\text{draft}tdraft​ and ttargett\_\text{target}ttarget​ are the latencies of a draft
and target model forward pass on a single token, and tverifyt\_\text{verify}tverify​ is the latency of a target
model forward pass on k+1k+1k+1 draft tokens
in parallel. In the sequel, we refer to the *average generation
length* τ:=E[# Tokens Generated per Round of Speculation]\tau := \mathbb{E}[\text{\# Tokens Generated per Round of Speculation}]τ:=E[# Tokens Generated per Round of Speculation] and the *iteration time
multiplier* Δt:=k×tdraft+tverifyttarget\Delta t := \frac{k \times t\_\text{draft} + t\_\text{verify}}{t\_\text{target}}Δt:=ttarget​k×tdraft​+tverify​​, so that the
throughput multiplier is simply τ/Δt\tau / \Delta tτ/Δt.

To obtain a more implementation-agnostic expression for Δt\Delta tΔt, we can break down the cost of a
forward pass into the compute cost and the memory cost, which are the
number of FLOPs and the number of memory accesses that need to be
performed. The memory cost can be further broken down into the size of
the LLM’s weights and the size of its KV cache. By multiplying the
memory cost in bytes with the hardware operational intensity (HOI) in
FLOPs per byte, we can obtain an estimate of the memory cost in [FLOP-equivalents](https://matx.com/research/lifetime_llm_cost#flop-equivalent)
and compare the compute and memory costs on equal footing. Assuming that
computation and memory accesses are maximally overlapped[2](#fn2), we
obtain the following expression for the cost of a forward pass. Forward Cost=max⁡(Compute Cost⏟FLOPs,(Weight Cost+KV Cache Cost)⏟Bytes×HOI⏟Memory Cost (FLOP-equivalents))\begin{align\*} \text{Forward Cost} &= \max(\underbrace{\text{Compute Cost}}\_{\text{FLOPs}}, \underbrace{\underbrace{(\text{Weight Cost} + \text{KV Cache Cost})}\_{\text{Bytes}} \times \text{HOI}}\_{\text{Memory Cost (FLOP-equivalents)}}) \end{align\*}Forward Cost​=max(FLOPsCompute Cost​​,Memory Cost (FLOP-equivalents)Bytes(Weight Cost+KV Cache Cost)​​×HOI​​)​

Finally, by substituting tdraftt\_{\text{draft}}tdraft​, tverifyt\_{\text{verify}}tverify​, and ttargett\_{\text{target}}ttarget​ with their corresponding
forward costs, we obtain the following implementation-agnostic[3](#fn3) approximation of the throughput
multiplier and use it as our evaluation metric. ThroughputMultiplier≈E[# Tokens Generated per Round of Speculation]k×max⁡(Cdraft,(Ndraft+KVdraft)×HOI)+max⁡((k+1)×Ctarget,(Ntarget+KVtarget)×HOI)max⁡(Ctarget,(Ntarget+KVtarget)×HOI)\begin{align\*} \begin{array}{c} \text{Throughput} \\ \text{Multiplier} \end{array} \approx \frac{\mathbb{E}[\text{\# Tokens Generated per Round of Speculation}]}{\frac{k \times \max(C\_\text{draft}, (N\_\text{draft} + \text{KV}\_\text{draft}) \times \text{HOI}) + \max((k + 1) \times C\_\text{target}, (N\_\text{target} + \text{KV}\_\text{target}) \times \text{HOI})}{\max(C\_\text{target}, (N\_\text{target} + \text{KV}\_\text{target}) \times \text{HOI})}} \end{align\*}ThroughputMultiplier​≈max(Ctarget​,(Ntarget​+KVtarget​)×HOI)k×max(Cdraft​,(Ndraft​+KVdraft​)×HOI)+max((k+1)×Ctarget​,(Ntarget​+KVtarget​)×HOI)​E[# Tokens Generated per Round of Speculation]​​ Here, CdraftC\_\text{draft}Cdraft​ is the
number of FLOPs performed during a forward pass through the draft model,
NdraftN\_\text{draft}Ndraft​ is the size of the draft
model’s weights in bytes, KVdraft\text{KV}\_\text{draft}KVdraft​ is the size of the
draft model’s KV cache in bytes, and similarly for CtargetC\_\text{target}Ctarget​, NtargetN\_\text{target}Ntarget​, and KVtarget\text{KV}\_\text{target}KVtarget​.

## SPIRe

We combine the following techniques when training our draft model
SPIRe in order to increase the throughput multiplier of speculative
decoding. The name SPIRe stands for Sparse KV cache, Pruned
Initialization, and Recurrent Feedback Transformer.

1. **Sliding window KV cache with attention sink**

   In the large-batch long-context regime, the memory cost of loading
   the KV cache is the dominant cost of decoding, and reducing the size of
   the KV cache can reduce the iteration time multiplier Δt\Delta tΔt by a larger factor than it reduces
   the average generation length τ\tauτ.
   Following MagicDec we use a StreamingLLM attention mask[4](#fn4) ([Xiao et al., 2024](https://arxiv.org/abs/2309.17453)), during
   training and inference, to ensure the draft model’s KV cache is sparse
   and that the size of the KV cache is constant with respect to the
   decoding sequence length. As shown in [Figure 4](#fig:4),
   this results in an increase in the throughput multiplier as the batch
   size increases, which is never the case for vanilla speculative
   decoding.
2. **Initialize by pruning the target model**

   In our experiments with an eight-layer target model, we initialized
   the draft model using the embedding layer, the last two transformer
   blocks, and the unembedding layer of the trained target model. As shown
   in [Figure 5](#ablations), initializing by pruning the target
   model produced a significantly higher value of τ\tauτ than random initialization. Matching the
   embedding dimension of the target and draft models allowed us to further
   improve the quality of the draft model, as we’ll explain below. Future
   work could explore more sophisticated pruning strategies, such as the
   one described by [Muralidharan
   et al., 2024](https://arxiv.org/abs/2407.14679).

   With short-to-medium contexts, MagicDec may underperform vanilla
   speculative decoding and even autoregressive decoding because the large
   increase in FLOPs outweighs the small decrease in KV cache loads. Given
   that context lengths vary considerably in production, we sought a draft
   model which accelerates decoding across a wide range of context lengths.
   Consequently, our draft model has 14\frac{1}{4}41​th as many parameters as the
   target model (ignoring embedding and unembedding parameters, which
   become negligible as the models are scaled up).
3. **Feedback Transformer and attending to target model
   activations**

   Feedback memory ([Fan et
   al., 2021](https://arxiv.org/abs/2002.09402)) was proposed to increase the representation capacity of
   Transformers, creating Feedback Transformers with stronger performance
   at any given inference cost. They are significantly more expensive to
   train, however, because they inhibit parallelism over the context
   length. Fortunately, we find that this disadvantage can be almost
   entirely mitigated when training a draft model, since we can arrange for
   very short draft rollouts. Furthermore, the additional performance due
   to feedback memory has been shown to be even greater for shallow models
   such as ours, and a shallower draft model is preferable for its smaller
   forward pass latency tdraftt\_\text{draft}tdraft​.

   The ℓ\ellℓ-th attention sublayer of
   the Feedback Transformer computes ztℓ=Attention(xtℓ,m<t)\begin{align\*} \mathbf{z}\_t^{\ell} = \text{Attention}(\mathbf{x}\_t^{\ell}, \mathbf{m}\_{<t}) \end{align\*}ztℓ​=Attention(xtℓ​,m<t​)​ using memory vectors mt\mathbf{m}\_tmt​ instead of the standard ztℓ=Attention(xtℓ,x<tℓ)\mathbf{z}\_t^{\ell} = \text{Attention}(\mathbf{x}\_t^{\ell}, \mathbf{x}\_{<t}^{\ell})ztℓ​=Attention(xtℓ​,x<tℓ​).
   The memory vectors mt\mathbf{m}\_tmt​ are
   defined as mt=∑ℓ=0nlayerexp⁡(wℓ)⋅xtℓ∑ℓ=0nlayerexp⁡(wℓ)\begin{align\*} \mathbf{m}\_t = \frac{\sum\_{\ell=0}^{n\_\text{layer}} \exp(w\_\ell) \cdot \mathbf{x}\_t^{\ell}}{\sum\_{\ell=0}^{n\_\text{layer}} \exp(w\_\ell)} \end{align\*}mt​=∑ℓ=0nlayer​​exp(wℓ​)∑ℓ=0nlayer​​exp(wℓ​)⋅xtℓ​​​ where xtℓ\mathbf{x}\_t^{\ell}xtℓ​ is
   the activation of the ttt-th token after
   the ℓ\ellℓ-th layer, (wℓ)∈Rnlayer+1(w\_\ell) \in \mathbb{R}^{{n\_\text{layer}}+1}(wℓ​)∈Rnlayer​+1
   are learnable weights, and layer 0 is the embedding layer. The
   dependence of each memory vector on all activations from the previous
   timestep does not increase the latency of decoding tokens
   autoregressively, but it does prevent training and prefill from being
   parallelized.

   We require our Feedback Transformer draft model to be able to
   generate blocks of kkk tokens given
   contexts of various lengths, where kkk is
   the maximum speculation depth. Thus during training, rather than
   performing SSS forward passes on a prefix
   of length SSS, we perform just kkk forward passes: we forego computing the
   first S−kS - kS−k memory vectors by
   substituting them with target model activations.

   ![](../static/sd_spire_mask.svg)

   Figure 2: The matrix QK⊤QK^\topQK⊤ in the 1st
   and 4th forward passes on a training sequence of length L=12L=12L=12 with a maximum speculation depth of
   k=4k=4k=4. In general we perform kkk forward passes on every kkk-th prefix of a sequence, in a process
   similar to batched autoregressive decoding with teacher forcing.

   Specifically, during training, our Feedback Transformer draft model
   autoregressively generates k=4k=4k=4 tokens
   following every kkk-th prefix of a
   sequence by performing kkk forward
   passes. Unlike [Fan et al.,
   2021](https://arxiv.org/abs/2002.09402), we share neither key and value projections nor memory vectors
   across layers: for t=S−k+1,…,St = S - k + 1, \dots, St=S−k+1,…,S, where S∈{k,2k,…,L}S \in \{k, 2k, \dots, L\}S∈{k,2k,…,L} is the length of a prefix, the memory vectors mti\mathbf{m}\_t^imti​ for the iii-th layer are defined as mti=∑ℓ=0nlayerexp⁡(wiℓ)⋅xtℓ∑ℓ=0nlayerexp⁡(wiℓ)\begin{align\*} \mathbf{m}\_t^i = \frac{\sum\_{\ell=0}^{n\_\text{layer}} \exp(w\_{i \ell}) \cdot \mathbf{x}\_t^{\ell}}{\sum\_{\ell=0}^{n\_\text{layer}} \exp(w\_{i \ell})} \end{align\*}mti​=∑ℓ=0nlayer​​exp(wiℓ​)∑ℓ=0nlayer​​exp(wiℓ​)⋅xtℓ​​​ where (wiℓ)∈Rnlayer×(nlayer+1)(w\_{i \ell}) \in \mathbb{R}^{{n\_\text{layer}} \times ({n\_\text{layer}} + 1)}(wiℓ​)∈Rnlayer​×(nlayer​+1) are
   learnable weights. For t=1,…,S−kt = 1, \dots, S - kt=1,…,S−k, we let mti=yti+6−1\mathbf{m}\_t^i = \mathbf{y}\_t^{i + 6 - 1}mti​=yti+6−1​, where ytℓ\mathbf{y}\_t^{\ell}ytℓ​ is the target model
   activation of the ttt-th token after the
   ℓ\ellℓ-th layer. In other words, the
   weights of the first (second) layer of the draft model are initialized
   from the weights of the seventh (eighth) layer of the target model, and
   we let mt1=yt6\mathbf{m}\_t^1 = \mathbf{y}\_t^6mt1​=yt6​,
   mt2=yt7\mathbf{m}\_t^2 = \mathbf{y}\_t^7mt2​=yt7​ for
   t=1,…,S−kt = 1, \dots, S - kt=1,…,S−k.

   If we did not match the embedding dimension of the draft and target
   model, we could have substituted past memory vectors with draft model
   embeddings, but this was worse than substituting with target model
   activations in our experiments. Our approach was also better than
   sharing memory vectors across layers and substituting past memory
   vectors with either draft model embeddings or post-final-layer target
   model activations.
4. **Distillation loss**

   Early implementations of speculative decoding for LLMs ([Leviathan et al., 2023](https://arxiv.org/abs/2211.17192); [Chen et al., 2023](https://arxiv.org/abs/2302.01318)) optimize
   the draft model’s parameters by minimizing the standard cross-entropy
   loss HardTargetLoss=Ex<t∼D[−∑i∈Vpdata(x<t)ilog⁡q(x<t)i]\begin{align\*} \text{HardTargetLoss} &= \mathbb{E}\_{x\_{<t} \sim D} \left[ - \sum\_{i \in V} p\_\text{data}(x\_{<t})\_i \log q(x\_{<t})\_i \right] \end{align\*}HardTargetLoss​=Ex<t​∼D​[−i∈V∑​pdata​(x<t​)i​logq(x<t​)i​]​ where x<tx\_{<t}x<t​ is a prefix
   sampled from a dataset DDD, pdata(x<t)p\_\text{data}(x\_{<t})pdata​(x<t​) is a one-hot vector
   corresponding to the token that follows x<tx\_{<t}x<t​, q(x<t)q(x\_{<t})q(x<t​) is a dense vector corresponding
   to the draft model’s probability distribution of the token that follows
   x<tx\_{<t}x<t​, and VVV is the vocabulary size. To train a draft
   model to produce outputs that are more similar to the outputs of the
   target model, we minimize MixedLoss(ω)=ω⋅Ex<t∼D[−∑i∈Vptarget(x<t)ilog⁡q(x<t)i]+(1−ω)⋅(−α)\begin{align\*} \text{MixedLoss}(\omega) = \omega \cdot \mathbb{E}\_{x\_{<t} \sim D} \left[ - \sum\_{i \in V} p\_\text{target}(x\_{<t})\_i \log q(x\_{<t})\_i \right] + (1 - \omega) \cdot (-\alpha) \end{align\*}MixedLoss(ω)=ω⋅Ex<t​∼D​[−i∈V∑​ptarget​(x<t​)i​logq(x<t​)i​]+(1−ω)⋅(−α)​ where ptarget(x<t)p\_\text{target}(x\_{<t})ptarget​(x<t​) is a dense vector
   corresponding to the target model’s probability distribution of the
   token that follows x<tx\_{<t}x<t​ and α\alphaα is the expected acceptance probability
   ([Leviathan et al., 2023](https://arxiv.org/abs/2211.17192)).
   Minimizing the first term minimizes the distillation loss ([Hinton et al., 2015](https://arxiv.org/abs/1503.02531)), and
   minimizing the second term maximizes the average generation length τ\tauτ. As shown in [Figure
   5](#ablations), ω=0.5\omega=0.5ω=0.5 marginally
   outperformed ω=1\omega=1ω=1.

## Evaluation

We fix an 8-layer, 67-million body-parameter multi-head attention
target model and vary the draft model, with all model architecture
details provided in the [Appendix](#appendix). Each model
with NNN total-parameters is trained on
20×N20 \times N20×N tokens ([Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556)) from
the [LongCrawl64](https://manifestai.com/articles/longcrawl64/)
dataset using sequences of length 1024. We compare drafting tokens for
the target model using the following draft models:

1. **Vanilla speculative decoding** ([Chen et al., 2023](https://arxiv.org/abs/2302.01318)). This
   draft model has 18\frac{1}{8}81​th as many
   body parameters as the target model, and is trained by minimizing the
   standard cross-entropy loss. This is the most widely understood
   implementation of speculative decoding, and, as can be seen in Figures
   [1](#fig:1) and [4](#fig:4), it is strong for
   short contexts and small batch sizes.
2. **MagicDec** ([Chen et al., 2024](https://arxiv.org/abs/2408.11049v3)). This
   draft model is identical to the target model, but it sparsely accesses
   its KV cache through a StreamingLLM attention mask with a window size of
   64 and a sink size of 1. This is an elegant implementation of
   speculative decoding, and is strong in the large-batch long-context
   regime.
3. **SPIRe** (ours). This draft model was described above
   in detail, and it has 14\frac{1}{4}41​th as
   many body parameters as the target model. During both training and
   inference, we use a StreamingLLM attention mask with a window size of 64
   and a sink size of 1.

We measure average generation lengths τ\tauτ by generating G=64G=64G=64 tokens given n=65536n=65536n=65536 contexts from the validation split of
the [LongCrawl64](https://manifestai.com/articles/longcrawl64/)
dataset, and we calculate iteration time multipliers Δt\Delta tΔt for different batch sizes and
context lengths using our performance model.

We use τ\tauτ’s corresponding to a
medium context length of L=512L=512L=512 to
calculate all throughput multipliers τ/Δt\tau / \Delta tτ/Δt in Figures [1](#fig:1), [4](#fig:4), [5](#fig:ablation), and [6](#fig:extrapolation), by assuming that τ\tauτ is the same when generating with shorter
or longer contexts. [Table 1](#table) shows that the average
generation length τ\tauτ is similar for
contexts of length L∈{256,512,960}L \in \{256, 512, 960\}L∈{256,512,960}, and our assumption is supported by e.g. [Figure 5](https://arxiv.org/pdf/2309.17453#page=7) of [Xiao et al., 2024](https://arxiv.org/abs/2309.17453), which
suggests that even with extremely long contexts, it suffices to attend
to the most recent tokens.

| Context Length | Output Length | Vanilla SD | MagicDec | SPIRe |
| --- | --- | --- | --- | --- |
| 960 | 64 | 2.644 ± 0.004 | 3.793 ± 0.004 | 3.382 ± 0.004 |
| 512 | 64 | 2.647 ± 0.004 | 3.891 ± 0.004 | 3.401 ± 0.004 |
| 256 | 64 | 2.637 ± 0.004 | 4.005 ± 0.004 | 3.427 ± 0.004 |

Table 1: Average generation lengths τ\tauτ, with 95% confidence intervals, for
different draft models and context lengths. Using a maximum speculation
depth of k=4k=4k=4, each round of speculation
generates between 1 and 5 tokens, with between 0 and 4 tokens accepted
before the first rejection.

We focus on a medium context of 512 tokens since we care about
accelerating decoding across a range of context lengths. If a context of
128K is considered “long” for Llama 3 70B ([Llama Team, 2024](https://arxiv.org/abs/2407.21783)), then a
context of 64K may be considered “medium”. As shown in [Figure 3](#fig:batch-size), 64K is four powers of two larger
than the longest context LcritL\_\text{crit}Lcrit​
for which verification with Llama 3 70B is compute-bound at
*some* batch size. A context length of 512 is four powers of two
larger than Lcrit=32L\_\text{crit} = 32Lcrit​=32 for our
target model, so we consider a context of 512 to be of medium
length.

![](../static/sd_batch_size.svg)

Figure 3: With a sufficiently long context, verification may never be
compute-bound even in the limit as batch size tends to infinity. For
context lengths where verification is compute bound for some batch size
BcritB\_\text{crit}Bcrit​, there’s no benefit in
terms of throughput (and some cost in terms of latency) to increasing
batch size above BcritB\_\text{crit}Bcrit​.

### Results

Our draft model SPIRe produces the largest increase in the modeled
throughput of generating tokens for most batch sizes and context
lengths, as shown in [Figure 4](#fig:4). SPIRe achieves a
lower average generation length τ\tauτ
than MagicDec, but drafting with SPIRe nevertheless produces a larger
speedup due to its lower cost of performing forward passes.

![](../static/sd_4.svg)

Figure 4: Speedup of generating tokens in a batch of size BBB given a context of length LLL. Using the highlighted values, we get that
SPIRe increases the modeled throughput of speculative decoding by 100%
compared to vanilla speculative decoding and by 35% compared to
MagicDec.

[Figure 1](#fig:1) is derived from [Figure
4](#fig:4) by focusing on a batch size of B=64B=64B=64. For medium contexts of around 512
tokens, [Figure 3](#fig:batch-size) shows that the
operational intensity of verification barely increases as we increase
the verification batch size beyond k×64=4×64=256k \times 64 = 4 \times 64 = 256k×64=4×64=256, but it does increase significantly as the
verification batch size is increased from 1 to 64.

### Ablations

See [Figure 5](#fig:ablation) for ablations of the
techniques used to train our draft model SPIRe. To ablate the sparse KV
cache, we conservatively assume the same average generation length τ\tauτ as SPIRe, and use our performance model
to calculate new throughput multipliers. We ablate the other techniques
by training new draft models and evaluating them.

![](../static/sd_ablation.svg)

Figure 5: Ablations of the techniques used to train our draft model
SPIRe. In order of importance, the techniques are 1) sparse KV cache, 2)
attending to target model activations, 3) distillation loss, 4)
initializing by pruning the target model, 5) feedback memory, and 6)
MixedLoss\text{MixedLoss}MixedLoss.

The throughput multiplier for SPIRe (without sparse KV cache) is
constant with respect to the context length LLL in [Figure 5](#fig:ablation)
since here the memory cost of generating a token is higher than the
compute cost, and since k=4k=4k=4 and Ntarget=4×NdraftN\_\text{target} = 4 \times N\_\text{draft}Ntarget​=4×Ndraft​,
LLL cancels out in our expression for the
iteration time multiplier Δt\Delta tΔt.

### Extrapolation

What if we extrapolate the results above to drafting for Llama 3 70B
while maintaining the ratio Ndraft/NtargetN\_\text{draft} / N\_\text{target}Ndraft​/Ntarget​, by assuming the same values of the average
generation length τ\tauτ but
recalculating iteration time multipliers using our performance model?
Our draft model SPIRe produces the largest increase in the modeled
throughput of generating tokens for most context lengths supported by
LLM providers, as shown in [Figure
6](#fig:extrapolation).

![](../static/sd_extrapolation.svg)

Figure 6: Estimated speedup of generating tokens from Llama 3 70B in
a batch of size BBB given a context of
length LLL. This figure is very similar,
but not identical, to [Figure 1](#fig:1).

## Cost Analysis

Each draft model costs a different amount in training FLOPs. We
ignore FLOPs due to embedding and unembedding parameters, which become
negligible as the draft model is scaled up. The cost to train our draft
model SPIRe is around 14\frac{1}{4}41​th the
cost to train the target model, since it has 14\frac{1}{4}41​th as many body parameters but
achieves lower MFU during training.

Suppose that training the target model costs 1 unit of our compute
budget, and that we expect to spend 10 units generating tokens for
customers using autoregressive decoding. Using our draft model SPIRe for
speculative decoding with (B,L)=(64,512)(B, L) = (64, 512)(B,L)=(64,512) amounts to investing 0.250.250.25
to train the draft model but saving 10×(1−1/2.78)=6.4010 \times (1 - 1/2.78) = 6.4010×(1−1/2.78)=6.40 units in decoding cost, and drafting with
MagicDec amounts to saving 10×(1−1/2.05)=5.1210 \times (1 - 1/2.05) = 5.1210×(1−1/2.05)=5.12 units in decoding cost (the throughput multipliers
2.78 and 2.05 are highlighted in [Figure 4](#fig:4)).
Overall, SPIRe saves 6.156.156.15 units, or
over 20% more than what MagicDec saves.

If we again extrapolate our results to drafting for Llama 3 70B with
(B,L)=(64,65536)(B, L) = (64, 65536)(B,L)=(64,65536), we find that
drafting with SPIRe saves 10×(1−1/2.82)=6.4510 \times (1 - 1/2.82) = 6.4510×(1−1/2.82)=6.45 units in decoding cost and drafting with MagicDec
saves 10×(1−1/2.14)=5.3310 \times (1 - 1/2.14) = 5.3310×(1−1/2.14)=5.33
units (the throughput multipliers 2.82 and 2.14 are highlighted in the
[Appendix](#appendix)). Overall, SPIRe saves 6.206.206.20 units, or over 16% more than what
MagicDec saves. If there is significant downward variation in batch
sizes and context lengths, then SPIRe outperforms MagicDec by a much
greater amount in terms of decoding cost saved.

## Related Work

- The only other work we’re aware of which attempts to use speculative
  decoding to increase the throughput of decoding, rather than to minimize
  latency, is MagicDec ([Chen
  et al., 2024](https://arxiv.org/abs/2408.11049v3)).
- Continuous batching reduces the need for speculative decoding, but
  both techniques are complementary. In vLLM, [speculative
  decoding is integrated with the system’s continuous batching
  architecture](https://blog.vllm.ai/2024/10/17/spec-decode.html), and studying their interaction further is an
  interesting direction for future work.
- Our technique is orthogonal to any technique used to accelerate
  prefill. In practice, accelerating both prefill and decoding is
  important for reducing the cost per token generated.
- We used the token verification algorithm in our experiments, but our
  results should hold when using the block verification ([Sun et al., 2024](https://arxiv.org/abs/2403.10444)) algorithm
  too.

## Conclusion

Minimizing cost per token requires maximizing throughput. We
demonstrated that draft model architecture significantly impacts
throughput, with optimal choices depending on the batch sizes and
context lengths expected in production. We proposed an
implementation-agnostic performance model for evaluating the throughput
of speculative decoding with different draft models, and we proposed a
draft model SPIRe which outperforms strong baselines when decoding with
large batch sizes and medium-to-long contexts. Future work can
empirically validate our performance model, use a more principled
long-context evaluation, and analyze the sensitivity of speedup with
respect to the maximum speculation depth kkk.

## Contributions and Acknowledgements

Sanjit Neelam proposed most aspects of the architecture of our draft
model SPIRe. Feedback Transformer was suggested by Reiner Pope, and
attending to target activations was co-developed by Sanjit Neelam and
Reiner Pope. Explicitly maximizing the expected acceptance probability
α\alphaα was proposed by Akshay Mishra.
The evaluation methodology was co-designed by Sanjit Neelam and Reiner
Pope. All experiments were conducted by Sanjit Neelam, who also wrote
all text and produced all figures. Daniel Heinlein and Vaclav Cvicek
provided valuable feedback that improved the presentation of
results.

We use [seqax](https://github.com/MatX-inc/seqax), our
research-focused LLM codebase built on [JAX](https://github.com/jax-ml/jax), to perform all
experiments. This work was supported by Cloud TPUs from Google’s [TPU Research
Cloud](https://sites.research.google/trc/about/).

## Appendix

See <https://github.com/MatX-inc/seqax/blob/SPIRe/spire_appendix.ipynb>.

## Footnotes

---

1. The cost is that speculative decoding always requires
   performing more FLOPs than autoregressive decoding; the draft model
   performs forward passes, and the target model performs forward passes on
   not only the draft tokens that are accepted but also those that are
   rejected. Thus, speculative decoding only makes sense if autoregressive
   decoding is sufficiently memory-bound.[⤴](#fnref1)
2. Optimized inference stacks with e.g. pipeline
   parallelism approach this ideal. Throughput multipliers are higher than
   if we assumed that computation and memory accesses could not be
   overlapped, in which case we would have Decode Cost=Compute Cost+Memory Cost\text{Decode Cost} = \text{Compute Cost} + \text{Memory Cost}Decode Cost=Compute Cost+Memory Cost.[⤴](#fnref2)
3. The inclusion of HOI in the throughput multiplier makes
   it hardware-dependent, but this only makes a difference in the
   compute-bound (bottom-left) region of [Figure 4](#fig:4).
   When decoding and verification are both memory-bound, HOI cancels out.[⤴](#fnref3)
4. [Xiao et al.,
   2024](https://arxiv.org/abs/2309.17453) use positions within the cache rather than those in the
   original text when adding positional information to tokens. We do the
   latter for SPIRe and the former for our implementation of MagicDec.[⤴](#fnref4)
