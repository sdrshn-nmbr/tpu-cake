# Future leakage in block-quantized attention

January 09, 2026

Akshay Mishra, Reiner Pope, Sanjit Neelam, Daniel Heinlein, Vaclav
Cvicek, Zaal Vasania, and James Hill-Khurana

Quantizing attention improves efficiency on two fronts: the model has
higher compute throughput, and loads fewer bytes per key/value. However,
training with block quantized attention can break causal modeling. We
present a fix that enables training with MXFP4 in both attention and the
attention gradient.

## Causal modeling

In causal language modeling, the final logits at position iii must depend only on tokens at positions
≤i\le i≤i. **Future leakage**
is when information from positions >i>i>i may influence the logits at position
iii. It poses an issue because it causes
a skew between training and decode. In typical setups, causal masks
prevent leakage in attention. But block quantization can introduce a
subtle new path for future leakage.

## Block quantization

Modern accelerators require using block-quantized matrix
multiplications for highest throughput. To use these instructions to
compute A×BA \times BA×B, the row vectors of
AAA and column vectors of BBB must be split into blocks of size kkk and quantized. The specific value of kkk depends on the format being used. For
example, the microscaling formats use k=32k=32k=32.

![](../static/row_block_split.svg)

Within a block, let x0,x1,…,xk−1x\_0, x\_1, \ldots, x\_{k-1}x0​,x1​,…,xk−1​ denote the precise (unquantized) elements. When
quantizing, we approximate each element xix\_ixi​ with a quantized value qiq\_iqi​ and a scale sss shared across the block: xi≈qi⋅sx\_i \approx q\_i \cdot sxi​≈qi​⋅s

There are various approaches to selecting sss with different tradeoffs. But in general
they choose an sss with a function of all
elements in the block.

Given sss, the quantized elements
are:

qi=round(xis)q\_i = \text{round}\left(\frac{x\_i}{s}\right)qi​=round(sxi​​)

With this procedure, sss and
consequently qiq\_iqi​ depend on
**all** the pre-quantized elements in the block. So if the
quantization block spans across different token positions, quantization
enables the later tokens in the block to influence earlier tokens.

## Quantizing attention

Causal attention for a single head takes queries Q\mathbf{Q}Q, keys K\mathbf{K}K, and values V\mathbf{V}V as inputs, and produces:

P=softmax(Q×KT+M)output=P×V\begin{array}{lcl} \mathbf{P} & = & \text{softmax}(\mathbf{Q} \times \mathbf{K}^T + \mathbf{M}) \\ \text{output} & = & \mathbf{P} \times \mathbf{V} \end{array}Poutput​==​softmax(Q×KT+M)P×V​

where M\mathbf{M}M is the causal
mask.

Our goal is to use block-quantized matrix multiplications for Q×KT\mathbf{Q} \times \mathbf{K}^TQ×KT and P×V\mathbf{P} \times \mathbf{V}P×V. So Q\mathbf{Q}Q and K\mathbf{K}K need to be quantized in blocks
formed along the head dimension, while P\mathbf{P}P and V\mathbf{V}V need to be quantized in blocks
formed along different token positions.

Quantizing P\mathbf{P}P is safe
despite blocking along token positions, since the causal mask zeros out
future probabilities. However, the quantized V\mathbf{V}V at position jjj can depend on values at positions >j> j>j, which can cause future leakage.

## When does quantized V\mathbf{V}V cause future leakage?

Consider query position iii and value
position jjj, with block indices:

bi=⌊ik⌋,bj=⌊jk⌋b\_i = \left\lfloor \frac{i}{k} \right\rfloor, \quad b\_j = \left\lfloor \frac{j}{k} \right\rfloorbi​=⌊ki​⌋,bj​=⌊kj​⌋

Leakage occurs when query position iii
and value position jjj share a
quantization block position (bi=bjb\_i = b\_jbi​=bj​):

![](../static/leaky_region_diagram.svg)

- **bi=bjb\_i = b\_jbi​=bj​
  (block-diagonal):** query iii
  attends to value jjj if j≤ij \le ij≤i. But the quantized value at jjj is computed from *all* positions in
  the block (including positions greater than iii). This breaks causality since the attention
  output at position iii can depend on
  positions greater than iii.
- **bi>bjb\_i > b\_jbi​>bj​ (past
  blocks):** All positions in block bjb\_jbj​ precede the first position in block bib\_ibi​. No leakage.
- **bi<bjb\_i < b\_jbi​<bj​ (future
  blocks):** The causal mask zeros P[i,j]\mathbf{P}[i,j]P[i,j] for j>ij > ij>i. No leakage.

## Solution and validation

Leakage only occurs when the **iii-th block-diagonal tile** of P\mathbf{P}P is multiplied with the iii-th quantized block of V\mathbf{V}V. So we can prevent future leakage
by using unquantized P\mathbf{P}P and
V\mathbf{V}V when multiplying the
**iii-th tile of P\mathbf{P}P** with the iii-th block of V\mathbf{V}V, while using block quantized
matrix multiplications everywhere else.

To validate this fix, we trained a pair of 1B-parameter models on the
C4 dataset, with MXFP4 for attention and the attention gradient. Both
models share the following configuration:

| **Parameter** | **Value** |
| --- | --- |
| Layers | 8 |
| d\_model | 2048 |
| d\_head | 128 |
| d\_ff | 16384 |
| Attention heads | 16 |
| Context length | 1024 |
| Scale selection | maxabs calibration |
| Quantized ops | matrix multiplies in attention + matrix multiplies in attention gradient |
| Forwards pass rounding | Round-to-Nearest |
| Backwards pass rounding | Stochastic Rounding |

We trained a “Leaky” model that used MXFP4 for all of P×V\mathbf{P} \times \mathbf{V}P×V, and a “Fixed”
model that used the proposed solution.

The Fixed model remained well behaved throughout training. However,
the Leaky model had training dynamics associated with future leakage:
the grad norms grew rapidly before the loss started to improve
suspiciously fast.

To confirm that the Leaky model was only doing better because of
future leakage, we evaluated loss in two modes on a heldout set. In
parallel mode, we ran prefill once per sequence. In autoregressive mode,
we averaged the loss for generating the sequence one token at a
time.

| Model | Parallel | Autoregressive | Gap |
| --- | --- | --- | --- |
| Leaky | 2.56 | 2.66 | +0.10 |
| Fixed | 2.64 | 2.64 | 0.00 |

Despite the Leaky model’s parallel loss being lower than the Fixed
model’s, the autoregressive loss was worse. The Leaky model’s gap in
loss between parallel and autoregressive modes indicates that it was
reliant on future signal. The Fixed model had no gap, demonstrating our
solution worked.

The Leaky model’s parallel mode was using quantization error to
encode information about upcoming tokens. Padding a block with zeros
(like we do in autoregressive mode) won’t change the selected scale. But
adding outliers to the end (which can happen in parallel mode) makes the
scale larger. Larger scales can cause earlier values to underflow.

![](../static/quantization_comparison.svg)

Example where a future outlier causes the second value to
underflow.

The model can use whether earlier values in a block underflowed to
infer something about upcoming tokens.

## Final thoughts

We hope the methodology here provides a starting point for
researchers interested in quantizing attention during training. In
future work, we aim to demonstrate that quantized attention can match
the quality of float baselines while still providing end-to-end
speedups.

If these research problems sound interesting, consider [working with us](https://matx.com/jobs)!
