## Introduction

Attention sampling speeds up the processing of large inputs by only
processing a fraction of the input in high resolution. To achieve that we make
use of an _attention_ distribution predicted in low resolution that provides
the information regarding the usefulness or importance of each part of the
image.

The following paragraphs will introduce the theory and the model as well as how
each part is implemented in this python library. The full details can be found
in our paper "[Processing Megapixel Images with Deep Attention-Sampling
Models][paper]" as well as our [poster][poster] and [presentation][presentation].
<!-- Math macros -->
\(
    \newcommand{\R}{\mathbb{R}}
    \DeclareMathOperator*{\Exp}{\mathbb{E}}
    \DeclareMathOperator*{\Var}{\mathbb{V}}
    \newcommand{\E}[2][]{\Exp_{#1}\!\left[ #2 \right]}
    \newcommand{\V}[2][]{\Var_{#1}\!\left[ #2 \right]}
    \newcommand{\diff}[2]{\frac{\partial #1}{\partial #2}}
    \newcommand{\norm}[1]{\left\|#1\right\|}
\)

## Attention

Firstly, we need to define what attention means in the context of neural
networks. Given an input \(x\) and \(K\) features \(f(x) \in \R^{K \times D}\),
we aggregate those features using an attention distribution \(a(x) \in \R_+^K\)
such that \(\sum_i a(x)_i = 1\) in the following way:

\begin{equation}
    \Phi(x) = \sum_{i=1}^K a(x)_i f(x)_i = \E[I \sim a(x)]{f(x)_I}.
    \label{eq:attention}
\end{equation}

The above definition of attention does not make use of "queries" and "keys",
that are common in NLP related papers, however, the query can be included in
\(a(x)\) as an extra input and then the definition becomes equally general. For
our purposes we assume the query to be always one, namely _what are the most
useful feature positions in our input_.

The functions defined above are implemented as neural networks and we refer to
them as **attention network** and **feature network**.

## Attention Sampling

Under the assumption that \(f(x)_i\) is difficult to compute, we aim to save
computational resources by avoiding to compute the features for all the
positions. We can instead approximate \(\Phi(x)\) using Monte-Carlo sampling:

\begin{equation}
\begin{aligned}
    Q &= \{q_i \sim a(x) \mid i \in \{1, 2, \dots, N\}\} \\
    \Phi(x) &\approx \frac{1}{N} \sum_{q \in Q} f(x)_q
\end{aligned}
\end{equation}

In our paper we show that

* the approximation above is optimal (of minimum variance) if we normalize the
  features
* we can derive an unbiased gradient estimator that uses only the samples Q to
  train our models in an end to end fashion
* we can derive similar unbiased estimators for sampling without replacement
  that ensures that we compute \(f(x)_i\) only once

## Practical Implementation

In most cases, the attention weights are a function of the features; thus in
order to compute the attention distribution and sample feature positions we
need to compute all the features which makes our approximation unnecessary.

Instead, we propose computing the attention in low resolution directly from the
input and consequently much faster than computing all the features. The full
pipeline for images is depicted in Figure 1.

<div class="fig">
<img src="../img/architecture.jpg" alt="Attention sampling pipeline for images"
    style="width: 75%" />
<span>Given an image, a) we downscale it  b) we compute the attention
distribution c) we sample positions from the attention d) extract patches and
compute features from these positions e) average the features</span>
</div>

The above pipeline is implemented by the method
`ats.core.attention_sampling(...)` that accepts as parameters, the attention
network, the feature network, the size and number of the patches to be sampled.
See the [API documentation][api] for more details.

[paper]: https://arxiv.org/abs/1905.03711
[poster]: https://idiap.ch/~katharas/pdfs/ats-icml-poster.pdf
[presentation]: https://www.videoken.com/embed/UqSyCaz9wFQ?tocitem=31
[api]: keras-api.md
