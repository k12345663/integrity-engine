# Mathematical foundations (with proofs, verified on this system)

This document states the mathematical facts the engine relies on, proves each one, and points to a numerical check that runs on the **real system**: its embedding model, its sample corpus and its calibration file. Run it with

```bash
python scripts/verify_math.py        # 15 checks -> evaluation/math_verification.json
```

Current result: **15/15 checks pass** on 176 sentence embeddings (dimension $d = 384$) from the 12 readable sample submissions. The model-free parts are also covered by `tests/test_math.py`, which runs in CI.

**Notation.** Vectors are columns in $\mathbb{R}^d$. $a \cdot b = \sum_i a_i b_i$ is the dot product, $\lVert a \rVert = \sqrt{a \cdot a}$ is the length, and $\hat a = a / \lVert a \rVert$ is the unit vector in the direction of $a$. $E \in \mathbb{R}^{n \times d}$ stacks $n$ unit embeddings as rows.

---

## 1. Cosine similarity is a dot product of unit vectors

**Definition.** $\cos(a,b) = \dfrac{a \cdot b}{\lVert a \rVert\,\lVert b \rVert}$, the cosine of the angle $\theta$ between $a$ and $b$.

**Theorem 1 (normalisation).** For $a \neq 0$, $\lVert \hat a \rVert = 1$, and $\cos(a,b) = \hat a \cdot \hat b$.

*Proof.* $\lVert \hat a \rVert = \lVert a \rVert / \lVert a \rVert = 1$. Then
$\hat a \cdot \hat b = \frac{a}{\lVert a\rVert}\cdot\frac{b}{\lVert b\rVert} = \frac{a\cdot b}{\lVert a\rVert\lVert b\rVert} = \cos(a,b)$. $\square$

**Why it matters.** `Embedder.encode(..., normalize_embeddings=True)` stores $\hat a$. A FAISS *inner-product* index (`IndexFlatIP`) can then return cosine similarities directly, with no division at query time. One matrix product computes every pairwise cosine at once: $G = E E^\top$.

**Verified (T1, T2).** $\max_i |\lVert e_i\rVert - 1| = 1.2\times10^{-7}$. The cosine computed from raw (unnormalised) model outputs equals the dot product of the stored unit vectors to within $2.5\times10^{-7}$ (float32 rounding).

## 2. Scores are bounded: the Cauchy–Schwarz inequality

**Theorem 2.** $|a \cdot b| \le \lVert a\rVert\,\lVert b\rVert$, hence $-1 \le \cos(a,b) \le 1$. Equality $\cos = 1$ holds iff $b = \lambda a$ with $\lambda > 0$.

*Proof.* For $b \ne 0$ and any real $t$, $0 \le \lVert a - t b\rVert^2 = \lVert a\rVert^2 - 2t\,(a\cdot b) + t^2\lVert b\rVert^2$. This quadratic in $t$ is never negative, so its discriminant is $\le 0$: $4(a\cdot b)^2 - 4\lVert a\rVert^2\lVert b\rVert^2 \le 0$. Equality needs $\lVert a - tb \rVert = 0$ for some $t$, i.e. $a = tb$. $\square$

**Consequence: why self-matches must be excluded.** $\cos(a,a) = 1$ is the largest possible value, so every chunk's nearest neighbour in the index is itself (or an identical copy). `find_passages` therefore discards every neighbour from the query's own document, and requests `top_k + |own chunks|` neighbours so that `top_k` real candidates remain.

**Verified (T3, T9).**
- Across all $176^2$ pairs, $\min G = -0.153$ and $\max G = 1.000$.
- FAISS scores match the brute-force $E E^\top$ to within $8.5\times10^{-8}$.
- The top FAISS score is always $1$: in 22 of 50 queries the tie is with an identical template sentence in another file.

## 3. The Gram matrix is symmetric positive semi-definite

**Theorem 3.** $G = E E^\top$ is symmetric with $G_{ii} = 1$, positive semi-definite, and $\operatorname{rank} G \le d$.

*Proof.*
- **Symmetry:** $G^\top = (EE^\top)^\top = EE^\top$.
- **Diagonal:** $G_{ii} = e_i\cdot e_i = 1$.
- **Positive semi-definite:** for any $v \in \mathbb{R}^n$, $v^\top G v = v^\top E E^\top v = \lVert E^\top v\rVert^2 \ge 0$, so every eigenvalue is $\ge 0$.
- **Rank:** $\operatorname{rank}(EE^\top) = \operatorname{rank}(E) \le \min(n, d)$. $\square$

**Verified (T3b, T4).** The smallest eigenvalue is $-1.1\times10^{-15}$, i.e. zero up to rounding, and the rank is $135$. The rank is below $n = 176$ because the 150-word case prompt and the copied paragraphs repeat *identical* sentences across files. Identical rows add no new dimension, so the rank deficit is itself a fingerprint of duplicated text.

## 4. Cosine ranking equals nearest-neighbour ranking

**Theorem 4.** For unit vectors, $\lVert \hat a - \hat b\rVert^2 = 2 - 2\cos(a,b)$.

*Proof.* $\lVert \hat a - \hat b\rVert^2 = \lVert\hat a\rVert^2 - 2\,\hat a\cdot\hat b + \lVert\hat b\rVert^2 = 1 - 2\cos(a,b) + 1$. $\square$

**Corollary.** Since $x \mapsto 2 - 2x$ is strictly decreasing, sorting by *highest cosine* gives exactly the same order as sorting by *smallest Euclidean distance*. Every Euclidean nearest-neighbour index (HNSW, IVF, a vector database) can therefore serve the same search. This is what makes the scaling plan in README §8 valid.

**Verified (T5, T5b).** The identity holds to $4.8\times10^{-7}$ on all 15,400 sentence pairs, and the top-50 neighbour lists under both orders are identical.

## 5. Similarity is "almost transitive": the angular triangle inequality

**Theorem 5.** Let $\theta(a,b) = \arccos(\hat a\cdot\hat b) \in [0,\pi]$. Then $\theta(a,c) \le \theta(a,b) + \theta(b,c)$.

*Proof.* $\theta$ is the great-circle (geodesic) distance between $\hat a$ and $\hat b$ on the unit sphere $S^{d-1}$. Geodesic distance is a metric: the shortest path from $\hat a$ to $\hat c$ is no longer than the path that goes through $\hat b$. $\square$

**Corollary (collusion chains).** If $\cos(a,b) \ge s$ and $\cos(b,c) \ge s$ with $s \ge 0$, then
$$\cos(a,c) \ge \cos(2\arccos s) = 2s^2 - 1.$$
*Proof.* Since $\theta(a,c) \le 2\arccos s \le \pi$ and $\cos$ is decreasing on $[0,\pi]$, $\cos\theta(a,c) \ge \cos(2\arccos s)$. The double-angle formula $\cos 2x = 2\cos^2 x - 1$ gives $2s^2 - 1$. $\square$

*Example.* If A copies B and B copies C, each at $s = 0.95$, then A and C still have cosine $\ge 2(0.95)^2 - 1 = 0.805$. A chain of copies cannot hide by passing through an intermediate document: the end points stay above the 0.80 direct-match threshold.

**Verified (T6).** On 20,000 random triples of real embeddings, the minimum slack $\theta_{ab} + \theta_{bc} - \theta_{ac}$ is $0$ and never negative.

## 6. Orthogonality, anisotropy and orthogonal projection

**Definition.** $a \perp b$ (orthogonal) iff $a\cdot b = 0$, i.e. $\cos = 0$: "no shared direction".

**Theorem 6 (random directions are nearly orthogonal).** If $x, y$ are independent and uniformly distributed on $S^{d-1}$, then $\mathbb{E}[x\cdot y] = 0$ and $\operatorname{Var}[x\cdot y] = 1/d$.

*Proof.* Fix $y$. By rotational symmetry the distribution of $x\cdot y$ is that of a single coordinate $x_1$. Symmetry $x_1 \mapsto -x_1$ gives $\mathbb{E}[x_1] = 0$. Since $\sum_{i=1}^d x_i^2 = 1$ and all coordinates are exchangeable, $\mathbb{E}[x_1^2] = 1/d$. $\square$

So in $d = 384$ dimensions, unrelated *random* vectors have cosine $0 \pm 1/\sqrt{384} = 0 \pm 0.051$.

**Verified (T8).** On 2,000 random unit vectors in $\mathbb{R}^{384}$: mean $-0.0000$, standard deviation $0.0510$ (theory $0.0510$).

**Observation (anisotropy).** Real sentence embeddings are *not* spread uniformly. In this corpus, sentences from *different* submissions have mean cosine $\mathbf{0.181}$, far from $0$, and the mean embedding has length $\lVert\mu\rVert = 0.433$. All vectors lean towards a common direction $u = \mu/\lVert\mu\rVert$. This is the mathematical reason two unrelated answers to the same case can reach cosine $\approx 0.85$, and why a pure cosine threshold could not separate paraphrase from same-topic text (EVALUATION.md §6).

**Theorem 7 (orthogonal projector).** For a unit vector $u$, $P = I - uu^\top$ satisfies $P^2 = P$, $P^\top = P$ and $Pu = 0$. So $P$ is the orthogonal projection onto the hyperplane $u^\perp$, and $Px \perp u$ for every $x$.

*Proof.* $P^\top = I - (uu^\top)^\top = P$. Then $P^2 = I - 2uu^\top + u(u^\top u)u^\top = I - uu^\top = P$, because $u^\top u = 1$. Next, $Pu = u - u(u^\top u) = 0$. Finally $u\cdot Px = u^\top x - (u^\top u)(u^\top x) = 0$. $\square$

**Verified (T7, T7b, T7c).** $P$ built from the corpus mean passes all three identities. Projecting out $u$ and re-normalising moves the mean cross-document cosine from $0.181$ to $-0.008$, i.e. to orthogonal on average.

**Design note.** This "mean-centring" is a known way to make cosine thresholds more meaningful. The engine does not apply it yet, because $\mu$ depends on the corpus and would change between batches. It is listed as future work. The current system handles the same problem with the cross-encoder and the anchor rule.

## 7. The percentage scores

Let $S$ be a document's scoring sentences (template and reference lists removed), let $w(s)$ be the word count of sentence $s$, and let $M_j \subseteq S$ be the sentences matched by source $j$.

$$\text{coverage}_j = 100\cdot\frac{\sum_{s\in M_j} w(s)}{\sum_{s\in S} w(s)},\qquad \text{peer\%} = 100\cdot\frac{\sum_{s\in \bigcup_j M_j} w(s)}{\sum_{s\in S} w(s)}.$$

**Lemma 8.**
1. $0 \le \text{peer\%} \le 100$.
2. $\max_j \text{coverage}_j \le \text{peer\%} \le \min\big(100, \sum_j \text{coverage}_j\big)$.
3. No sentence is counted twice.

*Proof.* Each $M_j \subseteq \bigcup_j M_j \subseteq S$, and word counts are non-negative, which gives (1) and the left inequality of (2). The union bound $w(\bigcup M_j) \le \sum_j w(M_j)$ gives the right inequality. (3) holds because the numerator sums over a *set* of sentence ids (`coverage()` builds `covered = {sid ...}`). $\square$

**Lemma 9 (direction).** Coverage is not symmetric: $\text{coverage}(A{\to}B) \ne \text{coverage}(B{\to}A)$ in general, because the denominators differ.

*Example from the corpus.* The 28 s video is $81.2\%$ covered by `sub_01`, while `sub_01` is only $17.7\%$ covered by the video.

**Jaccard** $J(X,Y) = |X\cap Y| / |X\cup Y| \in [0,1]$, with $J = 1$ iff $X = Y$ (because $X\cap Y \subseteq X\cup Y$). It only *labels* matches ("verbatim" when $J \ge 0.7$); it never decides them.

## 8. Template (boilerplate) threshold

With $N$ readable documents, a sentence is template iff its document frequency satisfies $\mathrm{DF} \ge t(N) = \max\big(3, \lceil 0.25N\rceil\big)$.

**Proposition 10.**
1. Copying between two students is never mistaken for template.
2. A text shared by $k$ students is treated as template iff $k \ge t(N)$.

*Proof.* (1) A pair has $\mathrm{DF} = 2 < 3 \le t(N)$. (2) is the definition. $\square$

**Consequence (stated honestly).** A collusion ring of $k \ge t(N)$ students sharing one paragraph would be hidden as template. For $N = 12$, $t = 3$; for $N = 10{,}000$, $t = 2{,}500$, which makes this practically impossible at scale. EVALUATION.md §7.6 discusses this.

## 9. The AI-likelihood model

$$p = \sigma(z) = \frac{1}{1+e^{-z}},\qquad z = b + \sum_{i=1}^{4} w_i\,\tilde z_i,\qquad \tilde z_i = \operatorname{clip}\!\Big(\frac{x_i - m_i}{s_i}, -3, 3\Big).$$

**Lemma 11.**
1. $\sigma$ maps $\mathbb{R}$ onto $(0,1)$.
2. $\sigma$ is strictly increasing, with $\sigma'(z) = \sigma(z)\,(1-\sigma(z)) > 0$.
3. $\sigma(-z) = 1 - \sigma(z)$.

*Proof.* $e^{-z} > 0$ gives $0 < \sigma < 1$. Differentiating gives $\sigma' = e^{-z}/(1+e^{-z})^2 = \sigma(1-\sigma)$. For (3), $\frac{1}{1+e^{z}} = \frac{e^{-z}}{e^{-z}+1} = 1 - \sigma(z)$. $\square$

**Proposition 12 (clipping bounds the score).** Because $|\tilde z_i| \le 3$,
$$b - 3\sum_i |w_i| \;\le\; z \;\le\; b + 3\sum_i|w_i|.$$
With the shipped calibration (`config/ai_calibration.json`) this gives $z \in [-13.53, 14.59]$. So the AI score lies in $[1.3\times10^{-4}\%,\ 100\% - 4.6\times10^{-5}\%]$: it can never be exactly 0 or 100, and no single extreme feature can push it past these limits.

**Verified (T10).** Computed from the shipped calibration file.

**Each signal's contribution** $w_i\tilde z_i$ is additive in $z$ (log-odds). This is why the reports can list exactly which signal pushed a score up. The model is explainable by construction.

**Perplexity.** For tokens $t_1\ldots t_m$, $\mathrm{PPL} = \exp\big(\frac{1}{m}\sum_k -\log p(t_k \mid t_{<k})\big)$. Since $0 < p \le 1$, every term $-\log p \ge 0$, so $\mathrm{PPL} \ge 1$, with equality only if the model predicts every token with certainty. **Burstiness** is the standard deviation of the per-sentence averages of $-\log p$, so it is $\ge 0$.

## 10. Evaluation statistics

**Precision, recall, F1.** $P = \frac{TP}{TP+FP}$, $R = \frac{TP}{TP+FN}$, $F_1 = \frac{2PR}{P+R}$, the harmonic mean of $P$ and $R$.

**Lemma 13.** $\min(P,R) \le F_1 \le \frac{P+R}{2}$.

*Proof.* The upper bound is the harmonic–arithmetic mean inequality: $\frac{P+R}{2} - \frac{2PR}{P+R} = \frac{(P-R)^2}{2(P+R)} \ge 0$. For the lower bound, assume $P \le R$; then $\frac{2PR}{P+R} \ge \frac{2PR}{2R} = P$. $\square$

So F1 punishes imbalance: a detector that flags everything (recall 1, low precision) cannot score well.

**Proposition 14 (how much 3/3 really proves).** If the true detection rate is $p$, observing 3 successes in 3 independent trials has probability $p^3$. The one-sided 95% Clopper–Pearson lower bound solves $p^3 = 0.025$, i.e.
$$p \;\ge\; 0.025^{1/3} = 0.292.$$
So a perfect 3/3 on the sample corpus is consistent with a true recall as low as 29%. This is why the evaluation adds sentence-level metrics and the 9,725-pair public benchmark.

**Verified (T11).**

## 11. Cost of exact search (why the index choice matters at scale)

A flat (exact) search of one query against $N$ vectors costs $N\cdot d$ multiply-adds. All-vs-all over $N$ chunks costs $N^2 d$:

| Corpus | Chunks $N$ | All-vs-all $N^2 d$ |
|---|---|---|
| Sample (15 files) | 23 | $\approx 2.0\times10^{5}$ (instant) |
| One challenge (10,000 files) | ~130,000 | $\approx 6.5\times10^{12}$ (seconds on a GPU, minutes on CPU) |
| Multi-year archive | ~6,500,000 | $\approx 1.6\times10^{16}$ (needs HNSW / IVF) |

HNSW reduces the per-query cost to roughly $O(\log N)$ graph hops. That is safe here because stage 2 re-scores every candidate exactly (Theorem 4 guarantees the approximate Euclidean search targets the same neighbours).

---

### Summary of verified checks

| ID | Claim | Section |
|---|---|---|
| T1 | stored embeddings have unit length | 1 |
| T2 | dot product of unit vectors = cosine | 1 |
| T3, T3b | cosine in [−1, 1]; Gram matrix symmetric with unit diagonal | 2, 3 |
| T4 | Gram matrix positive semi-definite, rank ≤ d | 3 |
| T5, T5b | ‖a−b‖² = 2 − 2cos; cosine order = nearest-neighbour order | 4 |
| T6 | angular triangle inequality | 5 |
| T7, T7b, T7c | anisotropy; orthogonal projector identities; projection restores orthogonality | 6 |
| T8 | random 384-d unit vectors: mean 0, std 1/√d | 6 |
| T9 | FAISS = exact cosine; top hit is always self or duplicate | 2 |
| T10 | AI score bounded strictly inside (0, 100) | 9 |
| T11 | 3/3 recall lower bound 0.292 | 10 |
