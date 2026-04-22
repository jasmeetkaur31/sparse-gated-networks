# Self-Pruning Neural Network — Report

> **Case Study: The Self-Pruning Neural Network**
> Tredence Analytics — AI Engineering Internship 2025

---

## 1. Why Does L1 Penalty on Sigmoid Gates Encourage Sparsity?

The core insight is in the **gradient behaviour** of different penalties:

| Penalty | Formula | Gradient w.r.t. gate `g` (for g > 0) |
|---------|---------|----------------------------------------|
| L2 | Σ g²    | 2g → **weakens as g → 0** |
| L1 | Σ \|g\| | **+1 (constant)** regardless of magnitude |

Since our gates pass through sigmoid (so g > 0 always), the L1 gradient
is a **constant −λ** pointing toward zero. This "constant pull" will
drive a gate all the way to zero if the classification gradient cannot
overpower it.

L2, by contrast, produces a gradient that **shrinks with g** — it makes
weights small but rarely zero. This is the mathematical reason why:
- **L2 regularisation → small weights**
- **L1 regularisation → zero weights (true sparsity)**

### Visualisation of the gradient difference

```
L2 gradient: g = 0.5 → 1.0,  g = 0.1 → 0.2,  g = 0.01 → 0.02  (fades out)
L1 gradient: g = 0.5 → 1.0,  g = 0.1 → 1.0,  g = 0.01 → 1.0   (constant!)
```

---

## 2. Beyond Sigmoid: Hard Concrete Gates

This implementation adds an optional **Hard Concrete** gating mechanism
(Louizos et al., 2018 — *"Learning Sparse Neural Networks through L0 Regularization"*).

The idea: stretch the sigmoid over the interval (ζ, β) = (−0.1, 1.1),
then **hard-clip** to [0, 1]:

```
s      = sigmoid(gate_score + noise)       ← noise only during training
s_bar  = s × (1.1 − (−0.1)) + (−0.1)     ← stretch to (−0.1, 1.1)
gate   = clamp(s_bar, 0, 1)               ← clip to [0, 1]
```

This places probability mass **exactly at 0 and exactly at 1**, meaning
gates can hit exact zero during the forward pass — something sigmoid never achieves.

| Property | Sigmoid Gate | Hard Concrete Gate |
|----------|-------------|-------------------|
| Range | (0, 1) — open | [0, 1] — closed |
| Exact zeros | ✗ (asymptotic) | ✓ (during eval) |
| Gradient through 0 | Vanishes | Well-defined |
| Stochastic training | ✗ | ✓ (noise injection) |

---

## 3. Lambda Warm-up Strategy

A key practical insight: **if λ is large from epoch 1**, the sparsity
loss overpowers the classification loss before the network has learned anything useful,
causing accuracy collapse.

Solution: **linearly ramp λ from 0 → target over the first 30% of training**.

```
epoch 1  : effective λ = 0
epoch 10 : effective λ = target × 10/30
epoch 30+: effective λ = target (full)
```

This lets the network first **learn a useful representation**, then **prune
the redundant connections** — the natural order of learning.

---

## 4. Results Table

> *(These are representative expected results; actual numbers depend on
> hardware, random seed, and number of epochs.)*

| Lambda (λ) | Test Accuracy | Sparsity Level (%) | Notes |
|------------|---------------|---------------------|-------|
| 1e-4       | ~82–85%       | ~10–25%             | Low pruning pressure, high accuracy |
| 1e-3       | ~78–82%       | ~40–60%             | Good accuracy/sparsity trade-off ✓ |
| 1e-2       | ~70–76%       | ~75–90%             | Aggressive pruning, accuracy drops |

**Interpretation:**
- **Low λ (1e-4):** The sparsity penalty is a gentle nudge. Most gates remain
  open. The network performs well but isn't substantially compressed.
- **Medium λ (1e-3):** The Pareto-optimal point. A large fraction of weights
  are pruned with modest accuracy cost. Best for deployment.
- **High λ (1e-2):** Sparsity loss dominates. The network learns to classify
  with very few active connections, but accuracy suffers. The warm-up schedule
  prevents complete collapse.

---

## 5. Gate Distribution Plot

See `outputs/gate_distributions.png`.

A successful pruning result shows a **bimodal gate distribution**:

```
Count
  │
  ██                              ██
  ██                             ███
  ██                            ████
  ████                         █████
  █████████             ████████████
──┼──────────────────────────────────── Gate value
  0                                   1
  ▲ pruned connections        retained ▲
```

The large spike at 0 means most connections have been eliminated.
The cluster near 1 represents the surviving "important" connections.
Very little probability mass in the middle (0.1–0.9) — the network
has made crisp binary keep/prune decisions.

---

## 6. Architecture

```
Input (3×32×32)
    │
    ▼
[Conv Block 1]  64 filters, BatchNorm, ReLU, MaxPool, Dropout
    │
    ▼
[Conv Block 2]  128 filters, BatchNorm, ReLU, MaxPool, Dropout
    │
    ▼
[Conv Block 3]  256 filters, BatchNorm, ReLU, MaxPool
    │
    ▼ Flatten → 4096-dim
    │
    ▼
[PrunableLinear 4096→512]  ← gates here
    │ ReLU + Dropout
    ▼
[PrunableLinear 512→256]   ← gates here
    │ ReLU + Dropout
    ▼
[PrunableLinear 256→10]    ← gates here
    │
    ▼
 Logits (10 classes)
```

**Why prune only the head?**
Conv layers reuse each kernel across spatial positions (weight sharing),
so they have far fewer unique parameters. The dense head is where
parameter redundancy accumulates — it is the natural target for
connection pruning.

---

## 7. How to Run

```bash
# Install dependencies
pip install torch torchvision matplotlib numpy

# Default run (3 lambda values, sigmoid gates, 30 epochs)
python train.py

# Custom run
python train.py \
    --lambdas 1e-4 1e-3 1e-2 \
    --gate_mode hard_concrete \
    --epochs 50 \
    --lr 1e-3 \
    --batch_size 128

# Outputs
#   outputs/gate_distributions.png  — gate histogram
#   outputs/training_curves.png     — accuracy + sparsity vs epochs
#   outputs/report.md               — this report (auto-generated)
#   outputs/results.json            — full metrics
#   checkpoints/best_lambda_*.pt    — best model per λ
```

---

## 8. References

1. Louizos et al. (2018). *Learning Sparse Neural Networks through L0 Regularization.* ICLR 2018.
2. Han et al. (2015). *Learning both Weights and Connections for Efficient Neural Networks.* NeurIPS 2015.
3. Frankle & Carlin (2019). *The Lottery Ticket Hypothesis.* ICLR 2019.

---

*Report generated by `train.py` — Tredence Analytics AI Engineering Internship Case Study*
