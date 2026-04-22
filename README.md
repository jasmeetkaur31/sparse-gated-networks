# Self-Pruning Neural Network on CIFAR-10
**Tredence Analytics — AI Engineering Internship Case Study**

A neural network that **learns to prune itself during training** using learnable gate parameters and L1 sparsity regularisation.

---

## Key Ideas

### 1. PrunableLinear Layer
Each weight `w_ij` is multiplied by a learnable gate `g_ij ∈ [0,1]`:
```
pruned_weight = weight × sigmoid(gate_score)
output        = input @ pruned_weight.T + bias
```
Gradients flow through both `weight` and `gate_score` via autograd.

### 2. Two Gate Modes
| Mode | Mechanism | Exact zeros? |
|------|-----------|-------------|
| `sigmoid` | `g = sigmoid(s)` | ✗ |
| `hard_concrete` | Stretched + clipped sigmoid | ✓ |

Hard Concrete (Louizos et al., 2018) places probability mass exactly AT 0 and 1, enabling truly sparse networks.

### 3. Sparsity Loss
```
Total Loss = CrossEntropy(logits, labels) + λ × Σ gates
```
L1 penalty on gates creates a **constant gradient toward zero**, unlike L2 which weakens as gates shrink.

### 4. Lambda Warm-up
λ ramps from 0 → target over the first 30% of training. This prevents the sparsity loss from overwhelming classification before the network learns useful features.

---

## Usage

```bash
pip install torch torchvision matplotlib numpy

# Default: 3 lambda values, sigmoid mode, 30 epochs
python train.py

# Custom
python train.py --lambdas 1e-4 1e-3 1e-2 --gate_mode hard_concrete --epochs 50
```

## Outputs
| File | Description |
|------|-------------|
| `outputs/gate_distributions.png` | Gate histogram per λ |
| `outputs/training_curves.png` | Accuracy & sparsity vs epochs |
| `outputs/report.md` | Full analysis report |
| `outputs/results.json` | All metrics |
| `checkpoints/best_lambda_*.pt` | Best checkpoint per λ |

---

## Results (Expected)

| Lambda | Test Acc | Sparsity |
|--------|----------|----------|
| 1e-4 | ~83% | ~15% |
| 1e-3 | ~80% | ~50% |
| 1e-2 | ~73% | ~82% |

---

## References
- Louizos et al. (2018) — *L0 Regularization for Neural Networks* (Hard Concrete)
- Han et al. (2015) — *Learning Weights and Connections*
