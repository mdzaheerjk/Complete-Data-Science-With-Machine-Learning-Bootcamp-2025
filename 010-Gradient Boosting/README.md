# 🌲 Gradient Boosting — Complete Job-Ready Notes

> **"Gradient Boosting is gradient descent in function space — instead of updating parameters, we add new functions."**  
> — Jerome Friedman (inventor, 2001)

---

## 📚 Table of Contents

1. [Plain English Intuition](#1-plain-english-intuition)
2. [From AdaBoost → Gradient Boosting](#2-from-adaboost--gradient-boosting)
3. [Gradient Descent — Deep Dive](#3-gradient-descent--deep-dive)
4. [Gradient Boosting — Full Algorithm + Math](#4-gradient-boosting--full-algorithm--math)
5. [Loss Functions + Their Gradients](#5-loss-functions--their-gradients)
6. [Decision Trees as Base Learners](#6-decision-trees-as-base-learners)
7. [Hyperparameters — Every One Explained](#7-hyperparameters--every-one-explained)
8. [Bias-Variance Tradeoff](#8-bias-variance-tradeoff)
9. [Regularization Strategies](#9-regularization-strategies)
10. [Python Code — From Scratch to sklearn](#10-python-code--from-scratch-to-sklearn)
11. [XGBoost, LightGBM, CatBoost](#11-xgboost-lightgbm-catboost)
12. [Feature Importance + Interpretability (SHAP)](#12-feature-importance--interpretability-shap)
13. [Comparison Table: All Ensemble Methods](#13-comparison-table-all-ensemble-methods)
14. [Interview Q&A + Common Mistakes](#14-interview-qa--common-mistakes)
15. [Resources Map](#15-resources-map)
16. [Quick Reference Card](#16-quick-reference-card)

---

## 1. Plain English Intuition

> **Simple statement:** Build trees one at a time. Each new tree fixes the mistakes left by all previous trees combined.

### The Student Analogy

| Round | What happens |
|-------|-------------|
| **Round 1** | You predict everyone's salary as ₹50,000 (the average). You're very wrong for senior engineers and very wrong for interns. |
| **Round 2** | You train a tiny tree specifically on your *errors* — "for the people I was ₹30,000 too low on, add ₹30,000." |
| **Round 3** | Still some error? Train another tiny tree on *those remaining errors*. |
| **...** | Repeat. |
| **Final** | Add all the tiny trees' corrections together. You get a very good predictor. |

Each tiny tree = **weak learner** (intentionally simple, shallow, biased).  
The sum of all weak learners = **strong learner** (low bias, controlled variance).

---

## 2. From AdaBoost → Gradient Boosting

### AdaBoost (1995) — the precursor

```
Algorithm:
1. Assign equal weights to all samples: wᵢ = 1/N
2. Fit tree on weighted samples
3. Increase weights of MISCLASSIFIED samples
4. Decrease weights of correctly classified samples
5. Repeat → weighted majority vote
```

**Limitation:** Tied to exponential loss. Can't change the loss function easily.

### Gradient Boosting (Friedman, 2001) — the generalization

Key insight: **Re-weighting samples ≈ fitting to negative gradient of loss**.

Instead of changing sample weights, directly fit new trees to the **negative gradient (pseudo-residuals)**.

This works for ANY differentiable loss function — regression, classification, ranking, survival analysis.

---

## 3. Gradient Descent — Deep Dive

### Standard Gradient Descent (parameter space)

We minimize a loss L(θ) by updating parameters θ:

```
θ_new = θ_old − η · ∇_θ L(θ)
```

Where:
- `θ` = model parameters (weights, biases)
- `η` = learning rate (step size)
- `∇_θ L(θ)` = gradient of loss w.r.t parameters

### Variants

```
Batch GD    : use ALL training samples to compute gradient
Stochastic GD: use ONE sample per step  → fast but noisy
Mini-batch GD: use a BATCH of k samples → balance of both
```

### Gradient Descent in Function Space (Gradient Boosting)

Instead of optimizing parameters θ, we optimize the **prediction function F(x)** itself:

```
F_m(x) = F_{m-1}(x) + η · h_m(x)
```

Where `h_m(x)` is the new tree fitted to the **negative gradient**:

```
h_m(x) ≈ - ∂L/∂F(x)    evaluated at F = F_{m-1}
```

#### Why this works:

Adding a tree that approximates the negative gradient = moving the function F in the direction that decreases loss. This is exactly gradient descent, but instead of moving a number (θ), we move an entire function (F).

```
Parameter space:    θ      → θ - η · g           (move a number)
Function space:     F(x)   → F(x) + η · h_m(x)   (move a function)
```

---

## 4. Gradient Boosting — Full Algorithm + Math

### Setup

```
Training data   : {(x₁,y₁), (x₂,y₂), ..., (xₙ,yₙ)}
Loss function   : L(y, F(x))  — must be differentiable in F
Number of trees : M
Learning rate   : η ∈ (0, 1]
```

---

### Algorithm (Friedman, 2001)

#### Step 0 — Initialize with a constant

```
F₀(x) = argmin_γ  Σᵢ L(yᵢ, γ)
```

For **MSE loss** (L = ½(y−F)²):
```
∂/∂γ Σᵢ ½(yᵢ−γ)² = 0  →  Σᵢ(yᵢ−γ) = 0  →  γ = ȳ
∴ F₀(x) = ȳ = (1/N) Σᵢ yᵢ
```

For **log-loss** (binary classification):
```
F₀(x) = log(p₀ / (1−p₀))    where p₀ = Σ yᵢ / N  (class proportion)
```

---

#### Step 1 — For m = 1, 2, ..., M:

**1a. Compute pseudo-residuals (= negative gradient)**

```
rᵢₘ = − [ ∂L(yᵢ, F(xᵢ)) / ∂F(xᵢ) ]_{F = F_{m-1}}
```

| Loss | L(y, F) | Gradient ∂L/∂F | Pseudo-residuals rᵢ |
|------|---------|----------------|----------------------|
| MSE | ½(y−F)² | F−y | y−F (the residuals!) |
| MAE | |y−F| | sign(F−y) | sign(y−F) |
| Log-loss | −y·log(p)−(1−y)·log(1−p) | p−y | y−p |

**For MSE specifically:**
```
rᵢ = − ∂/∂F [½(yᵢ−F)²]
   = − (F − yᵢ)
   = yᵢ − F_{m-1}(xᵢ)       ← just the ordinary residuals
```

This is why for regression, GB literally fits trees to residuals!

---

**1b. Fit a regression tree hₘ to the pseudo-residuals**

```
hₘ = fit_tree(X, r)      where r = (r₁ₘ, r₂ₘ, ..., rₙₘ)
```

The tree partitions input space into J leaf regions:
```
R₁ₘ, R₂ₘ, ..., R_{Jm}
```

---

**1c. Compute optimal leaf value (line search)**

For each leaf region j, find the best constant γ that minimizes loss:

```
γ_{jm} = argmin_γ  Σ_{xᵢ ∈ R_{jm}}  L(yᵢ,  F_{m-1}(xᵢ) + γ)
```

For MSE: `γ_{jm} = mean of residuals rᵢ in leaf j`

For log-loss: `γ_{jm} = Σ rᵢ / Σ pᵢ(1−pᵢ)` (Newton-Raphson step)

---

**1d. Update the model**

```
F_m(x) = F_{m-1}(x) + η · Σⱼ γ_{jm} · 1[x ∈ R_{jm}]
                                    ^
                       (output of tree hₘ at x)
```

---

### Final Model

```
F_M(x) = F₀(x) + η Σ_{m=1}^{M} hₘ(x)
```

For regression: output is F_M(x) directly.  
For binary classification: `p = sigmoid(F_M(x)) = 1 / (1 + e^{-F_M(x)})`  
For multiclass: one tree sequence per class, output = softmax(F_M^k(x))

---

### Math Summary Table

```
Symbol    Meaning
-------------------------------------------------
N         number of training samples
M         number of trees (n_estimators)
η         learning rate (0 < η ≤ 1)
L(y,F)    loss function
rᵢ        pseudo-residual for sample i
hₘ        mth weak learner (regression tree)
J         number of leaves per tree
R_{jm}    jth leaf region of mth tree
γ_{jm}    optimal value for leaf j of tree m
F_m(x)    model after m trees
```

---

## 5. Loss Functions + Their Gradients

### Regression

| Name | sklearn `loss=` | L(y,F) | −∂L/∂F (pseudo-residuals) | Use when |
|------|----------------|--------|---------------------------|----------|
| MSE | `'squared_error'` | ½(y−F)² | y−F | default, symmetric errors |
| MAE | `'absolute_error'` | \|y−F\| | sign(y−F) | outliers present |
| Huber | `'huber'` | MSE if \|r\|≤δ, else MAE | y−F (MSE zone) / δ·sign(y−F) (MAE zone) | best of both |
| Quantile | `'quantile'` | α·max(y−F,0) + (1−α)·max(F−y,0) | α or α−1 | prediction intervals |

**Huber loss detail:**

```
L_δ(y,F) = {  ½(y−F)²           if |y−F| ≤ δ    ← MSE near the center
            {  δ(|y−F| − ½δ)    if |y−F| > δ    ← MAE in the tails
```

### Classification

| Task | sklearn `loss=` | L(y,F) | −∂L/∂F |
|------|----------------|--------|--------|
| Binary | `'log_loss'` | −y·log(σ(F))−(1−y)·log(1−σ(F)) | y−σ(F) = y−p |
| Binary | `'exponential'` | e^{−yF} (AdaBoost) | y·e^{−yF} |
| Multiclass | `'log_loss'` | −Σₖ yₖ·log(pₖ) | yₖ−pₖ per class |

---

## 6. Decision Trees as Base Learners

Gradient Boosting uses **regression trees** for BOTH regression and classification.

### Why regression trees for classification?

- Trees output a **real-valued score** (log-odds), not a class label
- Final classification uses `sigmoid` or `softmax` transformation
- Pseudo-residuals are always real numbers → regression tree fits perfectly

### Why shallow trees?

```
max_depth = 1  → decision stump, too weak, needs thousands of trees
max_depth = 3  → sweet spot (default, interactions up to 3-way)
max_depth = 5  → for complex data, but risk of overfitting
max_depth = 10 → almost always overfits
```

Shallow trees = high bias + low variance = **perfect weak learners** for boosting.
Boosting's job is to reduce bias iteratively. Each tree needs to be correctable.

### Tree split criterion: `criterion='friedman_mse'`

Friedman's improvement over standard MSE — uses expected improvement from a split:

```
Friedman MSE = (w_L · w_R / (w_L + w_R)) · (ȳ_L − ȳ_R)²
```

This prefers splits where left and right means differ greatly, weighted by sample counts.

---

## 7. Hyperparameters — Every One Explained

```python
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
```

### Group 1: Number of Trees and Learning Rate (most important!)

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `n_estimators` | 100 | 100–5000 | More trees → lower bias, risk of overfitting |
| `learning_rate` | 0.1 | 0.001–0.5 | Shrinks each tree. Smaller η → need more trees |

**The fundamental tradeoff:**

```
High η (0.3)  + Low  n_estimators (100)  → fast training, may underfit/overfit
Low  η (0.01) + High n_estimators (3000) → slow training, better generalization
```

**Rule:** If you lower `learning_rate` by 10×, increase `n_estimators` by ~10×.

---

### Group 2: Tree Complexity

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `max_depth` | 3 | 1–10 | Primary tree complexity control |
| `min_samples_split` | 2 | 2–50 | Min samples to attempt a split |
| `min_samples_leaf` | 1 | 1–50 | Min samples in any leaf |
| `min_weight_fraction_leaf` | 0.0 | 0.0–0.5 | Like min_samples_leaf but weighted |
| `max_leaf_nodes` | None | 4–100 | Alternative to max_depth (best-first growth) |
| `min_impurity_decrease` | 0.0 | 0.0–0.1 | Split only if impurity drops ≥ this |
| `ccp_alpha` | 0.0 | 0.0–0.1 | Cost-complexity pruning (higher = more pruning) |

---

### Group 3: Stochastic Regularization

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `subsample` | 1.0 | 0.5–1.0 | Fraction of training rows per tree (< 1 = Stochastic GB) |
| `max_features` | None | `'sqrt'`, `'log2'`, float | Features considered per split |

**Stochastic Gradient Boosting** (subsample < 1):
- Randomly sample rows before fitting each tree
- Reduces variance (like bagging) + regularizes
- Also speeds up training
- Typical: `subsample=0.8`

---

### Group 4: Early Stopping

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `n_iter_no_change` | None | 5–50 | Stop if no improvement after this many rounds |
| `validation_fraction` | 0.1 | 0.05–0.3 | Fraction of train data held out for early stopping |
| `tol` | 1e-4 | 1e-5–1e-3 | Min improvement to count as "progress" |

---

### Group 5: Loss and Criterion

| Parameter | Default | Options | Notes |
|-----------|---------|---------|-------|
| `loss` (Classifier) | `'log_loss'` | `'log_loss'`, `'exponential'` | Use log_loss always unless AdaBoost comparison |
| `loss` (Regressor) | `'squared_error'` | `'squared_error'`, `'absolute_error'`, `'huber'`, `'quantile'` | |
| `criterion` | `'friedman_mse'` | `'friedman_mse'`, `'squared_error'` | Friedman MSE almost always better |
| `alpha` | 0.9 | 0.0–1.0 | Quantile for `loss='quantile'` or `'huber'` |

---

### Group 6: Misc

| Parameter | Default | Notes |
|-----------|---------|-------|
| `warm_start` | False | If True, reuse previous fit — add more trees to existing model |
| `random_state` | None | Set for reproducibility |
| `n_jobs` | None | Not used (trees built sequentially by design — use HistGB for parallel) |
| `verbose` | 0 | Set to 1 to see training progress |

---

## 8. Bias-Variance Tradeoff

```
Total Expected Error = Bias² + Variance + Irreducible Noise
```

### How Gradient Boosting addresses each:

```
Iteration 0:    F₀ = ȳ             → HIGH BIAS (always predicts mean)
Iteration 1:    F₁ = F₀ + η·h₁    → bias drops
Iteration 2:    F₂ = F₁ + η·h₂    → bias drops further
...
Iteration M:    F_M                → bias ≈ 0  (if M large enough)
```

**Boosting reduces bias** by sequentially correcting systematic errors.

**Variance increases** as M grows (can memorize training data).

### Regularization reduces variance:

| Technique | Variance effect |
|-----------|----------------|
| Small `learning_rate` | Shrinks each correction → smooth function |
| `subsample < 1` | Random row sampling → trees don't see all patterns |
| `max_depth` small | Each tree has fewer degrees of freedom |
| `min_samples_leaf` large | Leaves must generalize, not overfit |

### The bias-variance diagram by hyperparameter:

```
                    ↑ Error
                    |
         Variance   |  \
         (overfitting|   \       <- optimal zone
                    |    \  ___/
         Bias       |     \/
         (underfitting|
                    |_________________→
                        n_estimators (with small η)
                        (or max_depth)
```

---

## 9. Regularization Strategies

### Strategy 1: Shrinkage (Learning Rate)

```python
# The model is: F_m = F_{m-1} + η · h_m
# Smaller η → each tree contributes less → slower convergence → less overfit

gb = GradientBoostingRegressor(
    learning_rate=0.05,    # was 0.1
    n_estimators=400       # compensate: was 200
)
```

### Strategy 2: Subsampling (Stochastic GB)

```python
gb = GradientBoostingRegressor(
    subsample=0.8,         # use 80% of rows per tree
    max_features='sqrt'    # use sqrt(p) features per split
)
# subsample < 1 also allows computing out-of-bag error estimate
```

### Strategy 3: Tree Depth

```python
gb = GradientBoostingRegressor(
    max_depth=3,           # default, usually good
    # OR use max_leaf_nodes instead:
    max_leaf_nodes=8,      # exactly 8 leaves per tree
    max_depth=None         # must set None when using max_leaf_nodes
)
```

### Strategy 4: Early Stopping

```python
gb = GradientBoostingClassifier(
    n_estimators=2000,
    learning_rate=0.01,
    n_iter_no_change=30,     # stop if 30 rounds show no improvement
    validation_fraction=0.15,
    tol=1e-5,
    random_state=42
)
gb.fit(X_train, y_train)
print(f"Stopped at: {gb.n_estimators_} trees")
```

### Strategy 5: Pruning (ccp_alpha)

```python
# Higher alpha → more pruning → simpler trees
gb = GradientBoostingRegressor(ccp_alpha=0.01)
```

---

## 10. Python Code — From Scratch to sklearn

### 10.1 Manual Gradient Boosting from Scratch (MSE)

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

np.random.seed(42)

# Toy data
X = np.linspace(0, 10, 200).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.normal(0, 0.2, 200)

# ---------- MANUAL GB ----------
learning_rate = 0.1
n_trees = 100
trees = []

# Step 0: Initialize to mean
F = np.full_like(y, y.mean(), dtype=float)
print(f"Initial MSE (F₀ = mean): {mean_squared_error(y, F):.4f}")

# Steps 1..M
for m in range(n_trees):
    # 1a. Pseudo-residuals (negative gradient of MSE)
    residuals = y - F          # for MSE: rᵢ = yᵢ - F(xᵢ)

    # 1b. Fit regression tree to residuals
    tree = DecisionTreeRegressor(max_depth=3)
    tree.fit(X, residuals)
    trees.append(tree)

    # 1d. Update model
    F += learning_rate * tree.predict(X)

print(f"Final  MSE (M={n_trees} trees): {mean_squared_error(y, F):.4f}")

plt.figure(figsize=(10, 4))
plt.scatter(X, y, alpha=0.3, s=15, label='Data')
plt.plot(X, F, color='crimson', lw=2, label=f'GB prediction (M={n_trees})')
plt.legend(); plt.title('Manual Gradient Boosting — Regression'); plt.tight_layout()
plt.show()
```

### 10.2 Full sklearn Regression Pipeline

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings; warnings.filterwarnings('ignore')

# ── 1. Data
housing = fetch_california_housing()
X, y = housing.data, housing.target
feature_names = housing.feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ── 2. Fit
gb_reg = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    max_features='sqrt',
    min_samples_leaf=5,
    random_state=42
)
gb_reg.fit(X_train, y_train)

# ── 3. Evaluate
y_pred = gb_reg.predict(X_test)
rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
mae    = mean_absolute_error(y_test, y_pred)
r2     = r2_score(y_test, y_pred)

print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"R²   : {r2:.4f}")

# ── 4. Staged predictions (track learning curve)
train_mse, val_mse = [], []
for F_train in gb_reg.staged_predict(X_train):
    train_mse.append(mean_squared_error(y_train, F_train))
for F_val in gb_reg.staged_predict(X_test):
    val_mse.append(mean_squared_error(y_test, F_val))

best_n = np.argmin(val_mse) + 1
print(f"\nOptimal n_estimators: {best_n}")

plt.figure(figsize=(10, 5))
plt.plot(train_mse, label='Train MSE', color='blue', lw=1)
plt.plot(val_mse,   label='Val MSE',   color='red',  lw=1)
plt.axvline(best_n, color='green', ls='--', label=f'Best = {best_n}')
plt.xlabel('Number of Trees'); plt.ylabel('MSE')
plt.title('GB Learning Curve (staged_predict)')
plt.legend(); plt.tight_layout(); plt.show()
```

### 10.3 Full sklearn Classification Pipeline

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (accuracy_score, roc_auc_score,
                              classification_report, RocCurveDisplay,
                              ConfusionMatrixDisplay)
from sklearn.model_selection import StratifiedKFold, cross_validate

data   = load_breast_cancer()
X, y   = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

gb_clf = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    max_features='sqrt',
    min_samples_leaf=3,
    n_iter_no_change=20,
    validation_fraction=0.1,
    random_state=42
)
gb_clf.fit(X_train, y_train)

y_pred  = gb_clf.predict(X_test)
y_proba = gb_clf.predict_proba(X_test)[:, 1]

print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}")
print(f"Stopped  : {gb_clf.n_estimators_} trees (early stopping)")
print()
print(classification_report(y_test, y_pred, target_names=data.target_names))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
RocCurveDisplay.from_predictions(y_test, y_proba, ax=axes[0], name='GB')
ConfusionMatrixDisplay.from_predictions(y_test, y_pred,
    display_labels=data.target_names, ax=axes[1])
plt.suptitle('Gradient Boosting — Classification Evaluation')
plt.tight_layout(); plt.show()
```

### 10.4 Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint, uniform

# ── Option A: GridSearchCV (exhaustive)
param_grid = {
    'n_estimators'  : [100, 300, 500],
    'learning_rate' : [0.01, 0.05, 0.1],
    'max_depth'     : [2, 3, 5],
    'subsample'     : [0.7, 0.8, 1.0],
}
grid = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1
)
grid.fit(X_train, y_train)
print("Best params :", grid.best_params_)
print("Best ROC-AUC:", grid.best_score_)

# ── Option B: RandomizedSearchCV (faster for large search spaces)
param_dist = {
    'n_estimators'  : randint(100, 2000),
    'learning_rate' : uniform(0.005, 0.2),
    'max_depth'     : randint(2, 8),
    'subsample'     : uniform(0.5, 0.5),
    'max_features'  : ['sqrt', 'log2', 0.5],
    'min_samples_leaf': randint(1, 20)
}
rand_search = RandomizedSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_dist, n_iter=50, cv=5, scoring='roc_auc',
    n_jobs=-1, random_state=42, verbose=1
)
rand_search.fit(X_train, y_train)
print("Best params :", rand_search.best_params_)
```

### 10.5 Early Stopping Properly

```python
# Method 1: sklearn built-in
gb = GradientBoostingClassifier(
    n_estimators=5000,
    learning_rate=0.01,
    max_depth=3,
    n_iter_no_change=30,
    validation_fraction=0.1,
    tol=1e-5,
    random_state=42
)
gb.fit(X_train, y_train)
print(f"Best n_estimators: {gb.n_estimators_}")

# Method 2: staged_predict manual early stopping
from sklearn.metrics import log_loss

gb2 = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.05,
                                   max_depth=3, random_state=42)
gb2.fit(X_train, y_train)

val_losses = [log_loss(y_test, proba)
              for proba in gb2.staged_predict_proba(X_test)]

best_n2 = np.argmin(val_losses) + 1
print(f"Manual early stopping: {best_n2} trees, val_loss={val_losses[best_n2-1]:.4f}")

# Method 3: warm_start (add trees incrementally)
gb3 = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05,
                                   max_depth=3, warm_start=True, random_state=42)
best_loss3, best_n3 = float('inf'), 0
for n in range(100, 2000, 50):
    gb3.n_estimators = n
    gb3.fit(X_train, y_train)
    loss = log_loss(y_test, gb3.predict_proba(X_test))
    if loss < best_loss3:
        best_loss3, best_n3 = loss, n
print(f"Warm-start best: {best_n3} trees, loss={best_loss3:.4f}")
```

### 10.6 Feature Importance — All Three Methods

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

# ── Method 1: Impurity-based (fast, can be biased)
importances_1 = gb_clf.feature_importances_
idx1 = np.argsort(importances_1)[::-1]

# ── Method 2: Permutation importance (slower, more reliable)
perm = permutation_importance(gb_clf, X_test, y_test,
                               n_repeats=20, random_state=42, n_jobs=-1)
importances_2 = perm.importances_mean
idx2 = np.argsort(importances_2)[::-1]

# ── Plot both
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].barh(range(10), importances_1[idx1[:10]][::-1], color='steelblue')
axes[0].set_yticks(range(10))
axes[0].set_yticklabels(data.feature_names[idx1[:10]][::-1])
axes[0].set_title('Impurity-based importance')

axes[1].barh(range(10),
             importances_2[idx2[:10]][::-1],
             xerr=perm.importances_std[idx2[:10]][::-1],
             color='salmon')
axes[1].set_yticks(range(10))
axes[1].set_yticklabels(data.feature_names[idx2[:10]][::-1])
axes[1].set_title('Permutation importance (±std)')

plt.suptitle('Feature Importance Comparison')
plt.tight_layout(); plt.show()

# ── Method 3: SHAP (best for interpretation)
# pip install shap
import shap

explainer   = shap.TreeExplainer(gb_clf)
shap_values = explainer.shap_values(X_test)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
shap.summary_plot(shap_values, X_test,
                  feature_names=data.feature_names, show=False)
plt.tight_layout(); plt.show()

# Single prediction explanation
shap.waterfall_plot(shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=X_test[0],
    feature_names=list(data.feature_names)))
```

### 10.7 Cross Validation + Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Pipeline (GB doesn't need scaling but good practice)
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('gb', GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=42
    ))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X, y, cv=cv,
                          scoring='roc_auc', n_jobs=-1)
print(f"CV ROC-AUC: {scores.mean():.4f} ± {scores.std():.4f}")

# Full cross_validate with multiple metrics
from sklearn.model_selection import cross_validate
results = cross_validate(pipe, X, y, cv=cv,
                          scoring=['roc_auc', 'accuracy', 'f1'],
                          n_jobs=-1)
for metric, vals in results.items():
    if metric.startswith('test_'):
        print(f"{metric[5:]:<12} {vals.mean():.4f} ± {vals.std():.4f}")
```

### 10.8 HistGradientBoosting (Fast sklearn, for big data)

```python
from sklearn.ensemble import (HistGradientBoostingClassifier,
                               HistGradientBoostingRegressor)

# For datasets > 10k samples — much faster than GradientBoosting
hgb = HistGradientBoostingClassifier(
    max_iter=1000,             # equivalent to n_estimators
    learning_rate=0.05,
    max_depth=4,
    min_samples_leaf=20,
    l2_regularization=0.1,    # L2 on leaf values
    max_bins=255,              # number of bins for histogram (like LightGBM)
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42
)
hgb.fit(X_train, y_train)

print("HistGB Accuracy:", accuracy_score(y_test, hgb.predict(X_test)))
print("HistGB ROC-AUC :", roc_auc_score(y_test, hgb.predict_proba(X_test)[:,1]))

# HistGB handles NaN natively!
X_with_nan = X_train.copy()
X_with_nan[np.random.rand(*X_train.shape) < 0.1] = np.nan
hgb.fit(X_with_nan, y_train)   # works without imputation
```

---

## 11. XGBoost, LightGBM, CatBoost

### 11.1 XGBoost

**Key extras over sklearn GB:**
- L1 (`reg_alpha`) and L2 (`reg_lambda`) regularization on leaf weights
- 2nd-order Taylor expansion for better leaf value optimization
- Native missing value handling
- GPU support
- Much faster (parallel column scanning)

```python
# pip install xgboost
import xgboost as xgb

xgb_clf = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,    # fraction of features per tree
    colsample_bylevel=1.0,   # fraction of features per tree level
    colsample_bynode=1.0,    # fraction of features per node
    reg_alpha=0.1,           # L1
    reg_lambda=1.0,          # L2
    gamma=0.1,               # min loss reduction for a split
    min_child_weight=3,      # like min_samples_leaf (sum of hessians)
    eval_metric='logloss',
    early_stopping_rounds=20,
    use_label_encoder=False,
    n_jobs=-1,
    random_state=42
)

xgb_clf.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=False
)

print("XGB best iteration:", xgb_clf.best_iteration)
print("XGB Accuracy:", accuracy_score(y_test, xgb_clf.predict(X_test)))
print("XGB ROC-AUC :", roc_auc_score(y_test, xgb_clf.predict_proba(X_test)[:,1]))
```

**XGBoost math extra — leaf weight with L2:**

```
w_j* = − (Σ gᵢ) / (Σ hᵢ + λ)

where gᵢ = ∂L/∂F (first derivative)
      hᵢ = ∂²L/∂F² (second derivative / hessian)
      λ   = reg_lambda (L2 penalty)
```

This is a Newton step (uses curvature), vs sklearn GB which just uses gradient (first-order).

---

### 11.2 LightGBM

**Key differences:**
- **Leaf-wise growth** (best-first) vs level-wise in XGBoost/sklearn
- Histogram-based splits (much faster, less memory)
- `num_leaves` is the primary complexity parameter
- Native categorical features
- GOSS (Gradient-based One-Side Sampling) — keeps large-gradient samples, randomly samples small-gradient ones

```python
# pip install lightgbm
import lightgbm as lgb

lgb_clf = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,            # main complexity param (default 31)
    max_depth=-1,             # -1 = unlimited (use num_leaves)
    subsample=0.8,            # row sampling
    colsample_bytree=0.8,     # feature sampling per tree
    reg_alpha=0.1,
    reg_lambda=0.1,
    min_child_samples=20,     # like min_samples_leaf
    class_weight='balanced',  # for imbalanced data
    random_state=42,
    n_jobs=-1
)

callbacks = [lgb.early_stopping(30), lgb.log_evaluation(100)]

lgb_clf.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=callbacks
)

print("LGB Accuracy:", accuracy_score(y_test, lgb_clf.predict(X_test)))
print("LGB ROC-AUC :", roc_auc_score(y_test, lgb_clf.predict_proba(X_test)[:,1]))

# Plot training history
lgb.plot_metric(lgb_clf.evals_result_, metric='binary_logloss')
```

**Level-wise vs Leaf-wise:**

```
Level-wise (XGBoost, sklearn):           Leaf-wise (LightGBM):
        Split ALL leaves at each depth         Split BEST leaf regardless of depth

        Root                                   Root
       /    \                                 /    \
      A      B        →                      A      B
     / \    / \                             / \
    C   D  E   F                           C   D      ← only best leaf splits
```

Leaf-wise: lower loss but can overfit → control with `num_leaves` and `min_child_samples`.

---

### 11.3 CatBoost

**Key features:**
- Native categorical encoding (ordered target encoding, no need for preprocessing)
- Ordered boosting (prevents target leakage by using different permutations)
- GPU support
- Great out-of-the-box performance

```python
# pip install catboost
from catboost import CatBoostClassifier, Pool

# Identify categorical columns
cat_features = [0, 3]   # column indices of categorical features

train_pool = Pool(X_train, y_train, cat_features=cat_features)
test_pool  = Pool(X_test,  y_test,  cat_features=cat_features)

cat_clf = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=5,
    l2_leaf_reg=3.0,
    subsample=0.8,
    rsm=0.8,                 # colsample equivalent
    eval_metric='AUC',
    early_stopping_rounds=30,
    use_best_model=True,
    random_seed=42,
    verbose=100
)

cat_clf.fit(train_pool, eval_set=test_pool)

print("CAT Accuracy:", accuracy_score(y_test, cat_clf.predict(X_test)))
print("CAT ROC-AUC :", roc_auc_score(y_test, cat_clf.predict_proba(X_test)[:,1]))
```

---

### 11.4 Comparison Table

| Feature | sklearn GB | HistGB | XGBoost | LightGBM | CatBoost |
|---------|-----------|--------|---------|----------|---------|
| Speed | Slow | Fast | Fast | Fastest | Fast |
| Memory | Medium | Low | Medium | Low | Medium |
| GPU | No | No | Yes | Yes | Yes |
| Missing values | No | Yes | Yes | Yes | Yes |
| Categorical | No | Partial | No | Partial | Yes (native) |
| Tree growth | Level | Level | Level | Leaf-wise | Symmetric |
| L1/L2 reg | No | L2 | Yes | Yes | L2 |
| Best for | Learning | Big data | Kaggle | Big data | Categoricals |
| sklearn API | Native | Native | Yes | Yes | Yes |

---

## 12. Feature Importance + Interpretability (SHAP)

### Types of feature importance

#### 1. Impurity-based (Mean Decrease in Impurity)

```python
importances = gb_clf.feature_importances_
# Pros: fast, built-in
# Cons: biased toward high-cardinality features
```

Computed as: for each feature, total reduction in split criterion (MSE/Gini) weighted by samples, averaged across all trees.

#### 2. Permutation Importance

```python
from sklearn.inspection import permutation_importance
result = permutation_importance(gb_clf, X_test, y_test,
                                 n_repeats=30, random_state=42)
# For each feature:
#   1. Shuffle that feature's values
#   2. Measure drop in performance
#   3. Average over n_repeats permutations
# Pros: model-agnostic, no cardinality bias
# Cons: slow, correlated features split importance between them
```

#### 3. SHAP (SHapley Additive exPlanations) — Best

Based on Shapley values from cooperative game theory.

For a single prediction `f(x)`:
```
f(x) = E[f(x)] + Σᵢ φᵢ

where φᵢ = contribution of feature i
      E[f(x)] = model's expected output (base value)
```

Properties: local accuracy (explanations sum to prediction), consistency, missingness.

```python
import shap

# TreeExplainer: fast exact SHAP for tree models
explainer   = shap.TreeExplainer(gb_clf)
shap_values = explainer.shap_values(X_test)  # shape: (n_samples, n_features)

# Global: feature importance
shap.summary_plot(shap_values, X_test, feature_names=data.feature_names)

# Global: bar chart
shap.summary_plot(shap_values, X_test,
                  feature_names=data.feature_names, plot_type='bar')

# Local: single prediction
shap.waterfall_plot(shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=X_test[0],
    feature_names=list(data.feature_names)))

# Dependence plot: how feature X affects SHAP value
shap.dependence_plot('worst radius', shap_values, X_test,
                      feature_names=data.feature_names)
```

### Partial Dependence Plots (PDP)

```python
from sklearn.inspection import PartialDependenceDisplay

features = [0, 1, (0, 1)]   # single features + interaction
PartialDependenceDisplay.from_estimator(
    gb_clf, X_test, features,
    feature_names=data.feature_names,
    kind='both'              # 'average' (PDP) + 'individual' (ICE)
)
plt.suptitle('Partial Dependence Plots')
plt.tight_layout(); plt.show()
```

---

## 13. Comparison Table: All Ensemble Methods

| | Bagging (RF) | Boosting (GB) | Stacking |
|---|---|---|---|
| **Base learners** | Independent | Sequential | Heterogeneous |
| **Parallelizable** | Yes | No (sequential) | Partial |
| **Primary goal** | Reduce variance | Reduce bias | Both |
| **Sensitive to noise** | Low | Moderate-High | Low |
| **Sensitive to outliers** | Low | Moderate (MSE) | Varies |
| **Overfitting risk** | Low | Medium (needs tuning) | Low |
| **Training speed** | Fast | Medium-Slow | Slow |
| **Hyperparameters** | Few | Many | Many |
| **Interpretability** | Moderate | Moderate | Low |
| **Typical accuracy** | High | Higher | Highest |

---

## 14. Interview Q&A + Common Mistakes

### Must-Know Interview Questions

**Q1: What are pseudo-residuals and why are they called that?**

A: Pseudo-residuals are the negative gradient of the loss function w.r.t the current model output:
`rᵢ = −∂L(yᵢ, F(xᵢ)) / ∂F(xᵢ)`.  
For MSE loss they equal the ordinary residuals `yᵢ − F(xᵢ)`.  
For other losses they generalize residuals (hence "pseudo"). They tell each tree "in what direction and by how much should the model change for this sample."

---

**Q2: Why do we use shallow trees (max_depth=3) as base learners?**

A: Shallow trees are high-bias, low-variance weak learners. Boosting's job is to reduce bias iteratively — so we need trees that are "fixable." Deep trees would memorize residuals (low bias but high variance), causing the ensemble to overfit the training data.

---

**Q3: What is the difference between Gradient Boosting and AdaBoost?**

A: AdaBoost re-weights training samples (misclassified samples get higher weights, then the next tree focuses on them). Gradient Boosting fits the next tree to the negative gradient (pseudo-residuals) of the loss. GB is a strict generalization — AdaBoost corresponds to GB with exponential loss. GB works for any differentiable loss; AdaBoost is primarily for classification with exponential loss.

---

**Q4: What does the learning rate actually do mathematically?**

A: It scales the contribution of each tree: `Fₘ = F_{m-1} + η·hₘ`. A smaller η means each tree corrects only a small fraction of the remaining error. This acts as regularization — the model converges more slowly but is less likely to overfit because no single tree dominates the ensemble.

---

**Q5: Why does LightGBM use leaf-wise growth and is it always better?**

A: Leaf-wise growth always splits the leaf with the largest loss reduction, regardless of depth. This achieves lower training loss with fewer splits (more efficient). However, it can create very deep trees quickly, causing overfitting. You must set `num_leaves` carefully (and usually `min_child_samples`) to control this. It is not always better — for small datasets or noisy data, level-wise growth (XGBoost, sklearn) is more stable.

---

**Q6: How does XGBoost's objective differ from sklearn's Gradient Boosting?**

A: sklearn GB uses first-order Taylor expansion (gradient only) to compute pseudo-residuals, then does a 1D line search for optimal leaf values. XGBoost uses a second-order Taylor expansion (gradient + hessian), solving exactly for leaf weights:

```
w_j = − Σgᵢ / (Σhᵢ + λ)
```

This is a Newton step, which converges faster and allows regularization via λ directly in the leaf weight formula.

---

### Common Mistakes

| Mistake | Why bad | Fix |
|---------|---------|-----|
| High `learning_rate` (0.5+) | Overshoots, unstable | Use 0.01–0.1 |
| Not using early stopping | Overfits with too many trees | Set `n_iter_no_change` |
| Large `max_depth` (8+) | Each tree overfits residuals | Use 2–5 |
| Not cross-validating | Overly optimistic evaluation | Always use CV |
| Using GB on 100k+ rows | Slow | Use HistGB or LightGBM |
| Ignoring outliers | MSE amplifies them | Use `loss='huber'` |
| Not checking staged_predict | Miss optimal n_estimators | Always plot learning curve |
| Feature scaling | Unnecessary | GB is scale-invariant |
| Tuning n_estimators and lr independently | Wrong interaction | Always tune together |

---

### When to use what

```
Dataset size < 10k    → sklearn GradientBoosting
Dataset size 10k-1M   → XGBoost or LightGBM
Dataset size > 1M     → LightGBM or HistGradientBoosting
Many categoricals     → CatBoost
Need sklearn API      → HistGradientBoosting (sklearn 1.0+)
GPU available         → XGBoost or LightGBM (device='gpu')
Tabular data overall  → XGBoost (most battle-tested in industry)
```

---

## 15. Resources Map

### 📙 Hands-On Machine Learning — Aurélien Géron (3rd edition)

**Chapter 7: Ensemble Learning and Random Forests**

- Section 7.5: Boosting → AdaBoost → GBRT (Gradient Boosted Regression Trees)
- Key code: `staged_predict`, early stopping, XGBoost integration
- Code: [github.com/ageron/handson-ml3/blob/main/07_ensemble_learning.ipynb](https://github.com/ageron/handson-ml3/blob/main/07_ensemble_learning.ipynb)

```
What to read:
• p.225–235: AdaBoost → Gradient Boosting transition
• p.236–240: Learning rate + n_estimators tradeoff
• p.240–243: XGBoost chapter
• Code: GradientBoostingRegressor staged_predict example (p.238)
```

---

### 📗 Introduction to Statistical Learning (ISLP) — James, Witten, Hastie, Tibshirani

**Chapter 8.2: Boosting**

- Algorithm 8.2: Boosting for regression trees (3 parameters: B, λ, d)
- Explains: shrinkage (λ = learning rate), interaction depth (d = max_depth)
- Python labs: Lab 8.3 in ISLP companion book
- Free PDF + code: [statlearning.com](https://www.statlearning.com)

```
Key passages:
• p.344–349: Algorithm, intuition, tuning parameters
• p.349: "Unlike fitting a single large decision tree to the data, which amounts to
  fitting the data hard... the boosting approach instead learns slowly"
• Lab 8.3: BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
```

---

### 📕 ESL — Elements of Statistical Learning (Hastie, Tibshirani, Friedman)

**Chapter 10: Boosting and Additive Trees**

- The deepest mathematical treatment
- 10.1: Introduction to boosting
- 10.3: Forward stagewise additive modeling
- 10.4: Exponential loss and AdaBoost
- 10.9: Boosting trees
- 10.10: Numerical optimization via gradient boosting
- 10.11: Right-sized trees for boosting
- 10.12: Regularization (shrinkage, subsampling)
- Free PDF: [web.stanford.edu/~hastie/ElemStatLearn](https://web.stanford.edu/~hastie/ElemStatLearn/)

---

### 🎥 StatQuest with Josh Starmer — YouTube

**Gradient Boosting series (WATCH IN ORDER):**

| # | Title | What you learn |
|---|-------|----------------|
| 1 | [Gradient Boost Part 1: Regression Main Ideas](https://www.youtube.com/watch?v=3CC4N4z3GJc) | Intuition, residuals, step-by-step |
| 2 | [Gradient Boost Part 2: Regression Details](https://www.youtube.com/watch?v=2xudPOBz-vs) | Math derivation for MSE, leaf values |
| 3 | [Gradient Boost Part 3: Classification Main Ideas](https://www.youtube.com/watch?v=jxuNLH5dXCs) | Log-odds, classification framing |
| 4 | [Gradient Boost Part 4: Classification Details](https://www.youtube.com/watch?v=StWoqdkvDPk) | Full math derivation for log-loss |
| 5 | [XGBoost Part 1: Trees](https://www.youtube.com/watch?v=OtD8wVaFm6E) | Similarity score, gain |
| 6 | [XGBoost Part 2: Math Details](https://www.youtube.com/watch?v=ZVFeW798-2I) | 2nd order Taylor, regularization |
| 7 | [XGBoost Part 3: Math Details continued](https://www.youtube.com/watch?v=oRrKeUCEbq8) | Pruning, gamma |
| 8 | [XGBoost Part 4: Crazy Cool Optimizations](https://www.youtube.com/watch?v=oRrKeUCEbq8) | Approximate split, sparsity |
| 9 | [AdaBoost](https://www.youtube.com/watch?v=LsK-xG1cLYA) | Predecessor, sample weights |

> **Start with Part 1** — Josh's visual trees-to-residuals walkthrough is the clearest explanation that exists.

---

### 🎓 Andrew Ng ML Specialization — Coursera

**Course 2: Advanced Learning Algorithms**  
**Week 4: Decision Trees + Tree Ensembles**

Key lectures to watch:
- "Using multiple decision trees" — why ensembles beat single trees
- "Random forest algorithm" — bagging vs boosting comparison
- "XGBoost" — Andrew Ng's recommended algorithm for tabular data
- "When to use decision trees vs neural networks" — practical career advice

**Andrew Ng's exact quote:**

> *"XGBoost is one of the most used learning algorithms in the last few years... For many applications on structured/tabular data, decision tree ensembles including boosted trees will outperform even large neural networks."*

Course link: [coursera.org/specializations/machine-learning-introduction](https://www.coursera.org/specializations/machine-learning-introduction)

---

### 📚 scikit-learn Documentation

| Page | URL |
|------|-----|
| `GradientBoostingClassifier` | [sklearn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html) |
| `GradientBoostingRegressor` | [sklearn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html) |
| `HistGradientBoostingClassifier` | [sklearn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html) |
| Ensemble methods guide | [sklearn.org/stable/modules/ensemble.html#gradient-tree-boosting](https://scikit-learn.org/stable/modules/ensemble.html#gradient-tree-boosting) |
| `permutation_importance` | [sklearn.org/stable/modules/permutation_importance.html](https://scikit-learn.org/stable/modules/permutation_importance.html) |
| `PartialDependenceDisplay` | [sklearn.org/stable/modules/partial_dependence.html](https://scikit-learn.org/stable/modules/partial_dependence.html) |

---

### 📄 Foundational Papers

| Paper | Authors | Year | Link |
|-------|---------|------|------|
| Greedy Function Approximation: A Gradient Boosting Machine | J. Friedman | 2001 | [statweb.stanford.edu/~jhf/ftp/trebst.pdf](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf) |
| Stochastic Gradient Boosting | J. Friedman | 2002 | [statweb.stanford.edu/~jhf/ftp/stobst.pdf](https://statweb.stanford.edu/~jhf/ftp/stobst.pdf) |
| XGBoost: A Scalable Tree Boosting System | T. Chen, C. Guestrin | 2016 | [arxiv.org/abs/1603.02754](https://arxiv.org/abs/1603.02754) |
| LightGBM: A Highly Efficient GBDT | Ke et al. | 2017 | [papers.nips.cc/paper/6907](https://papers.nips.cc/paper/6907) |
| CatBoost: unbiased boosting with categorical features | Prokhorenkova et al. | 2018 | [arxiv.org/abs/1706.09516](https://arxiv.org/abs/1706.09516) |

---

## 16. Quick Reference Card

```
════════════════════════════════════════════════════════════
 GRADIENT BOOSTING — COMPLETE QUICK REFERENCE
════════════════════════════════════════════════════════════

 ALGORITHM:
   F₀(x) = mean(y)
   For m = 1 to M:
     rᵢ  = yᵢ − F_{m-1}(xᵢ)     [pseudo-residuals, MSE case]
     hₘ  = shallow tree fit to (X, r)
     Fₘ  = F_{m-1} + η · hₘ
   Output: Fₘ(x)

 KEY FORMULA:
   rᵢₘ = − ∂L(yᵢ, F(xᵢ)) / ∂F(xᵢ)   ← negative gradient

 HYPERPARAMETER PRIORITY ORDER:
   1. n_estimators + learning_rate  (always tune together)
   2. max_depth                     (2–5, usually 3)
   3. subsample                     (0.7–0.9 for stochastic GB)
   4. max_features                  ('sqrt' or 0.5–0.8)
   5. min_samples_leaf              (1–20 for smoothing)
   6. ccp_alpha                     (0.0–0.1 for pruning)

 FAST RECIPE FOR ANY DATASET:
   Step 1: Start with learning_rate=0.1, n_estimators=300
   Step 2: Use staged_predict to find optimal n_estimators
   Step 3: Switch to learning_rate=0.05, 2× n_estimators
   Step 4: Add subsample=0.8, max_features='sqrt'
   Step 5: Tune max_depth (try 2,3,4,5)
   Step 6: Use early stopping for final model

 LOSS FUNCTIONS:
   Regression  : 'squared_error', 'absolute_error', 'huber'
   Binary      : 'log_loss'  →  F → sigmoid → p
   Multiclass  : 'log_loss'  →  F → softmax → p_k

 WHEN TO USE WHICH LIBRARY:
   <10k rows    → sklearn GradientBoosting  (educational, no installs)
   Any size     → XGBoost  (industry standard, GPU, L1/L2)
   Large data   → LightGBM  (fastest, leaf-wise, low memory)
   Categoricals → CatBoost  (native encoding, ordered boosting)
   sklearn only → HistGradientBoosting  (fast, handles NaN)

 INTERVIEW ESSENTIALS:
   • Pseudo-residuals = negative gradient of loss = y−F for MSE
   • Gradient descent in FUNCTION space, not parameter space
   • Shallow trees → weak learners → boosting reduces bias
   • Learning rate shrinks each correction → regularization
   • XGBoost uses 2nd order Taylor (hessian) for better leaf values
   • LightGBM uses leaf-wise growth → faster but needs care with num_leaves
════════════════════════════════════════════════════════════
```

---

*Compiled from: Hands-On ML (Géron, 3rd ed.), ISLP (James et al.), ESL (Hastie et al.), StatQuest (Josh Starmer), Andrew Ng ML Specialization, scikit-learn documentation, and Friedman (2001, 2002).*
