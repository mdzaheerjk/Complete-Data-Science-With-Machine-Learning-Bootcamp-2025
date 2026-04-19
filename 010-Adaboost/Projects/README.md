# 🚀 AdaBoost — Complete Job-Ready Notes
> **Sources:** Andrew Ng ML Specialization · Hands-On ML (Aurélien Géron) · StatQuest (Josh Starmer) · Scikit-Learn Docs · ISLP (James et al.)

---

## 📚 Table of Contents
1. [What is AdaBoost?](#1-what-is-adaboost)
2. [Big Picture — How It Works](#2-big-picture--how-it-works)
3. [Weak Learners — Decision Stumps](#3-weak-learners--decision-stumps)
4. [The Core Intuition — Sample Weighting](#4-the-core-intuition--sample-weighting)
5. [AdaBoost Algorithm — Step by Step](#5-adaboost-algorithm--step-by-step)
6. [The Full Math — AdaBoost.M1](#6-the-full-math--adaboostm1)
7. [AdaBoost as Gradient Descent in Function Space](#7-adaboost-as-gradient-descent-in-function-space)
8. [Exponential Loss — The Loss Function](#8-exponential-loss--the-loss-function)
9. [AdaBoost for Regression (AdaBoost.R2)](#9-adaboost-for-regression-adaboostr2)
10. [Bias-Variance Tradeoff in AdaBoost](#10-bias-variance-tradeoff-in-adaboost)
11. [AdaBoost vs Random Forest vs XGBoost vs GBM](#11-adaboost-vs-random-forest-vs-xgboost-vs-gbm)
12. [Hyperparameters — Complete Guide](#12-hyperparameters--complete-guide)
13. [Python Code — From Scratch to Sklearn](#13-python-code--from-scratch-to-sklearn)
14. [AdaBoost From Scratch (Pure Python)](#14-adaboost-from-scratch-pure-python)
15. [Feature Importance](#15-feature-importance)
16. [Hyperparameter Tuning](#16-hyperparameter-tuning)
17. [Cross-Validation](#17-cross-validation)
18. [Handling Imbalanced Data](#18-handling-imbalanced-data)
19. [Tuning Strategy (Interview-Ready)](#19-tuning-strategy-interview-ready)
20. [Common Interview Questions](#20-common-interview-questions)
21. [Resources](#21-resources)

---

## 1. What is AdaBoost?

**Simple Statement:**
AdaBoost (Adaptive Boosting) is an ensemble method that combines many **weak learners** (slightly better than random) into a single **strong learner**. It does this by training each new learner on a **reweighted dataset** — misclassified samples get higher weights so the next learner focuses more on them.

**Key Ideas:**
- Invented by Freund & Schapire (1996) — won the **Gödel Prize** in 2003
- Uses **decision stumps** (1-level trees) as default weak learners
- Sequentially trains learners, each correcting the previous ones' mistakes
- Combines learners via a **weighted majority vote**
- More accurate learners get **higher voting power**
- Sensitive to **noisy data and outliers** (it keeps increasing their weights)

> 📖 *ISLP, Ch. 8.2.3*: "AdaBoost fits a sequence of weak learners on modified versions of the data, producing a sequence of weak classifiers. The predictions from all are combined through a weighted majority vote."

> 🎬 *StatQuest*: ["AdaBoost, Clearly Explained"](https://www.youtube.com/watch?v=LsK-xG1cLYA)

> 📖 *Hands-On ML, Ch. 7*: "AdaBoost is similar to a committee of experts where each expert specializes in cases that others find difficult."

---

## 2. Big Picture — How It Works

```
Training Data: (x₁,y₁), ..., (xₙ,yₙ)   y ∈ {-1, +1}
Initialize: all sample weights wᵢ = 1/n   (uniform)

────────────────────────────────────────────────────────
Round 1:
  Train weak learner h₁ on weighted data
  Compute weighted error ε₁
  Compute learner weight α₁ (higher if error is low)
  Update sample weights:
     Misclassified → weights INCREASE
     Correctly classified → weights DECREASE
────────────────────────────────────────────────────────
Round 2:
  Train weak learner h₂ on NEW weighted data
  h₂ focuses MORE on what h₁ got wrong
  Compute ε₂, α₂, update weights again
────────────────────────────────────────────────────────
...repeat for T rounds...
────────────────────────────────────────────────────────

Final Prediction:
  H(x) = sign(α₁h₁(x) + α₂h₂(x) + ... + αₜhₜ(x))
  
  = sign( Σₜ αₜ hₜ(x) )
```

**Analogy:**
Imagine you're a teacher with a class of students. You give an exam. The students who got wrong answers must study harder next time (higher weight). You ask a new tutor to focus specifically on those students. Repeat until all students pass. The final grade is a weighted vote of all tutors' assessments — better tutors (lower error) have more say.

---

## 3. Weak Learners — Decision Stumps

AdaBoost uses **decision stumps** by default: decision trees with `max_depth=1`.

### What is a Decision Stump?

```
         Feature X ≤ threshold?
              /          \
           YES             NO
           /                 \
    Predict class A      Predict class B
```

- One split, two leaves
- Slightly better than random guessing (accuracy > 50%)
- Each stump focuses on **one feature** and **one threshold**
- Individually weak, but powerful when combined

### Why Stumps?

- Simple enough to be fast
- Diverse enough across rounds (different features/thresholds chosen)
- Low complexity = low variance = safe to boost
- Theoretical guarantee: any weak learner (>50% accuracy) can be boosted to arbitrary accuracy

> 🎬 *StatQuest*: "A single stump is barely better than a coin flip, but 500 of them together can be incredibly powerful."

---

## 4. The Core Intuition — Sample Weighting

### Iteration 1: All weights equal

```
Data: ● ● ● ● ○ ○ ○ ○   (● = class +1, ○ = class -1)
Weights: all = 1/8

Stump h₁: splits on feature X ≤ 3
  → Correctly classifies: ● ● ● ○ ○ ○
  → Misclassifies: ● ● (two positives predicted as negative)
```

### After Weight Update

```
Misclassified samples (● ●) → weights INCREASE  📈
Correctly classified → weights DECREASE          📉

New data distribution (visually):
● ● ● ● ● ● ○ ○ ○ ○   (misclassified ● represented more)
```

### Iteration 2: Focuses on hard cases

```
Stump h₂ now trained on reweighted data
  → It will try hard to correctly classify the previously misclassified ●
  → It may now misclassify some easy ○ samples
  → Weights update again...
```

This is **adaptive** boosting — the model adapts its attention based on where it's failing.

---

## 5. AdaBoost Algorithm — Step by Step

### Input
- Training data: $\{(x_1, y_1), ..., (x_n, y_n)\}$ where $y_i \in \{-1, +1\}$
- Number of rounds: $T$
- Weak learner: $\mathcal{H}$ (e.g., decision stump)

### Algorithm

```
Step 1: Initialize sample weights
        wᵢ⁽¹⁾ = 1/n    for i = 1, ..., n

Step 2: For t = 1 to T:

    a) Train weak learner hₜ on data weighted by w⁽ᵗ⁾
       hₜ: X → {-1, +1}

    b) Compute weighted error:
                    Σᵢ wᵢ⁽ᵗ⁾ · 1[hₜ(xᵢ) ≠ yᵢ]
       εₜ = ─────────────────────────────────────
                         Σᵢ wᵢ⁽ᵗ⁾

    c) Compute learner weight:
                 1     1 - εₜ
       αₜ = ─── · ln(────────)
                 2       εₜ

    d) Update sample weights:
       wᵢ⁽ᵗ⁺¹⁾ = wᵢ⁽ᵗ⁾ · exp(-αₜ · yᵢ · hₜ(xᵢ))

    e) Normalize weights:
                    wᵢ⁽ᵗ⁺¹⁾
       wᵢ⁽ᵗ⁺¹⁾ ← ──────────────
                   Σⱼ wⱼ⁽ᵗ⁺¹⁾

Step 3: Final prediction:
        H(x) = sign( Σₜ αₜ · hₜ(x) )
```

---

## 6. The Full Math — AdaBoost.M1

### Weighted Error Rate

$$\varepsilon_t = \frac{\sum_{i=1}^{n} w_i^{(t)} \cdot \mathbf{1}[h_t(x_i) \neq y_i]}{\sum_{i=1}^{n} w_i^{(t)}}$$

- $\varepsilon_t \in [0, 1]$
- $\varepsilon_t < 0.5$ required (weak learner condition)
- $\varepsilon_t = 0$: perfect learner
- $\varepsilon_t = 0.5$: random guessing

### Learner Weight (Amount of Say)

$$\alpha_t = \frac{1}{2} \ln\left(\frac{1 - \varepsilon_t}{\varepsilon_t}\right)$$

| $\varepsilon_t$ | $\alpha_t$ | Meaning |
|---|---|---|
| 0.01 | 2.30 | Very good → high authority |
| 0.1 | 1.10 | Good → high authority |
| 0.3 | 0.42 | OK → moderate authority |
| 0.4 | 0.20 | Poor → low authority |
| 0.5 | 0.0 | Random → zero authority |
| >0.5 | Negative | Worse than random → negative authority (flip predictions) |

**Simple Statement:** A learner that's right 90% of the time gets ~5× more voting power than one that's right 60% of the time.

### Weight Update Rule

$$w_i^{(t+1)} = w_i^{(t)} \cdot \exp\left(-\alpha_t \cdot y_i \cdot h_t(x_i)\right)$$

**Case Analysis** (where $y_i \in \{-1, +1\}$ and $h_t(x_i) \in \{-1, +1\}$):

| $y_i \cdot h_t(x_i)$ | Result | Weight change |
|---|---|---|
| $+1$ (correct) | $\exp(-\alpha_t)$ | Decreases (since $\alpha_t > 0$) |
| $-1$ (wrong) | $\exp(+\alpha_t)$ | Increases |

**Normalization** (to keep weights as a distribution):

$$w_i^{(t+1)} \leftarrow \frac{w_i^{(t+1)}}{\sum_{j=1}^{n} w_j^{(t+1)}}$$

### Normalization Constant

$$Z_t = \sum_{i=1}^n w_i^{(t)} \cdot \exp(-\alpha_t y_i h_t(x_i))$$

$$= 2\sqrt{\varepsilon_t(1-\varepsilon_t)}$$

This is minimized when $\alpha_t = \frac{1}{2}\ln\frac{1-\varepsilon_t}{\varepsilon_t}$, connecting to the exponential loss minimization.

### Final Strong Classifier

$$H(x) = \text{sign}\left(\sum_{t=1}^{T} \alpha_t h_t(x)\right)$$

$$= \text{sign}(F(x))$$

Where $F(x) = \sum_t \alpha_t h_t(x)$ is the **margin** or **confidence score**.

### Training Error Bound

$$\text{Training Error}(H) \leq \prod_{t=1}^{T} 2\sqrt{\varepsilon_t(1-\varepsilon_t)} = \prod_{t=1}^{T} Z_t$$

If each weak learner has error $\varepsilon_t = 0.5 - \gamma_t$ (edge $\gamma_t > 0$):

$$\text{Training Error}(H) \leq \exp\left(-2\sum_{t=1}^{T} \gamma_t^2\right)$$

**Key Result:** Training error **decreases exponentially** with the number of rounds! This is the theoretical guarantee of AdaBoost.

> 📖 *ISLP, Ch. 8.2.3*: "This bound shows that, as long as each weak classifier is slightly better than chance, the training error drops to zero exponentially fast."

---

## 7. AdaBoost as Gradient Descent in Function Space

### AdaBoost Minimizes Exponential Loss

AdaBoost is equivalent to **Forward Stagewise Additive Modeling** that minimizes the exponential loss:

$$L(y, F(x)) = e^{-y F(x)}$$

At each step $t$, we solve:

$$(\alpha_t, h_t) = \arg\min_{\alpha, h} \sum_{i=1}^{n} \exp\left(-y_i \left(F_{t-1}(x_i) + \alpha h(x_i)\right)\right)$$

$$= \arg\min_{\alpha, h} \sum_{i=1}^{n} w_i^{(t)} \exp\left(-\alpha y_i h(x_i)\right)$$

Where $w_i^{(t)} = \exp(-y_i F_{t-1}(x_i))$ are the **current sample weights**.

### Gradient of Exponential Loss

$$\frac{\partial L}{\partial F(x_i)} = -y_i \cdot e^{-y_i F(x_i)}$$

The negative gradient (pseudo-residual direction) is:

$$-\frac{\partial L}{\partial F(x_i)} = y_i \cdot e^{-y_i F(x_i)} = y_i \cdot w_i$$

**Interpretation:** Samples with large $w_i$ (misclassified ones) have the largest pseudo-residuals — training on weighted data IS gradient descent!

### Comparison: AdaBoost vs Gradient Boosting

| | AdaBoost | Gradient Boosting |
|---|---|---|
| **Loss function** | Exponential: $e^{-yF}$ | Any differentiable loss |
| **Mechanism** | Reweight samples | Fit pseudo-residuals |
| **Equivalent to** | Functional gradient descent on exp loss | Functional gradient descent on chosen loss |
| **Sensitivity to outliers** | Very high (exp loss blows up) | Moderate (depends on loss choice) |

> 📖 *Hands-On ML, Ch. 7*: "AdaBoost is a special case of gradient boosting using the exponential loss function and decision stumps."

---

## 8. Exponential Loss — The Loss Function

$$L_{\text{exp}}(y, F(x)) = e^{-y \cdot F(x)}$$

Where $y \in \{-1, +1\}$ and $F(x) = \sum_t \alpha_t h_t(x)$.

### Properties

| Condition | $yF(x)$ | Loss |
|---|---|---|
| Correct, confident | Large positive | $e^{-\text{large}} \approx 0$ (low loss) |
| Correct, uncertain | Near 0 | $e^0 = 1$ |
| Wrong, uncertain | Near 0 | $e^0 = 1$ |
| Wrong, confident | Large negative | $e^{\text{large}} \gg 1$ (huge loss!) |

### Why Exponential Loss is Problematic

$$\frac{d L}{d F} = -y \cdot e^{-yF} \implies \text{gradient grows exponentially with misclassification confidence}$$

- Outliers get **exponentially large weights** → model spends all effort on them
- This makes AdaBoost very sensitive to **label noise**
- Compare: logistic loss grows linearly with margin → much more robust

### Loss Comparison

```
Loss │
     │    Exponential ╮
     │               ╱╲
     │              ╱  ╲___
     │  0-1 Loss ___      \
     │  Logistic __╮       \___
     │             ╰────────────
     └────────────────────────── yF(x)
        wrong            correct
```

---

## 9. AdaBoost for Regression (AdaBoost.R2)

For regression problems, sklearn uses **AdaBoost.R2** algorithm:

### Algorithm

```
Initialize: wᵢ = 1/n

For t = 1 to T:
    1. Train regression tree hₜ on weighted data
    2. Compute predictions hₜ(xᵢ)
    3. Compute maximum absolute error:
       D_max = max|yᵢ - hₜ(xᵢ)|

    4. Compute loss for each sample:
       Linear:   lossᵢ = |yᵢ - hₜ(xᵢ)| / D_max
       Square:   lossᵢ = (|yᵢ - hₜ(xᵢ)| / D_max)²
       Exp:      lossᵢ = 1 - exp(-|yᵢ - hₜ(xᵢ)| / D_max)

    5. Weighted average loss:
       ε_t = Σᵢ wᵢ · lossᵢ

    6. Learner weight:
       α_t = ε_t / (1 - ε_t)    ← note: different from classification!

    7. Update weights:
       wᵢ ← wᵢ · α_t^(1 - lossᵢ)    (low loss → weight decreases)

    8. Normalize weights

Final Prediction:
    Weighted median of {hₜ(x)} with weights {log(1/αₜ)}
```

---

## 10. Bias-Variance Tradeoff in AdaBoost

### AdaBoost Primarily Reduces Bias

Unlike Random Forest (reduces variance), AdaBoost primarily **reduces bias** by sequentially correcting errors.

```
Single Stump:
    High Bias ─────────────────── Low Variance
    (too simple to fit data)

AdaBoost after T rounds:
    Low Bias ──────────────────── Low-Medium Variance
    (sequentially reduces bias)
```

### Effect of Number of Rounds (n_estimators)

```
            Training Error     Test Error
T = 10      0.20               0.22       ← High bias (underfitting)
T = 50      0.10               0.11       ← Good
T = 100     0.05               0.08       ← Good
T = 200     0.02               0.08       ← Minimal improvement
T = 500     0.01               0.09       ← Slight overfitting (noise)
T = 1000    0.00               0.12       ← Overfitting (noisy data)
```

### AdaBoost and Overfitting — The Curious Property

**Surprising:** AdaBoost often does NOT overfit even with many rounds (on clean data). This is because:

1. The **margin** $y_i F(x_i)$ continues to grow even after training error hits 0
2. Larger margins → better generalization (similar to SVM theory)
3. **VC dimension theory** explains why: growing margins increase effective complexity slowly

**But:** On **noisy data**, AdaBoost DOES overfit because outliers get exponentially large weights.

> 📖 *ISLP, Ch. 8.2.3*: "Unlike other boosting methods, AdaBoost is relatively resistant to overfitting on clean data, though this property breaks down with label noise."

### Hyperparameter Effect on Bias-Variance

| Parameter | Increase → | Bias | Variance |
|---|---|---|---|
| `n_estimators` | More rounds | Decreases | May increase |
| `learning_rate` | Higher | Decreases faster | Increases |
| `max_depth` (base) | Deeper trees | Decreases | Increases |
| `min_samples_leaf` (base) | Larger | Increases | Decreases |

---

## 11. AdaBoost vs Random Forest vs XGBoost vs GBM

| Feature | AdaBoost | Random Forest | Gradient Boosting | XGBoost |
|---|---|---|---|---|
| **Building strategy** | Sequential | Parallel | Sequential | Sequential |
| **Error correction** | Sample reweighting | None (independent) | Pseudo-residuals | 2nd-order Taylor |
| **Loss function** | Exponential (fixed) | N/A | Any differentiable | Any differentiable |
| **Default base learner** | Stumps (depth=1) | Full trees | Shallow trees (depth=3-5) | Shallow trees (depth=6) |
| **Reduces** | Bias | Variance | Bias | Bias + regularized |
| **Overfitting risk** | Low (clean data) | Low | Moderate | Moderate (but regularized) |
| **Outlier sensitivity** | Very High | Low | Moderate | Moderate |
| **Speed** | Fast (stumps) | Fast (parallel) | Slow | Fast (parallel splits) |
| **Regularization** | None built-in | Implicit (avg) | None (sklearn adds some) | L1 + L2 built-in |
| **Interpretability** | Medium | Low | Low | Low |
| **Hyperparameters** | Few | Medium | Many | Many |
| **Missing values** | Manual imputation | Manual imputation | Manual imputation | Native |
| **Feature scaling** | Not needed | Not needed | Not needed | Not needed |

---

## 12. Hyperparameters — Complete Guide

### 🔧 Core AdaBoost Parameters

| Parameter | Default | Range | Effect |
|---|---|---|---|
| `n_estimators` | 50 | 50–2000 | Number of weak learners; more = lower bias, risk of overfit on noisy data |
| `learning_rate` | 1.0 | 0.01–1.0 | Shrinks each learner's contribution; lower = need more estimators |
| `algorithm` | `'SAMME.R'` | `'SAMME'`, `'SAMME.R'` | SAMME.R uses probabilities (better); SAMME uses discrete predictions |
| `random_state` | None | int | For reproducibility |

### 🌳 Base Estimator Parameters

| Parameter | Default | Notes |
|---|---|---|
| `estimator` | `DecisionTreeClassifier(max_depth=1)` | Can use any sklearn estimator |
| `max_depth` (of base tree) | 1 (stump) | Increase for more complex base learners |
| `min_samples_leaf` (base) | 1 | Control leaf size of base tree |

### 🎯 SAMME vs SAMME.R

| | SAMME | SAMME.R |
|---|---|---|
| **Full name** | Stagewise Additive Modeling using Multi-class Exponential loss | SAMME with Real-valued predictions |
| **Uses** | Class labels $h_t(x) \in \{0,...,K-1\}$ | Class probabilities $p_{tk}(x)$ |
| **Update** | Discrete: $\alpha_t = \log\frac{1-\varepsilon_t}{\varepsilon_t} + \log(K-1)$ | Real: $\alpha_{tk} = (K-1)[\log p_{tk} - \frac{1}{K}\sum_j \log p_{tj}]$ |
| **Convergence** | Slower | Faster |
| **Sklearn default** | No | Yes (`SAMME.R`) |

### 🎯 Tuning Priority

```
1. n_estimators — use validation curve / cross-val
2. learning_rate — inversely related to n_estimators
3. base_estimator depth — max_depth=1 (default), try 2-3
4. min_samples_leaf — reduce overfitting with noisy data
```

---

## 13. Python Code — From Scratch to Sklearn

### Installation

```bash
pip install scikit-learn numpy pandas matplotlib seaborn shap optuna
```

### Basic Usage — Classification

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             roc_auc_score, confusion_matrix,
                             ConfusionMatrixDisplay)

# Load data
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Default AdaBoost (stumps + SAMME.R) ───────────────────────────
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),  # decision stump
    n_estimators=200,
    learning_rate=1.0,
    algorithm='SAMME',         # use SAMME for sklearn >= 1.4
    random_state=42
)

ada.fit(X_train, y_train)

y_pred = ada.predict(X_test)
y_prob = ada.predict_proba(X_test)[:, 1]

print(f"Test Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC-ROC       : {roc_auc_score(y_test, y_prob):.4f}")
print()
print(classification_report(y_test, y_pred, target_names=data.target_names))

# ── Confusion Matrix ─────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=data.target_names)
disp.plot(cmap='Blues')
plt.title('AdaBoost — Confusion Matrix')
plt.show()
```

### Basic Usage — Regression

```python
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Data
X_r, y_r = fetch_california_housing(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X_r, y_r, test_size=0.2, random_state=42
)

# AdaBoost Regression
ada_reg = AdaBoostRegressor(
    estimator=DecisionTreeRegressor(max_depth=3),
    n_estimators=200,
    learning_rate=0.5,
    loss='linear',              # 'linear', 'square', 'exponential'
    random_state=42
)

ada_reg.fit(X_tr, y_tr)
y_pred_r = ada_reg.predict(X_te)

print(f"Test RMSE : {np.sqrt(mean_squared_error(y_te, y_pred_r)):.4f}")
print(f"Test MAE  : {mean_absolute_error(y_te, y_pred_r):.4f}")
print(f"Test R²   : {r2_score(y_te, y_pred_r):.4f}")
```

### Tracking Error Through Boosting Rounds

```python
# Staged predictions — see error evolution round by round
staged_scores = []

for i, y_staged in enumerate(ada.staged_predict(X_test)):
    staged_scores.append(accuracy_score(y_test, y_staged))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(staged_scores) + 1), staged_scores, color='steelblue')
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy')
plt.title('AdaBoost — Test Accuracy per Round')
plt.grid(True)

# Also track train accuracy
staged_train = [accuracy_score(y_train, y_s)
                for y_s in ada.staged_predict(X_train)]

plt.subplot(1, 2, 2)
plt.plot(staged_train, label='Train', color='steelblue')
plt.plot(staged_scores, label='Test', color='coral')
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy')
plt.title('Train vs Test Accuracy per Round')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

best_n = np.argmax(staged_scores) + 1
print(f"Best n_estimators: {best_n} (Test Acc: {max(staged_scores):.4f})")
```

### Visualizing Individual Stumps

```python
from sklearn.tree import plot_tree

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (tree, alpha) in enumerate(
    zip(ada.estimators_[:3], ada.estimator_weights_[:3])
):
    plot_tree(tree, ax=axes[idx],
              feature_names=feature_names,
              class_names=data.target_names,
              filled=True, rounded=True, fontsize=9)
    axes[idx].set_title(f'Stump {idx+1}\nα = {alpha:.3f}')

plt.suptitle('First 3 AdaBoost Stumps with Learner Weights', fontsize=13)
plt.tight_layout()
plt.show()
```

### Inspecting AdaBoost Components

```python
# Learner weights α_t
print("Learner weights (first 10):", ada.estimator_weights_[:10])

# Weighted errors ε_t
print("Weighted errors (first 10):", ada.estimator_errors_[:10])

# All base estimators
print(f"Number of stumps: {len(ada.estimators_)}")

# Verify: higher accuracy stumps get higher weights
errors = ada.estimator_errors_
weights = ada.estimator_weights_

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(errors, color='coral', alpha=0.7)
plt.xlabel('Round')
plt.ylabel('Weighted Error ε_t')
plt.title('Error per Round')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(weights, color='steelblue', alpha=0.7)
plt.xlabel('Round')
plt.ylabel('Learner Weight α_t')
plt.title('Alpha per Round')
plt.grid(True)
plt.tight_layout()
plt.show()
```

### AdaBoost with Deeper Base Learners

```python
# Deeper trees → lower bias, but risk of overfitting
from sklearn.model_selection import cross_val_score

depths = [1, 2, 3, 4, 5]
results = {}

for depth in depths:
    ada_d = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=depth),
        n_estimators=200,
        learning_rate=0.5,
        algorithm='SAMME',
        random_state=42
    )
    scores = cross_val_score(ada_d, X_train, y_train,
                             cv=5, scoring='roc_auc', n_jobs=-1)
    results[depth] = scores

print("Depth | Mean AUC | Std")
for d, s in results.items():
    print(f"  {d}   |  {s.mean():.4f}  | {s.std():.4f}")
```

### Pipeline with AdaBoost

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# For mixed-type datasets
numeric_features = X_df.select_dtypes(include='number').columns.tolist()
categorical_features = X_df.select_dtypes(include='object').columns.tolist()

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
    # Note: AdaBoost with trees doesn't need scaling
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('ada', AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=200,
        learning_rate=0.5,
        algorithm='SAMME',
        random_state=42
    ))
])

pipeline.fit(X_train, y_train)
print("Pipeline AUC:", roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1]))
```

---

## 14. AdaBoost From Scratch (Pure Python)

```python
import numpy as np

class DecisionStump:
    """Single-level decision tree for binary classification {-1, +1}."""

    def __init__(self):
        self.feature_idx = None
        self.threshold = None
        self.polarity = 1        # +1 or -1 (flip prediction direction)

    def predict(self, X):
        n = X.shape[0]
        preds = np.ones(n)
        col = X[:, self.feature_idx]

        if self.polarity == 1:
            preds[col < self.threshold] = -1
        else:
            preds[col >= self.threshold] = -1

        return preds


class AdaBoostScratch:
    """AdaBoost.M1 from scratch for binary classification."""

    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.alphas = []
        self.stumps = []

    def fit(self, X, y):
        """
        X: shape (n_samples, n_features)
        y: shape (n_samples,) with values in {-1, +1}
        """
        n, p = X.shape
        # Step 1: Initialize uniform weights
        w = np.ones(n) / n

        for t in range(self.n_estimators):
            stump = DecisionStump()
            min_error = float('inf')

            # Find best stump: iterate over features and thresholds
            for j in range(p):
                thresholds = np.unique(X[:, j])
                for threshold in thresholds:
                    for polarity in [1, -1]:
                        stump_t = DecisionStump()
                        stump_t.feature_idx = j
                        stump_t.threshold = threshold
                        stump_t.polarity = polarity

                        preds = stump_t.predict(X)

                        # Step 2b: Weighted error
                        wrong = (preds != y).astype(float)
                        error = np.dot(w, wrong)   # Σ wᵢ · 1[h(xᵢ)≠yᵢ]

                        if error < min_error:
                            min_error = error
                            stump = stump_t

            # Clip to avoid log(0)
            min_error = np.clip(min_error, 1e-10, 1 - 1e-10)

            # Step 2c: Learner weight α_t
            alpha = 0.5 * np.log((1 - min_error) / min_error)

            # Get stump predictions
            preds = stump.predict(X)

            # Step 2d: Update weights
            # w_i ← w_i · exp(-α_t · y_i · h_t(x_i))
            w = w * np.exp(-alpha * y * preds)

            # Step 2e: Normalize
            w /= w.sum()

            self.stumps.append(stump)
            self.alphas.append(alpha)

        return self

    def predict(self, X):
        """Final prediction: sign(Σ α_t · h_t(x))"""
        # Shape: (n_estimators, n_samples)
        stump_preds = np.array([s.predict(X) for s in self.stumps])
        # Weighted sum
        F = np.dot(np.array(self.alphas), stump_preds)
        return np.sign(F)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


# ── Test it ─────────────────────────────────────────────────────
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X_sc, y_sc = make_classification(n_samples=500, n_features=10,
                                  random_state=42)
y_sc = np.where(y_sc == 0, -1, 1)   # convert to {-1, +1}

X_tr_sc, X_te_sc, y_tr_sc, y_te_sc = train_test_split(
    X_sc, y_sc, test_size=0.2, random_state=42
)

ada_scratch = AdaBoostScratch(n_estimators=100)
ada_scratch.fit(X_tr_sc, y_tr_sc)

print(f"Scratch AdaBoost Accuracy: {ada_scratch.score(X_te_sc, y_te_sc):.4f}")

# Compare with sklearn
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

ada_sk = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=100, algorithm='SAMME', random_state=42
)
y_sk = np.where(y_sc == -1, 0, 1)   # sklearn needs {0, 1}
X_tr_sk, X_te_sk, y_tr_sk, y_te_sk = train_test_split(
    X_sc, y_sk, test_size=0.2, random_state=42
)
ada_sk.fit(X_tr_sk, y_tr_sk)
print(f"Sklearn AdaBoost Accuracy: {ada_sk.score(X_te_sk, y_te_sk):.4f}")
```

---

## 15. Feature Importance

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

# ── MDI Feature Importance (accumulated over all stumps) ─────────
importances = ada.feature_importances_
feat_imp_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False).head(15)

plt.figure(figsize=(10, 6))
plt.barh(feat_imp_df['feature'], feat_imp_df['importance'],
         color='steelblue', alpha=0.8)
plt.xlabel('Feature Importance (MDI)')
plt.title('AdaBoost — Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# ── Permutation Importance (more reliable) ───────────────────────
perm = permutation_importance(
    ada, X_test, y_test,
    n_repeats=30,
    random_state=42,
    n_jobs=-1
)

perm_df = pd.DataFrame({
    'feature': feature_names,
    'importance_mean': perm.importances_mean,
    'importance_std': perm.importances_std
}).sort_values('importance_mean', ascending=False).head(15)

plt.figure(figsize=(10, 6))
plt.barh(perm_df['feature'], perm_df['importance_mean'],
         xerr=perm_df['importance_std'],
         color='coral', alpha=0.8)
plt.xlabel('Mean Accuracy Decrease')
plt.title('AdaBoost — Permutation Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# ── SHAP Values ──────────────────────────────────────────────────
import shap

explainer = shap.TreeExplainer(ada)
shap_values = explainer.shap_values(X_test)

sv = shap_values[1] if isinstance(shap_values, list) else shap_values
shap.summary_plot(sv, X_test, feature_names=feature_names)
shap.summary_plot(sv, X_test, plot_type='bar', feature_names=feature_names)
```

---

## 16. Hyperparameter Tuning

### Validation Curve for n_estimators

```python
from sklearn.model_selection import validation_curve

n_range = [10, 25, 50, 100, 150, 200, 300, 400, 500]

train_sc, val_sc = validation_curve(
    AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        algorithm='SAMME', random_state=42
    ),
    X_train, y_train,
    param_name='n_estimators',
    param_range=n_range,
    scoring='roc_auc',
    cv=5, n_jobs=-1
)

plt.figure(figsize=(10, 5))
plt.plot(n_range, np.mean(train_sc, axis=1), label='Train AUC')
plt.plot(n_range, np.mean(val_sc, axis=1), label='CV AUC')
plt.xlabel('n_estimators')
plt.ylabel('AUC-ROC')
plt.legend()
plt.title('Validation Curve — n_estimators')
plt.grid(True)
plt.show()
```

### GridSearchCV

```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold

param_grid = {
    'n_estimators': [50, 100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.5, 1.0],
    'estimator__max_depth': [1, 2, 3],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

base = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(),
    algorithm='SAMME',
    random_state=42
)

grid = GridSearchCV(
    base, param_grid,
    scoring='roc_auc',
    cv=cv,
    verbose=2,
    n_jobs=-1
)

grid.fit(X_train, y_train)
print("Best Params:", grid.best_params_)
print("Best AUC:   ", grid.best_score_)
```

### RandomizedSearchCV

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, loguniform

param_dist = {
    'n_estimators': randint(50, 1000),
    'learning_rate': loguniform(0.01, 1.0),
    'estimator__max_depth': [1, 2, 3, 4],
    'estimator__min_samples_leaf': randint(1, 10),
}

random_search = RandomizedSearchCV(
    base, param_distributions=param_dist,
    n_iter=100,
    scoring='roc_auc',
    cv=cv,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)
print("Best Params:", random_search.best_params_)
print("Best AUC:   ", random_search.best_score_)
```

### Optuna

```python
import optuna
from sklearn.model_selection import cross_val_score

def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 1000)
    learning_rate = trial.suggest_float('learning_rate', 1e-3, 1.0, log=True)
    max_depth = trial.suggest_int('max_depth', 1, 5)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)

    model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf
        ),
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        algorithm='SAMME',
        random_state=42
    )

    scores = cross_val_score(model, X_train, y_train,
                             cv=5, scoring='roc_auc', n_jobs=-1)
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, show_progress_bar=True)

print("Best params:", study.best_trial.params)
print("Best AUC:   ", study.best_value)

# Best model
bp = study.best_trial.params
best_ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(
        max_depth=bp['max_depth'],
        min_samples_leaf=bp['min_samples_leaf']
    ),
    n_estimators=bp['n_estimators'],
    learning_rate=bp['learning_rate'],
    algorithm='SAMME',
    random_state=42
)
best_ada.fit(X_train, y_train)
```

---

## 17. Cross-Validation

```python
from sklearn.model_selection import (cross_val_score, cross_validate,
                                     StratifiedKFold, learning_curve)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Single metric
auc_scores = cross_val_score(
    ada, X, y, cv=cv, scoring='roc_auc', n_jobs=-1
)
print(f"AUC: {auc_scores.mean():.4f} ± {auc_scores.std():.4f}")

# Multiple metrics
cv_results = cross_validate(
    ada, X, y,
    cv=cv,
    scoring=['roc_auc', 'accuracy', 'f1', 'precision', 'recall'],
    return_train_score=True,
    n_jobs=-1
)

for metric in ['test_roc_auc', 'test_accuracy', 'test_f1']:
    m = cv_results[metric]
    print(f"{metric:22s}: {m.mean():.4f} ± {m.std():.4f}")

# Learning Curve
train_sizes, train_sc, val_sc = learning_curve(
    ada, X, y,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

plt.figure(figsize=(10, 5))
plt.plot(train_sizes, np.mean(train_sc, axis=1), label='Train', color='steelblue')
plt.plot(train_sizes, np.mean(val_sc, axis=1), label='Validation', color='coral')
plt.fill_between(train_sizes,
                 np.mean(train_sc, axis=1) - np.std(train_sc, axis=1),
                 np.mean(train_sc, axis=1) + np.std(train_sc, axis=1), alpha=0.2)
plt.fill_between(train_sizes,
                 np.mean(val_sc, axis=1) - np.std(val_sc, axis=1),
                 np.mean(val_sc, axis=1) + np.std(val_sc, axis=1), alpha=0.2)
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.title('Learning Curve — AdaBoost')
plt.legend()
plt.grid(True)
plt.show()
```

---

## 18. Handling Imbalanced Data

```python
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import average_precision_score, f1_score

# Method 1: sample_weight in fit()
# AdaBoost accepts initial sample weights directly
sample_weights = compute_sample_weight('balanced', y=y_train)

ada_balanced = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=200,
    algorithm='SAMME',
    random_state=42
)
ada_balanced.fit(X_train, y_train, sample_weight=sample_weights)

y_prob_bal = ada_balanced.predict_proba(X_test)[:, 1]
print(f"PR-AUC (balanced): {average_precision_score(y_test, y_prob_bal):.4f}")
print(f"F1     (balanced): {f1_score(y_test, ada_balanced.predict(X_test)):.4f}")

# Method 2: SMOTE + AdaBoost
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

pipeline_smote = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('ada', AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=200,
        algorithm='SAMME',
        random_state=42
    ))
])
pipeline_smote.fit(X_train, y_train)

# Method 3: class_weight in base estimator
ada_cw = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, class_weight='balanced'),
    n_estimators=200,
    algorithm='SAMME',
    random_state=42
)
ada_cw.fit(X_train, y_train)

# For imbalanced: prefer PR-AUC over ROC-AUC
print(f"PR-AUC (class_weight): {average_precision_score(y_test, ada_cw.predict_proba(X_test)[:, 1]):.4f}")
```

---

## 19. Tuning Strategy (Interview-Ready)

### Complete Step-by-Step Guide

```
STEP 1: Baseline
─────────────────
AdaBoostClassifier(n_estimators=50, learning_rate=1.0,
                   estimator=DecisionTreeClassifier(max_depth=1))
Goal: Establish baseline; check for data quality issues

STEP 2: Find optimal n_estimators
──────────────────────────────────
Use staged_predict() or validation curve
Plot train vs test accuracy per round
Find elbow / best test accuracy
Typical range: 100–500

STEP 3: Tune learning_rate
──────────────────────────
Lower learning_rate + more n_estimators = more robust
Try: [0.01, 0.05, 0.1, 0.5, 1.0]
Keep n_estimators fixed (large), tune learning_rate

Step 4: Tune base estimator depth
──────────────────────────────────
max_depth=1 (stump): default, most common
max_depth=2 or 3: try if accuracy is low (underfitting)
Deeper base → lower bias but AdaBoost may overfit faster

STEP 5: Final evaluation
──────────────────────────
Train final model on full train set
Evaluate ONCE on held-out test set
```

### When AdaBoost Overfits (Noisy Data)

```python
# Signs: Train acc >> Test acc; error drops then rises
# Fix: Reduce complexity

ada_fix = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, min_samples_leaf=5),
    n_estimators=100,         # fewer rounds
    learning_rate=0.1,        # smaller steps
    algorithm='SAMME',
    random_state=42
)
```

### When AdaBoost Underfits

```python
# Signs: Both train and test acc are low
# Fix: Increase complexity

ada_fix = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=3),  # deeper base
    n_estimators=500,                               # more rounds
    learning_rate=0.5,
    algorithm='SAMME',
    random_state=42
)
```

### Learning Rate vs n_estimators Tradeoff

```python
# Test different combinations
configs = [
    (0.01, 1000), (0.05, 500), (0.1, 300), (0.5, 200), (1.0, 100)
]

for lr, n in configs:
    ada_t = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=n, learning_rate=lr,
        algorithm='SAMME', random_state=42
    )
    scores = cross_val_score(ada_t, X_train, y_train,
                             cv=5, scoring='roc_auc', n_jobs=-1)
    print(f"lr={lr:4.2f}, n={n:4d} → AUC: {scores.mean():.4f} ± {scores.std():.4f}")
```

---

## 20. Common Interview Questions

### Q1: How is AdaBoost different from Gradient Boosting?

**Answer:**
Both are sequential boosting algorithms but differ in mechanism:
- **AdaBoost** corrects errors by **reweighting samples** — misclassified samples get higher weights so the next learner focuses on them. It uses **exponential loss** implicitly.
- **Gradient Boosting** fits each new learner to the **pseudo-residuals** (negative gradient of the loss) — generalizes to any differentiable loss function.
- Mathematically, AdaBoost is a **special case of Gradient Boosting** with exponential loss and additive stumps.
- AdaBoost is more sensitive to outliers because the exponential loss blows up for confident misclassifications.

### Q2: What is the learner weight α_t and how is it derived?

**Answer:**

$$\alpha_t = \frac{1}{2}\ln\left(\frac{1-\varepsilon_t}{\varepsilon_t}\right)$$

It's derived by minimizing the exponential loss w.r.t. $\alpha$ analytically.

Properties:
- $\varepsilon_t = 0.5$ (random) → $\alpha_t = 0$ (no contribution)
- $\varepsilon_t \to 0$ (perfect) → $\alpha_t \to \infty$ (full authority)
- $\varepsilon_t > 0.5$ → $\alpha_t < 0$ (flip predictions)

### Q3: Why does AdaBoost increase weights for misclassified samples?

**Answer:**
The weight update rule is:

$$w_i^{(t+1)} = w_i^{(t)} \cdot e^{-\alpha_t y_i h_t(x_i)}$$

- If correctly classified: $y_i h_t(x_i) = +1$, so weight multiplied by $e^{-\alpha_t} < 1$ → decreases
- If misclassified: $y_i h_t(x_i) = -1$, so weight multiplied by $e^{+\alpha_t} > 1$ → increases

This forces each subsequent learner to focus on the **hard samples** that previous learners failed on. It's the adaptive mechanism that makes boosting work.

### Q4: Can AdaBoost overfit?

**Answer:**
Yes, but it's more nuanced than other methods:
- On **clean data**: AdaBoost is surprisingly resistant to overfitting. Even after training error hits 0, generalization keeps improving because the **margin** keeps growing.
- On **noisy data** (label noise, outliers): AdaBoost DOES overfit badly because outliers get exponentially large weights, causing the model to memorize them.
- Mitigation: reduce `n_estimators`, add `learning_rate` shrinkage, use cross-validation to find optimal rounds.

### Q5: What is SAMME vs SAMME.R?

**Answer:**
Both extend AdaBoost to **multi-class problems**:
- **SAMME** (Zhu et al. 2009): Uses discrete class predictions. Weight update adds $\log(K-1)$ term for $K$ classes. Reduces to standard AdaBoost when $K=2$.
- **SAMME.R**: Uses **probability estimates** from the base learner. Real-valued updates → faster convergence, better performance.
- **Sklearn default** was SAMME.R but from sklearn 1.4+ SAMME is recommended as SAMME.R is deprecated.

### Q6: Why is AdaBoost sensitive to outliers?

**Answer:**
The **exponential loss** $L = e^{-yF(x)}$ grows exponentially with confident misclassifications. An outlier with a wrong label will be consistently misclassified → its weight increases exponentially each round → the entire model tries to fit this one noisy point.

Compare: logistic loss grows **linearly** with margin → much more robust to outliers. This is why Gradient Boosting with logistic loss handles noise better than AdaBoost.

### Q7: Does AdaBoost need feature scaling?

**Answer:**
No. AdaBoost uses decision stumps/trees which make **threshold-based splits** and are invariant to monotonic feature transformations. Scaling changes values but not ordering, so it has no effect. Contrast: SVMs, KNN, logistic regression, and neural nets all require scaling.

### Q8: What's the theoretical guarantee of AdaBoost?

**Answer:**
**Training error bound:** If each weak learner has error $\varepsilon_t = 0.5 - \gamma_t$ (edge $\gamma_t > 0$):

$$\text{Training Error} \leq \exp\left(-2\sum_{t=1}^T \gamma_t^2\right)$$

This shows training error decreases **exponentially** with rounds — any weak learner (>50% accuracy) can be boosted to arbitrarily low training error. This is the formal statement of the boosting hypothesis and was one of the most important theoretical results in machine learning.

### Q9: How do you interpret the final AdaBoost prediction?

**Answer:**
$$H(x) = \text{sign}\left(\sum_{t=1}^T \alpha_t h_t(x)\right)$$

The argument of `sign()`, $F(x) = \sum_t \alpha_t h_t(x)$, is the **margin** or confidence score:
- $F(x) \gg 0$: confidently class +1
- $F(x) \ll 0$: confidently class -1
- $F(x) \approx 0$: uncertain, near decision boundary
- For probability: $P(y=1|x) = \frac{1}{1 + e^{-2F(x)}}$ (under certain assumptions)

---

## 21. Resources

### 📚 Books

| Book | Chapter | What You'll Learn |
|---|---|---|
| **ISLP** (James et al., 2023) | Ch. 8.2.3 — Boosting | Best conceptual intro to boosting |
| **Hands-On ML** (Géron, 3rd ed.) | Ch. 7 — Ensemble Methods | Code-heavy, AdaBoost + GBM |
| **ESL** (Hastie et al.) | Ch. 10 — Boosting | Advanced: exponential loss, FSAM |
| **Pattern Recognition** (Bishop) | Ch. 14 — Combining Models | Theoretical treatment |

### 🎬 StatQuest Videos (Josh Starmer)

| Video | Link | Runtime |
|---|---|---|
| AdaBoost, Clearly Explained | [Watch](https://www.youtube.com/watch?v=LsK-xG1cLYA) | 20 min |
| Gradient Boost Part 1 | [Watch](https://www.youtube.com/watch?v=3CC4N4z3GJc) | 16 min |
| Decision Trees | [Watch](https://www.youtube.com/watch?v=_L39rN6gz7Y) | 17 min |

### 🎓 Andrew Ng — ML Specialization

| Course | Week | Topic |
|---|---|---|
| Course 2: Advanced Algorithms | Week 4 | Decision Trees & Ensembles |
| Course 2: Advanced Algorithms | Week 4 | XGBoost (closely related) |

→ [Coursera ML Specialization](https://www.coursera.org/specializations/machine-learning-introduction)

> Andrew Ng's insight: "Boosting is one of the most powerful off-the-shelf ML methods. AdaBoost was historically the first practical boosting algorithm and paved the way for all modern gradient boosting methods."

### 🌐 Official Documentation

| Resource | URL |
|---|---|
| AdaBoostClassifier | https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html |
| AdaBoostRegressor | https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html |
| Ensemble Methods Guide | https://scikit-learn.org/stable/modules/ensemble.html#adaboost |
| Permutation Importance | https://scikit-learn.org/stable/modules/permutation_importance.html |
| SHAP Library | https://shap.readthedocs.io/ |
| Optuna | https://optuna.readthedocs.io/ |

### 📄 Original Papers

| Paper | Authors | Year |
|---|---|---|
| A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting | Freund & Schapire | 1997 |
| Additive Logistic Regression: a Statistical View of Boosting | Friedman, Hastie, Tibshirani | 2000 |
| Boosting the Margin: A New Explanation for the Effectiveness of Voting Methods | Schapire et al. | 1998 |
| Multi-class AdaBoost (SAMME) | Zhu et al. | 2009 |

### 💻 Practice Datasets

```python
# Binary Classification
from sklearn.datasets import load_breast_cancer    # medical, binary
from sklearn.datasets import load_heart_disease    # classic benchmark
from sklearn.datasets import make_classification   # synthetic, controllable

# Multi-class
from sklearn.datasets import load_wine             # 3 classes
from sklearn.datasets import load_iris             # classic, 3 classes

# Regression
from sklearn.datasets import load_diabetes
from sklearn.datasets import fetch_california_housing

# Kaggle
# - Titanic (binary)
# - Heart Disease (binary)
# - Credit Card Fraud (imbalanced binary)
```

---

## 📋 Quick Reference Cheat Sheet

```python
# ─── CLASSIFICATION ─────────────────────────────────────────────────
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

ada_clf = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),  # stump default
    n_estimators=200,          # number of weak learners
    learning_rate=1.0,         # shrinkage; lower → need more estimators
    algorithm='SAMME',         # use SAMME (SAMME.R deprecated in 1.4+)
    random_state=42
)
ada_clf.fit(X_train, y_train)
ada_clf.predict(X_test)
ada_clf.predict_proba(X_test)[:, 1]

# ─── REGRESSION ─────────────────────────────────────────────────────
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

ada_reg = AdaBoostRegressor(
    estimator=DecisionTreeRegressor(max_depth=3),
    n_estimators=200,
    learning_rate=0.5,
    loss='linear',             # 'linear', 'square', 'exponential'
    random_state=42
)

# ─── STAGED PREDICTIONS ─────────────────────────────────────────────
for i, y_staged in enumerate(ada_clf.staged_predict(X_test)):
    pass   # y_staged is prediction after round i+1

for proba_staged in ada_clf.staged_predict_proba(X_test):
    pass   # probability after each round

# ─── INSPECT COMPONENTS ─────────────────────────────────────────────
ada_clf.estimators_          # list of fitted stumps
ada_clf.estimator_weights_   # α_t values
ada_clf.estimator_errors_    # ε_t values
ada_clf.feature_importances_ # MDI feature importances

# ─── KEY MATH ────────────────────────────────────────────────────────
# Error:         ε_t = Σ wᵢ · 1[h(xᵢ)≠yᵢ] / Σ wᵢ
# Alpha:         α_t = ½ ln((1-ε_t)/ε_t)
# Weight update: wᵢ ← wᵢ · exp(-α_t · yᵢ · hₜ(xᵢ))
# Normalize:     wᵢ ← wᵢ / Σ wⱼ
# Prediction:    H(x) = sign(Σ αₜ · hₜ(x))
# Loss:          L = exp(-y · F(x))   ← exponential loss
# Error bound:   ≤ exp(-2 Σ γ_t²)    ← exponential decay!

# ─── FEATURE IMPORTANCE ─────────────────────────────────────────────
ada_clf.feature_importances_                          # MDI (fast)
from sklearn.inspection import permutation_importance
perm = permutation_importance(ada_clf, X_test, y_test, n_repeats=30)

import shap
explainer = shap.TreeExplainer(ada_clf)
shap_values = explainer.shap_values(X_test)

# ─── IMBALANCED DATA ─────────────────────────────────────────────────
from sklearn.utils.class_weight import compute_sample_weight
sw = compute_sample_weight('balanced', y=y_train)
ada_clf.fit(X_train, y_train, sample_weight=sw)
# Metric: use average_precision_score (PR-AUC), not accuracy
```

---

*Made with ❤️ for ML/DL job preparation. Good luck! 🚀*

> **Tip for Interviews:** Explain AdaBoost at 3 levels:
> 1. **Simple** — "Train weak learners sequentially; misclassified samples get higher weight so the next learner focuses on them; final answer is weighted majority vote"
> 2. **Technical** — "Each stump's voting power α_t = ½ ln((1-ε_t)/ε_t); sample weights updated as w_i ← w_i · exp(-α_t · y_i · h_t(x_i)); sensitive to outliers due to exponential loss"
> 3. **Mathematical** — "AdaBoost is Forward Stagewise Additive Modeling minimizing exponential loss L=e^{-yF(x)}; training error decays as exp(-2Σγ_t²) — any weak learner with edge γ>0 converges to zero training error"
