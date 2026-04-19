# 🌲 Random Forest — Complete Job-Ready Notes
> **Sources:** Andrew Ng ML Specialization · Hands-On ML (Aurélien Géron) · StatQuest (Josh Starmer) · Scikit-Learn Docs · ISLP (James et al.)

---

## 📚 Table of Contents
1. [What is Random Forest?](#1-what-is-random-forest)
2. [Big Picture — How It Works](#2-big-picture--how-it-works)
3. [Decision Trees Recap (Foundation)](#3-decision-trees-recap-foundation)
4. [Ensemble Methods — Why Combine Models?](#4-ensemble-methods--why-combine-models)
5. [Bagging — The Core Idea](#5-bagging--the-core-idea)
6. [Random Forest = Bagging + Feature Randomness](#6-random-forest--bagging--feature-randomness)
7. [The Math Behind Random Forest](#7-the-math-behind-random-forest)
8. [Gini Impurity & Entropy (Split Criteria)](#8-gini-impurity--entropy-split-criteria)
9. [Out-of-Bag (OOB) Error](#9-out-of-bag-oob-error)
10. [Bias-Variance Tradeoff in RF](#10-bias-variance-tradeoff-in-rf)
11. [Feature Importance](#11-feature-importance)
12. [Random Forest vs Decision Tree vs Bagging](#12-random-forest-vs-decision-tree-vs-bagging)
13. [Random Forest vs XGBoost vs Gradient Boosting](#13-random-forest-vs-xgboost-vs-gradient-boosting)
14. [Hyperparameters — Complete Guide](#14-hyperparameters--complete-guide)
15. [Python Code — From Scratch to Sklearn](#15-python-code--from-scratch-to-sklearn)
16. [Feature Importance Code](#16-feature-importance-code)
17. [Hyperparameter Tuning](#17-hyperparameter-tuning)
18. [Cross-Validation](#18-cross-validation)
19. [Handling Imbalanced Data](#19-handling-imbalanced-data)
20. [Extra Trees (ExtraTreesClassifier)](#20-extra-trees-extratreesclassifier)
21. [Tuning Strategy (Interview-Ready)](#21-tuning-strategy-interview-ready)
22. [Common Interview Questions](#22-common-interview-questions)
23. [Resources](#23-resources)

---

## 1. What is Random Forest?

**Simple Statement:**
Random Forest is an ensemble of decision trees, where each tree is trained on a **random bootstrap sample** of data and uses a **random subset of features** at each split. Predictions are made by **majority vote** (classification) or **averaging** (regression).

**Key Ideas:**
- Many weak learners (deep trees) → one strong learner
- Trees are trained **in parallel** (independent of each other)
- Two sources of randomness: **bootstrap sampling** + **feature randomness**
- Reduces **variance** without increasing bias much
- Robust to **overfitting**, outliers, and noise
- No need for feature scaling

> 📖 *ISLP, Ch. 8.2.2*: "Random forests provide an improvement over bagged trees by way of a small tweak that decorrelates the trees."

> 🎬 *StatQuest*: ["Random Forests Part 1 — Building, Using, and Evaluating"](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ)

> 📖 *Hands-On ML, Ch. 7*: "Random Forests are generally great out-of-the-box, even without much tuning."

---

## 2. Big Picture — How It Works

```
Training Data (n samples, p features)
          │
          ▼
┌─────────────────────────────────────────┐
│  For each tree t = 1 to B:              │
│                                         │
│  1. Bootstrap Sample                    │
│     → Sample n rows WITH replacement   │
│     → ~63% unique, ~37% repeated       │
│                                         │
│  2. Build Decision Tree on bootstrap   │
│     → At EACH split, randomly select  │
│       m features from p total          │
│     → Choose best split among m only  │
│     → Grow tree fully (no pruning)     │
│                                         │
│  3. Store the trained tree             │
└─────────────────────────────────────────┘
          │
          ▼
Final Prediction:
  Classification → Majority vote of B trees
  Regression     → Average of B trees
```

**Analogy:**
You want to pick the best restaurant. Instead of asking one expert (who might be biased), you ask 500 people — each with different taste backgrounds and exposed to different restaurants. The restaurant most people vote for is likely the best choice. That's Random Forest — wisdom of the crowd.

---

## 3. Decision Trees Recap (Foundation)

Random Forest is built on Decision Trees. You must understand these first.

### How a Split is Chosen

A node is split to maximize **purity** of child nodes.

**For Classification → Gini Impurity or Entropy**
**For Regression → Mean Squared Error (MSE)**

### Splitting for Regression

At a node with samples $\{y_1, ..., y_n\}$, for a split at threshold $t$ on feature $j$:

$$\text{MSE}_\text{node} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \bar{y})^2$$

$$\text{Weighted MSE} = \frac{n_L}{n}\text{MSE}_L + \frac{n_R}{n}\text{MSE}_R$$

Choose feature $j$ and threshold $t$ that **minimizes weighted MSE**.

### Stopping Criteria
- `max_depth` reached
- `min_samples_split`: not enough samples to split
- `min_samples_leaf`: split would create too-small leaf
- All samples have same label (pure node)

> 🎬 *StatQuest*: ["Decision Trees"](https://www.youtube.com/watch?v=_L39rN6gz7Y)

---

## 4. Ensemble Methods — Why Combine Models?

### The Crowd is Smarter Than Any Individual

If we have $B$ independent classifiers each with error $\epsilon$ (where $\epsilon < 0.5$):

The probability that the **majority vote is wrong**:

$$P(\text{majority wrong}) = \sum_{k > B/2}^{B} \binom{B}{k} \epsilon^k (1-\epsilon)^{B-k}$$

**Example:** 100 classifiers, each 70% accurate ($\epsilon = 0.3$):

$$P(\text{majority wrong}) \approx 0.0000035$$  

→ The ensemble achieves **99.9997% accuracy!**

**Condition:** Classifiers must be **independent** (uncorrelated). Random Forest achieves this via bootstrap + feature randomness.

### Types of Ensembles

| Method | How | Reduces |
|---|---|---|
| **Bagging** | Train same model on different bootstrap samples | Variance |
| **Boosting** | Train sequentially, focus on errors | Bias |
| **Stacking** | Train meta-model on base model predictions | Both |
| **Voting** | Average/vote predictions from different models | Variance |

> 📖 *Hands-On ML, Ch. 7*: Ensemble methods work best when the base models are diverse and independent.

---

## 5. Bagging — The Core Idea

**Bagging = Bootstrap Aggregating** (Leo Breiman, 1996)

### Bootstrap Sampling

From $n$ training samples, draw $n$ samples **with replacement**:
- Each bootstrap sample has $n$ instances (same size)
- On average, **~63.2%** of original samples appear at least once
- **~36.8%** of samples are NOT in any given bootstrap sample (these are OOB samples)

**Why 63.2%?**  
Probability a specific sample is NOT chosen in one draw: $1 - \frac{1}{n}$

Probability it's NOT chosen in $n$ draws: $\left(1-\frac{1}{n}\right)^n \xrightarrow{n\to\infty} e^{-1} \approx 0.368$

So probability it IS chosen: $1 - 0.368 = 0.632$

### Bagging Algorithm

```
Input: Training set S = {(x₁,y₁), ..., (xₙ,yₙ)}, B trees

For b = 1 to B:
    1. S*_b = Bootstrap sample from S (n samples with replacement)
    2. Train decision tree T_b on S*_b (fully grown, no pruning)

Prediction (classification):
    ŷ = majority_vote{T_1(x), T_2(x), ..., T_B(x)}

Prediction (regression):
    ŷ = (1/B) Σ T_b(x)
```

### Why Bagging Reduces Variance

If we average $B$ independent models each with variance $\sigma^2$:

$$\text{Var}\left(\frac{1}{B}\sum_{b=1}^{B}T_b\right) = \frac{\sigma^2}{B}$$

But trees trained on bootstrap samples from the same data are **not independent** — they're **correlated** (correlation $\rho$):

$$\text{Var}_\text{bagging} = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$$

As $B \to \infty$: variance → $\rho\sigma^2$

**Problem:** If $\rho$ is large (trees are highly correlated), we can't reduce variance much.

**Random Forest's Solution:** Reduce $\rho$ by using only $m < p$ features at each split!

> 📖 *ISLP, Ch. 8.2.1*: "The main difference between bagging and random forests is the choice of predictor subset size $m$."

---

## 6. Random Forest = Bagging + Feature Randomness

### The Key Innovation

At **each split** in each tree:
- Instead of considering all $p$ features → consider only $m$ randomly selected features
- Choose the best split **among those $m$ features only**
- Default: $m = \sqrt{p}$ (classification), $m = p/3$ (regression)

### Why This Works

**The Decorrelation Argument:**

Suppose there's 1 very strong feature. In bagging, almost every tree will use it at the top → trees are **highly correlated** → averaging doesn't help much.

In Random Forest, that dominant feature is **excluded from many splits** → different trees find different patterns → trees are **less correlated** → averaging reduces variance more effectively.

$$\rho_\text{RF} < \rho_\text{Bagging}$$
$$\text{Var}_\text{RF} < \text{Var}_\text{Bagging}$$

### Effect of $m$ on Bias-Variance

| $m$ value | Correlation $\rho$ | Bias | Variance | When to use |
|---|---|---|---|---|
| Small $m$ (e.g., 1) | Low | High | Low | Noisy, many irrelevant features |
| $m = \sqrt{p}$ | Medium | Medium | Medium | Default for classification |
| $m = p/3$ | Medium | Medium | Medium | Default for regression |
| $m = p$ (all features) | High | Low | High | Same as Bagging |

> 🎬 *StatQuest*: ["Random Forests Part 2 — Missing Data and Clustering"](https://www.youtube.com/watch?v=sQ870aTKqiM)

---

## 7. The Math Behind Random Forest

### Prediction (Classification — Majority Vote)

Let $T_b(x) \in \{1, ..., K\}$ be the prediction of tree $b$. Final prediction:

$$\hat{y} = \arg\max_k \sum_{b=1}^{B} \mathbf{1}[T_b(x) = k]$$

For **soft voting** (average probabilities):

$$\hat{p}_k(x) = \frac{1}{B}\sum_{b=1}^{B} p_{bk}(x)$$
$$\hat{y} = \arg\max_k \hat{p}_k(x)$$

### Prediction (Regression — Average)

$$\hat{y}(x) = \frac{1}{B}\sum_{b=1}^{B} T_b(x)$$

### Bias-Variance Decomposition

For a single tree $T$:
$$\text{MSE}(T) = \text{Bias}^2(T) + \text{Var}(T) + \text{Irreducible Error}$$

For the average of $B$ trees with pairwise correlation $\rho$:
$$\text{Var}\left(\bar{T}\right) = \rho \cdot \sigma^2 + \frac{1-\rho}{B} \cdot \sigma^2$$

As $B \to \infty$: converges to $\rho \cdot \sigma^2$

**Bias stays the same** (deep trees have low bias).
**Variance decreases** (more trees + lower correlation = lower variance).

### Convergence — How Many Trees?

There is a **law of diminishing returns**. Error decreases roughly as:

$$\text{OOB Error}(B) \approx \text{OOB Error}(\infty) + \frac{C}{B}$$

After ~500 trees, adding more rarely helps. Use OOB error or validation set to decide.

---

## 8. Gini Impurity & Entropy (Split Criteria)

### Gini Impurity

For a node with $K$ classes, where $p_k$ = proportion of class $k$:

$$G = \sum_{k=1}^{K} p_k(1-p_k) = 1 - \sum_{k=1}^{K} p_k^2$$

- **Range:** $[0, 0.5]$ for binary, $[0, 1-1/K]$ in general
- **Pure node:** $G = 0$ (all same class)
- **Most impure (binary):** $G = 0.5$ (50/50 split)

**Weighted Gini for a split:**
$$G_\text{split} = \frac{n_L}{n} G_L + \frac{n_R}{n} G_R$$

**Information Gain (Gini-based):**
$$\text{Gain} = G_\text{parent} - G_\text{split}$$

### Entropy (Information Gain)

$$H = -\sum_{k=1}^{K} p_k \log_2(p_k)$$

- **Range:** $[0, \log_2 K]$
- **Pure node:** $H = 0$
- **Binary, most impure:** $H = 1$ bit

**Information Gain:**
$$IG = H_\text{parent} - \left(\frac{n_L}{n}H_L + \frac{n_R}{n}H_R\right)$$

### Gini vs Entropy

| | Gini | Entropy |
|---|---|---|
| **Computation** | Faster (no log) | Slower |
| **Result** | Similar | Very similar |
| **Bias** | Slightly favors larger partitions | Slightly favors balanced splits |
| **Default in sklearn** | `criterion='gini'` | `criterion='entropy'` |

> **Practical advice:** The choice rarely matters much. Gini is the default and is slightly faster.

### Numerical Example

Node with 10 samples: 6 class A, 4 class B

$$G = 1 - (0.6^2 + 0.4^2) = 1 - (0.36 + 0.16) = 0.48$$

$$H = -(0.6 \log_2 0.6 + 0.4 \log_2 0.4) = -(−0.442 − 0.529) = 0.971 \text{ bits}$$

After split: Left (8 samples: 6A, 2B), Right (2 samples: 0A, 2B)

$$G_L = 1 - (0.75^2 + 0.25^2) = 0.375$$
$$G_R = 1 - (0^2 + 1^2) = 0 \quad \text{(pure!)}$$
$$G_\text{split} = \frac{8}{10}(0.375) + \frac{2}{10}(0) = 0.30$$
$$\text{Gain} = 0.48 - 0.30 = 0.18$$

---

## 9. Out-of-Bag (OOB) Error

### What is OOB?

Each tree is trained on ~63% of data. The remaining ~37% (**Out-of-Bag** samples) are used to estimate generalization error — **for free**, without a separate validation set!

### How OOB Error is Computed

```
For each sample xᵢ:
    1. Find all trees that did NOT train on xᵢ (OOB trees for xᵢ)
    2. Get predictions from only those trees
    3. Aggregate (vote or average) → OOB prediction ŷᵢ_OOB

OOB Error = metric(y, ŷ_OOB)   # accuracy, MSE, etc.
```

**Mathematically:**

$$\text{OOB Error} = \frac{1}{n}\sum_{i=1}^{n} L\left(y_i,\ \frac{1}{|\mathcal{B}_i|}\sum_{b \in \mathcal{B}_i} T_b(x_i)\right)$$

Where $\mathcal{B}_i = \{b : x_i \notin \text{bootstrap sample}_b\}$

### Key Properties
- OOB error ≈ Leave-One-Out CV error for large forests
- Much cheaper than k-fold CV (no extra training)
- Reliable estimate of test error
- Enable with `oob_score=True` in sklearn

```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=500, oob_score=True, random_state=42)
rf.fit(X_train, y_train)
print(f"OOB Accuracy: {rf.oob_score_:.4f}")
```

> 📖 *ISLP, Ch. 8.2.2*: "The OOB approach is essentially the same as leave-one-out cross-validation."

---

## 10. Bias-Variance Tradeoff in RF

### Visual Understanding

```
Single Deep Tree:
    Low Bias  ──────────────────────── High Variance
    (fits training perfectly)          (changes a lot with different data)

Bagging (averaged trees):
    Low Bias  ──────────────────────── Reduced Variance
                                       (averaging cancels out variance)

Random Forest (decorrelated trees):
    Low Bias  ──────────────────────── Even Lower Variance
                                       (decorrelation + averaging)
```

### Effect of Hyperparameters on Bias-Variance

| Parameter | Increase → | Bias | Variance |
|---|---|---|---|
| `n_estimators` | More trees | Same | Decreases |
| `max_depth` | Deeper | Decreases | Increases |
| `min_samples_leaf` | More samples | Increases | Decreases |
| `max_features` | More features | Decreases | Increases |
| `max_samples` | More bootstrap samples | Same | Decreases slightly |

**Key insight from Andrew Ng ML Specialization:**  
> Boosting (XGBoost) reduces **bias** by building on errors.  
> Bagging/RF reduces **variance** by averaging independent models.

---

## 11. Feature Importance

### Mean Decrease in Impurity (MDI)

For each feature $j$, sum the weighted impurity decrease across all splits using feature $j$, averaged over all trees:

$$\text{FI}(j) = \frac{1}{B}\sum_{b=1}^{B}\sum_{\substack{t \in T_b \\ \text{split on } j}} \frac{n_t}{n} \cdot \Delta G_t$$

Where:
- $n_t$ = number of samples at node $t$
- $\Delta G_t$ = impurity decrease at node $t$
- Normalized so all importances sum to 1

**⚠️ Bias Warning (MDI):** MDI tends to inflate importance of **high-cardinality** features (many unique values). This is because high-cardinality features have more possible thresholds and more chances to look useful by chance.

### Permutation Importance (Better)

1. Compute baseline score on validation set
2. For feature $j$: **randomly shuffle** its values, compute new score
3. Importance = decrease in score due to shuffling

$$\text{PI}(j) = \text{score}_\text{baseline} - \text{score}_{\text{feature } j \text{ shuffled}}$$

Less biased, works with any model, but slower.

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(
    rf, X_test, y_test,
    n_repeats=30, random_state=42, n_jobs=-1
)
```

> 📖 *Hands-On ML, Ch. 7*: "Permutation importance is generally more reliable than MDI, especially when features have different scales or cardinalities."

---

## 12. Random Forest vs Decision Tree vs Bagging

| Feature | Decision Tree | Bagging | Random Forest |
|---|---|---|---|
| **Training** | Single tree | B trees on bootstrap samples | B trees on bootstrap + random features |
| **Feature selection at split** | All features | All features | $m = \sqrt{p}$ or $p/3$ |
| **Variance** | Very High | Medium | Low |
| **Bias** | Low (deep) | Low (deep) | Low (deep) |
| **Overfitting** | High | Medium | Low |
| **Interpretability** | High | Low | Low |
| **Parallelizable** | No (single) | Yes | Yes |
| **OOB Error** | No | Yes | Yes |
| **Speed** | Fast | Slow | Slow (but parallelizable) |

---

## 13. Random Forest vs XGBoost vs Gradient Boosting

| Feature | Random Forest | Gradient Boosting | XGBoost |
|---|---|---|---|
| **Tree building** | Parallel | Sequential | Sequential (parallel splits) |
| **Error correction** | None (independent) | Residuals (1st order) | 2nd-order Taylor expansion |
| **Reduces** | Variance | Bias | Bias (+ regularization) |
| **Overfitting** | Rarely | Moderate | Less (regularized) |
| **Speed** | Fast (parallel) | Slow | Fast |
| **Hyperparameter sensitivity** | Low | High | High |
| **Missing values** | Needs imputation | Needs imputation | Native support |
| **Feature scaling** | Not needed | Not needed | Not needed |
| **Best for** | Quick baseline, noisy data | Structured data | Kaggle, structured data |
| **Tuning effort** | Low | High | High |

> **Rule of Thumb:** Start with Random Forest as baseline → try XGBoost for better performance.

---

## 14. Hyperparameters — Complete Guide

### 🌲 Tree Structure Parameters

| Parameter | Default | Range | Effect |
|---|---|---|---|
| `n_estimators` | 100 | 100–2000 | More trees = lower variance; use OOB to pick |
| `max_depth` | None (full) | 5–50 or None | Deeper = lower bias, higher variance |
| `min_samples_split` | 2 | 2–20 | Min samples to split a node; higher = more conservative |
| `min_samples_leaf` | 1 | 1–20 | Min samples in a leaf; higher = smoother predictions |
| `max_features` | `"sqrt"` | `"sqrt"`, `"log2"`, int, float | Features per split; lower = less correlation between trees |
| `max_leaf_nodes` | None | 10–1000 | Limits tree complexity |
| `min_impurity_decrease` | 0.0 | 0.0–0.1 | Min impurity gain required for a split |

### 🎲 Randomness Parameters

| Parameter | Default | Notes |
|---|---|---|
| `bootstrap` | True | Whether to use bootstrap; False = use all data (pasting) |
| `max_samples` | None (= n) | If bootstrap=True, size of each bootstrap sample |
| `random_state` | None | Set for reproducibility |
| `oob_score` | False | Use OOB samples for evaluation |

### 🎯 Learning Task Parameters

| Parameter | Default | Notes |
|---|---|---|
| `criterion` | `"gini"` | `"gini"` or `"entropy"` for classification; `"squared_error"`, `"absolute_error"` for regression |
| `class_weight` | None | `"balanced"` for imbalanced data |
| `n_jobs` | None (=1) | `-1` to use all CPUs |
| `warm_start` | False | Add more trees to existing forest without retraining |

### 🎯 Tuning Priority Order

```
1. n_estimators → use OOB score or validation curve
2. max_features → most impactful for decorrelation
3. max_depth or min_samples_leaf → control tree size
4. min_samples_split
5. max_leaf_nodes (alternative to max_depth)
```

---

## 15. Python Code — From Scratch to Sklearn

### Installation

```bash
pip install scikit-learn numpy pandas matplotlib seaborn shap
```

### Basic Usage — Classification

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)

# Load data
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,           # fully grown trees
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',      # sqrt(p) features per split
    bootstrap=True,
    oob_score=True,           # free validation estimate
    class_weight=None,
    random_state=42,
    n_jobs=-1,
    verbose=0
)

rf.fit(X_train, y_train)

# Predict
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

# Evaluate
print(f"Test Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"OOB Accuracy  : {rf.oob_score_:.4f}")
print(f"AUC-ROC       : {roc_auc_score(y_test, y_prob):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))
```

### Basic Usage — Regression

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score

# Data
X_r, y_r = fetch_california_housing(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X_r, y_r, test_size=0.2, random_state=42
)

# Train
rfr = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=1/3,          # p/3 for regression
    bootstrap=True,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)

rfr.fit(X_tr, y_tr)
y_pred = rfr.predict(X_te)

print(f"Test RMSE : {np.sqrt(mean_squared_error(y_te, y_pred)):.4f}")
print(f"Test R²   : {r2_score(y_te, y_pred):.4f}")
print(f"OOB R²    : {rfr.oob_score_:.4f}")
```

### Validation Curve — Finding Optimal n_estimators

```python
from sklearn.model_selection import validation_curve

param_range = [10, 50, 100, 200, 300, 500, 750, 1000]

train_scores, val_scores = validation_curve(
    RandomForestClassifier(max_features='sqrt', random_state=42, n_jobs=-1),
    X_train, y_train,
    param_name='n_estimators',
    param_range=param_range,
    scoring='accuracy',
    cv=5,
    n_jobs=-1
)

plt.figure(figsize=(10, 5))
plt.plot(param_range, np.mean(train_scores, axis=1), label='Train Score')
plt.plot(param_range, np.mean(val_scores, axis=1), label='CV Score')
plt.xlabel('n_estimators')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Validation Curve — n_estimators')
plt.grid(True)
plt.show()
```

### OOB Score Curve

```python
oob_scores = []
n_trees_range = range(10, 501, 10)

for n in n_trees_range:
    rf_temp = RandomForestClassifier(
        n_estimators=n,
        oob_score=True,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    rf_temp.fit(X_train, y_train)
    oob_scores.append(rf_temp.oob_score_)

plt.figure(figsize=(10, 5))
plt.plot(list(n_trees_range), oob_scores)
plt.xlabel('Number of Trees')
plt.ylabel('OOB Accuracy')
plt.title('OOB Score vs Number of Trees')
plt.axvline(x=n_trees_range[np.argmax(oob_scores)], color='r', linestyle='--',
            label=f'Best: {n_trees_range[np.argmax(oob_scores)]} trees')
plt.legend()
plt.grid(True)
plt.show()
```

### Full Pipeline with Preprocessing

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# For datasets with mixed types
numeric_features = X.select_dtypes(include='number').columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
    # Note: RF doesn't need scaling!
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
    ('rf', RandomForestClassifier(
        n_estimators=300,
        max_features='sqrt',
        oob_score=False,    # OOB doesn't work inside Pipeline
        random_state=42,
        n_jobs=-1
    ))
])

pipeline.fit(X_train, y_train)
print("Pipeline Accuracy:", pipeline.score(X_test, y_test))
```

---

## 16. Feature Importance Code

### MDI Feature Importance

```python
import pandas as pd
import matplotlib.pyplot as plt

# After fitting rf
importances = rf.feature_importances_
std_importances = np.std([t.feature_importances_ for t in rf.estimators_], axis=0)

feat_imp_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances,
    'std': std_importances
}).sort_values('importance', ascending=False).head(15)

plt.figure(figsize=(10, 6))
plt.barh(feat_imp_df['feature'], feat_imp_df['importance'],
         xerr=feat_imp_df['std'], color='steelblue', alpha=0.8)
plt.xlabel('Mean Decrease in Impurity')
plt.title('Random Forest — Feature Importance (MDI)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
```

### Permutation Importance

```python
from sklearn.inspection import permutation_importance

# More reliable than MDI
perm_result = permutation_importance(
    rf, X_test, y_test,
    n_repeats=30,
    random_state=42,
    n_jobs=-1
)

perm_df = pd.DataFrame({
    'feature': feature_names,
    'importance_mean': perm_result.importances_mean,
    'importance_std': perm_result.importances_std
}).sort_values('importance_mean', ascending=False).head(15)

plt.figure(figsize=(10, 6))
plt.barh(perm_df['feature'], perm_df['importance_mean'],
         xerr=perm_df['importance_std'], color='coral', alpha=0.8)
plt.xlabel('Mean Accuracy Decrease')
plt.title('Random Forest — Permutation Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
```

### SHAP Values

```python
import shap

# Tree explainer — fast for tree-based models
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)

# For binary classification, shap_values is a list [class0, class1]
# Use class 1 values
sv = shap_values[1] if isinstance(shap_values, list) else shap_values

# Summary plot (beeswarm)
shap.summary_plot(sv, X_test, feature_names=feature_names)

# Bar plot
shap.summary_plot(sv, X_test, plot_type='bar', feature_names=feature_names)

# Single prediction explanation
shap.force_plot(
    explainer.expected_value[1],
    sv[0],
    X_test[0],
    feature_names=feature_names
)
```

---

## 17. Hyperparameter Tuning

### GridSearchCV

```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold

param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', 0.3],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    RandomForestClassifier(bootstrap=True, random_state=42, n_jobs=-1),
    param_grid=param_grid,
    scoring='roc_auc',
    cv=cv,
    verbose=2,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print("Best Params:", grid_search.best_params_)
print("Best AUC:   ", grid_search.best_score_)
```

### RandomizedSearchCV

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {
    'n_estimators': randint(100, 1000),
    'max_depth': [None, 5, 10, 15, 20, 30, 50],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', 0.2, 0.3, 0.5],
    'bootstrap': [True, False],
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_distributions=param_dist,
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
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'max_depth': trial.suggest_categorical('max_depth',
                         [None, 5, 10, 15, 20, 30]),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features',
                            ['sqrt', 'log2', 0.2, 0.3, 0.5]),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'random_state': 42,
        'n_jobs': -1
    }

    model = RandomForestClassifier(**params)
    scores = cross_val_score(model, X_train, y_train,
                             cv=5, scoring='roc_auc', n_jobs=-1)
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, show_progress_bar=True)

print("Best params:", study.best_trial.params)
print("Best AUC:  ", study.best_value)

# Get best model
best_rf = RandomForestClassifier(**study.best_trial.params)
best_rf.fit(X_train, y_train)
```

### Warm Start (Add Trees Incrementally)

```python
# Start with 100 trees, evaluate, then add more if needed
rf_warm = RandomForestClassifier(
    n_estimators=100,
    warm_start=True,      # keeps existing trees, adds new ones
    oob_score=True,
    random_state=42,
    n_jobs=-1
)

oob_scores = []
for n in range(100, 1001, 100):
    rf_warm.n_estimators = n
    rf_warm.fit(X_train, y_train)
    oob_scores.append(rf_warm.oob_score_)
    print(f"n_estimators={n}, OOB={rf_warm.oob_score_:.4f}")

# Best number of trees
best_n = (np.argmax(oob_scores) + 1) * 100
print(f"\nBest n_estimators: {best_n}")
```

---

## 18. Cross-Validation

```python
from sklearn.model_selection import (cross_val_score, cross_validate,
                                     StratifiedKFold, KFold)

cv_clf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Single metric
scores = cross_val_score(
    rf, X, y, cv=cv_clf, scoring='roc_auc', n_jobs=-1
)
print(f"AUC: {scores.mean():.4f} ± {scores.std():.4f}")

# Multiple metrics
cv_results = cross_validate(
    rf, X, y,
    cv=cv_clf,
    scoring=['roc_auc', 'accuracy', 'f1', 'precision', 'recall'],
    return_train_score=True,
    n_jobs=-1
)

for metric in ['test_roc_auc', 'test_accuracy', 'test_f1']:
    m = cv_results[metric]
    print(f"{metric:20s}: {m.mean():.4f} ± {m.std():.4f}")

# Learning Curve
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    rf, X, y,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5, scoring='accuracy', n_jobs=-1
)

plt.figure(figsize=(10, 5))
plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Train')
plt.plot(train_sizes, np.mean(val_scores, axis=1), label='Validation')
plt.fill_between(train_sizes,
                 np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                 np.mean(train_scores, axis=1) + np.std(train_scores, axis=1),
                 alpha=0.2)
plt.fill_between(train_sizes,
                 np.mean(val_scores, axis=1) - np.std(val_scores, axis=1),
                 np.mean(val_scores, axis=1) + np.std(val_scores, axis=1),
                 alpha=0.2)
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Learning Curve — Random Forest')
plt.grid(True)
plt.show()
```

---

## 19. Handling Imbalanced Data

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Method 1: class_weight='balanced'
rf_balanced = RandomForestClassifier(
    n_estimators=300,
    class_weight='balanced',    # weights = n_samples / (n_classes * np.bincount(y))
    random_state=42,
    n_jobs=-1
)

# Method 2: Manual class weights
classes = np.unique(y_train)
weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, weights))
print("Class weights:", class_weight_dict)

rf_manual = RandomForestClassifier(
    n_estimators=300,
    class_weight=class_weight_dict,
    random_state=42,
    n_jobs=-1
)

# Method 3: class_weight='balanced_subsample'
# Computes weights on each bootstrap sample separately
rf_subsample = RandomForestClassifier(
    n_estimators=300,
    class_weight='balanced_subsample',
    random_state=42,
    n_jobs=-1
)

# Method 4: SMOTE + RF
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

pipeline_smote = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1))
])
pipeline_smote.fit(X_train, y_train)

# Evaluation for imbalanced: use PR-AUC or F1, NOT accuracy
from sklearn.metrics import average_precision_score, f1_score

y_prob = rf_balanced.fit(X_train, y_train).predict_proba(X_test)[:, 1]
print(f"PR-AUC: {average_precision_score(y_test, y_prob):.4f}")
print(f"F1:     {f1_score(y_test, rf_balanced.predict(X_test)):.4f}")
```

---

## 20. Extra Trees (ExtraTreesClassifier)

**Extremely Randomized Trees** — a variant that adds even more randomness.

### Difference from Random Forest

| | Random Forest | Extra Trees |
|---|---|---|
| **Bootstrap** | Yes (bootstrap samples) | No (uses full dataset) |
| **Split threshold** | Best among $m$ features | **Random** threshold for each of $m$ features |
| **Speed** | Slower | **Faster** (no best-split search) |
| **Variance** | Low | Even Lower |
| **Bias** | Low | Slightly Higher |

```python
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

# Classification
et_clf = ExtraTreesClassifier(
    n_estimators=300,
    max_features='sqrt',
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)
et_clf.fit(X_train, y_train)
print(f"ExtraTrees Accuracy: {et_clf.score(X_test, y_test):.4f}")

# Regression
et_reg = ExtraTreesRegressor(
    n_estimators=300,
    max_features=1/3,
    random_state=42,
    n_jobs=-1
)
et_reg.fit(X_tr, y_tr)
print(f"ExtraTrees R²: {et_reg.score(X_te, y_te):.4f}")
```

**When to use Extra Trees over RF:**
- When speed is critical
- Very high-dimensional data
- When RF is overfitting (more randomness helps)

---

## 21. Tuning Strategy (Interview-Ready)

### Complete Step-by-Step Guide

```
STEP 1: Baseline
─────────────────
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
Evaluate: OOB score, cross-val score
Goal: Establish baseline performance

STEP 2: Find optimal n_estimators
──────────────────────────────────
Use warm_start + OOB score curve
Typical: 200–500 for most datasets

STEP 3: Tune max_features
──────────────────────────
Options: 'sqrt', 'log2', 0.2, 0.3, 0.5
This is the MOST important RF hyperparameter
Lower = more decorrelated trees = lower variance

STEP 4: Tune tree depth / complexity
──────────────────────────────────────
Option A: max_depth = [None, 10, 20, 30]
Option B: min_samples_leaf = [1, 2, 4, 8]
Option C: min_samples_split = [2, 5, 10]
These control bias-variance tradeoff

STEP 5: Final evaluation
──────────────────────────
Use held-out test set ONCE
Report: accuracy/AUC/RMSE + confidence intervals via bootstrapping
```

### Quick Cheat — When RF Overfits

```python
# Signs: High train score, low val/OOB score
# Fix: Increase regularization

rf_fixed = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,              # limit depth
    min_samples_leaf=5,        # larger leaves
    min_samples_split=10,      # harder to split
    max_features='sqrt',       # keep low
    max_leaf_nodes=100,        # hard limit
    random_state=42,
    n_jobs=-1
)
```

### Quick Cheat — When RF Underfits

```python
# Signs: Both train and val score are low
# Fix: Increase model complexity

rf_fixed = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,            # full trees
    min_samples_leaf=1,        # small leaves
    min_samples_split=2,       # easy splits
    max_features=0.5,          # more features
    random_state=42,
    n_jobs=-1
)
```

---

## 22. Common Interview Questions

### Q1: How does Random Forest reduce variance without increasing bias?

**Answer:**
- RF uses **fully grown trees** (low bias) — same as a single deep tree
- But it **averages** B trees, reducing variance by factor $\approx 1/B$ for uncorrelated trees
- The feature randomness **decorrelates** the trees, making the average more effective
- Result: same low bias, much lower variance than a single tree
- Mathematically: $\text{Var}(\bar{T}) = \rho\sigma^2 + \frac{(1-\rho)}{B}\sigma^2 \xrightarrow{B \to \infty} \rho\sigma^2$

### Q2: Why does RF use $\sqrt{p}$ features per split?

**Answer:**
- This is a hyperparameter (`max_features`), not a fixed rule
- $\sqrt{p}$ is the empirical default for classification, $p/3$ for regression
- Smaller $m$ → less correlated trees → more variance reduction
- Smaller $m$ → worse individual trees (higher bias) — tradeoff
- $\sqrt{p}$ empirically balances this tradeoff well across many datasets
- Always tune this as it's the most important RF hyperparameter

### Q3: What is OOB error and is it reliable?

**Answer:**
- Each tree is trained on ~63.2% of data; OOB = the other ~36.8%
- For each sample, we average predictions from all trees that didn't train on it
- OOB error ≈ Leave-One-Out CV for large forests
- Reliable because: OOB samples are truly unseen by each tree, large forests → many trees per OOB sample
- Advantage: free (no extra training), no data split needed
- Slight downside: each OOB estimate uses fewer trees than the full forest → slightly pessimistic

### Q4: When would you choose RF over XGBoost?

**Answer:**
- RF is **faster to train** (parallel) — prefer for quick iterations
- RF is **more robust** to hyperparameter choices — less tuning needed
- RF works better with **noisy data** — boosting can overfit to noise
- RF is better when **interpretability matters** — individual trees are visualizable
- XGBoost generally achieves **better predictive performance** on clean tabular data
- **Rule:** Use RF for baseline, use XGBoost when squeezing out maximum performance

### Q5: Does Random Forest need feature scaling?

**Answer:**
No. Decision trees (and therefore RF) make **rank-based splits** — only the ordering of feature values matters, not their magnitude. Scaling changes values but not ordering, so it has no effect on tree-based models.

Compare: Logistic Regression, SVM, KNN, and Neural Networks all require scaling.

### Q6: How does RF handle missing values?

**Answer:**
Sklearn's RF does NOT natively handle missing values — requires imputation first. Options:
1. **SimpleImputer** (mean/median/mode)
2. **KNNImputer** — use neighbor similarity
3. **IterativeImputer** — model-based imputation
4. **MissForest** (from `missingpy`) — uses RF itself to impute missing values iteratively — often best

```python
from sklearn.impute import SimpleImputer, KNNImputer
# or
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
```

### Q7: What's the difference between max_depth=None and max_depth=10?

**Answer:**
- `max_depth=None`: trees grow until all leaves are pure or have fewer than `min_samples_split` samples → **fully grown trees** → low bias, high variance (but averaging reduces variance)
- `max_depth=10`: trees stop at depth 10 → **pruned trees** → higher bias, lower variance
- In RF, `max_depth=None` is the default because RF controls variance through **averaging + decorrelation**, not through pruning
- Setting `max_depth` in RF is an additional regularization when needed

### Q8: What is `min_child_weight` in XGBoost vs `min_samples_leaf` in RF?

**Answer:**
- Both control the minimum size of a leaf node
- `min_samples_leaf` in RF: minimum number of training samples in a leaf
- `min_child_weight` in XGBoost: minimum sum of hessians (≈ samples for MSE, but sample-weighted for logistic loss)
- XGBoost's version is smarter — it accounts for sample uncertainty, not just count

---

## 23. Resources

### 📚 Books

| Book | Chapter | What You'll Learn |
|---|---|---|
| **ISLP** (James et al., 2023) | Ch. 8.2 — Bagging, RF, Boosting | Best theoretical foundation |
| **Hands-On ML** (Géron, 3rd ed.) | Ch. 7 — Ensemble Learning | Code-heavy, practical examples |
| **ESL** (Hastie et al.) | Ch. 15 — Random Forests | Advanced theory, original paper perspective |
| **Python ML** (Raschka) | Ch. 7 — Ensemble Methods | Good practical code |

### 🎬 StatQuest Videos (Josh Starmer)

| Video | Link | Runtime |
|---|---|---|
| Decision Trees | [Watch](https://www.youtube.com/watch?v=_L39rN6gz7Y) | 17 min |
| Random Forests Part 1 | [Watch](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ) | 9 min |
| Random Forests Part 2 (Missing Data) | [Watch](https://www.youtube.com/watch?v=sQ870aTKqiM) | 10 min |
| Gini Impurity | [Watch](https://www.youtube.com/watch?v=7VeUPuFGJHk) | 8 min |
| Entropy / Information Gain | [Watch](https://www.youtube.com/watch?v=YtebGVx-Fxw) | 12 min |
| Bagging | [Watch](https://www.youtube.com/watch?v=sVriC_Ys2cw) | 7 min |

### 🎓 Andrew Ng — ML Specialization

| Course | Week | Topic |
|---|---|---|
| Course 2: Advanced Algorithms | Week 4 | Decision Trees |
| Course 2: Advanced Algorithms | Week 4 | Tree Ensembles (RF, XGBoost) |

→ [Coursera ML Specialization](https://www.coursera.org/specializations/machine-learning-introduction)

> Andrew Ng's key insight: "Sampling with replacement + random feature selection = trees that are diverse enough to combine effectively."

### 🌐 Official Documentation

| Resource | URL |
|---|---|
| RandomForestClassifier | https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html |
| RandomForestRegressor | https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html |
| ExtraTreesClassifier | https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html |
| Ensemble Methods Guide | https://scikit-learn.org/stable/modules/ensemble.html |
| Permutation Importance | https://scikit-learn.org/stable/modules/permutation_importance.html |
| SHAP Library | https://shap.readthedocs.io/ |
| Optuna | https://optuna.readthedocs.io/ |

### 📄 Original Paper

| Paper | Authors | Year |
|---|---|---|
| Random Forests | Leo Breiman | 2001 |
| → https://link.springer.com/article/10.1023/A:1010933404324 | | |

### 💻 Practice Datasets

```python
# Classification
from sklearn.datasets import load_breast_cancer    # binary
from sklearn.datasets import load_wine             # multi-class
from sklearn.datasets import load_iris             # classic
from sklearn.datasets import fetch_covtype         # large, multi-class

# Regression
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import load_diabetes

# Kaggle Competitions (great for RF practice)
# - Titanic (binary classification)
# - House Prices (regression)
# - Forest Cover Type (multi-class, literally about forests!)
# - Credit Card Fraud (imbalanced classification)
```

---

## 📋 Quick Reference Cheat Sheet

```python
# ─── CLASSIFICATION ─────────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(
    n_estimators=500,           # more = better (diminishing returns)
    max_depth=None,             # fully grown (RF controls variance via averaging)
    min_samples_split=2,        # default
    min_samples_leaf=1,         # default
    max_features='sqrt',        # KEY: sqrt(p) for classification
    bootstrap=True,             # default: bootstrap sampling
    oob_score=True,             # free validation estimate
    class_weight=None,          # 'balanced' for imbalanced data
    random_state=42,
    n_jobs=-1                   # use all cores
)

# ─── REGRESSION ─────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    max_features=1/3,           # KEY: p/3 for regression
    oob_score=True,
    random_state=42,
    n_jobs=-1
)

# ─── EXTRA TREES ────────────────────────────────────────────────────
from sklearn.ensemble import ExtraTreesClassifier
et = ExtraTreesClassifier(n_estimators=300, max_features='sqrt',
                          random_state=42, n_jobs=-1)

# ─── KEY MATH ────────────────────────────────────────────────────────
# Gini:  G = 1 - Σ pₖ²
# Entropy: H = -Σ pₖ log₂(pₖ)
# Info Gain: IG = H_parent - (nL/n)·H_L - (nR/n)·H_R
# Variance of RF: ρσ² + (1-ρ)/B · σ²
# Bootstrap retention: 1 - (1-1/n)ⁿ ≈ 1 - e⁻¹ ≈ 0.632
# OOB fraction: ≈ 36.8%
# Default max_features: √p (classification), p/3 (regression)

# ─── FEATURE IMPORTANCE ─────────────────────────────────────────────
# MDI (fast, biased toward high-cardinality):
rf.feature_importances_

# Permutation (slower, unbiased):
from sklearn.inspection import permutation_importance
perm = permutation_importance(rf, X_test, y_test, n_repeats=30, n_jobs=-1)

# SHAP (best for interpretability):
import shap
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)

# ─── IMBALANCED DATA ─────────────────────────────────────────────────
# Option 1: class_weight='balanced'
# Option 2: class_weight='balanced_subsample'
# Option 3: SMOTE from imbalanced-learn
# Metric: use AUC-PR (average_precision_score), not accuracy
```

---

*Made with ❤️ for ML/DL job preparation. Good luck! 🚀*

> **Tip for Interviews:** Explain Random Forest at 3 levels:
> 1. **Simple** — "Many decision trees, each trained on random data + random features, then averaged"
> 2. **Technical** — "Bootstrap aggregation + feature randomness decorrelates trees, reducing variance while keeping bias low"
> 3. **Mathematical** — "Var(RF) = ρσ² + (1-ρ)/B · σ², where ρ is pairwise tree correlation. Feature randomness reduces ρ; averaging reduces the (1-ρ)/B term"
