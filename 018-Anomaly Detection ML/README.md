# 🔴 Anomaly Detection — Complete ML/DL Job-Ready Notes

> **"Find the needle in the haystack — the rare, unusual, or suspicious data point that doesn't fit the pattern."**
> Anomaly Detection (also called Outlier Detection) identifies data points that deviate significantly from the expected distribution or behavior.

---

## 📌 Table of Contents

1. [Intuition](#1-intuition)
2. [Types of Anomalies](#2-types-of-anomalies)
3. [Approaches Overview](#3-approaches-overview)
4. [Gaussian (Normal) Distribution — The Foundation](#4-gaussian-normal-distribution--the-foundation)
5. [Density Estimation Algorithm — Full Math](#5-density-estimation-algorithm--full-math)
6. [Multivariate Gaussian Distribution](#6-multivariate-gaussian-distribution)
7. [Univariate vs Multivariate Gaussian](#7-univariate-vs-multivariate-gaussian)
8. [Choosing the Threshold (Epsilon)](#8-choosing-the-threshold-epsilon)
9. [Isolation Forest — Algorithm & Math](#9-isolation-forest--algorithm--math)
10. [Local Outlier Factor (LOF)](#10-local-outlier-factor-lof)
11. [One-Class SVM](#11-one-class-svm)
12. [Elliptic Envelope](#12-elliptic-envelope)
13. [Bias-Variance & Evaluation Challenge](#13-bias-variance--evaluation-challenge)
14. [Hyperparameters — Complete Guide](#14-hyperparameters--complete-guide)
15. [Python from Scratch — Gaussian Anomaly Detection](#15-python-from-scratch--gaussian-anomaly-detection)
16. [Scikit-learn Implementation](#16-scikit-learn-implementation)
17. [Full Pipeline with Best Practices](#17-full-pipeline-with-best-practices)
18. [Anomaly Detection vs Supervised Learning](#18-anomaly-detection-vs-supervised-learning)
19. [Feature Engineering for Anomaly Detection](#19-feature-engineering-for-anomaly-detection)
20. [Evaluation Metrics](#20-evaluation-metrics)
21. [Pros and Cons of Each Method](#21-pros-and-cons-of-each-method)
22. [Interview Questions](#22-interview-questions)
23. [Resources](#23-resources)

---

## 1. Intuition

**Simple statement:** Model what "normal" looks like. Any new data point that is sufficiently different from normal is flagged as an anomaly.

**Real-world use cases:**

| Domain | Anomaly Example |
|--------|----------------|
| **Fraud Detection** | Unusual credit card transaction |
| **Manufacturing** | Defective product on assembly line |
| **Network Security** | Intrusion / cyberattack traffic |
| **Healthcare** | Abnormal patient vital signs |
| **Finance** | Stock price manipulation |
| **IT Operations** | Server CPU spike / memory leak |
| **E-commerce** | Bot traffic / fake reviews |

**Core idea:**

```
Step 1: Learn the distribution of NORMAL data → p(x)
Step 2: For a new point x_test:
          if p(x_test) < ε  →  ANOMALY    🚨
          if p(x_test) ≥ ε  →  NORMAL     ✅
```

**Key challenge:** Anomalies are RARE — often < 0.1% of data. You can't train a standard classifier because you have almost no anomaly examples.

---

## 2. Types of Anomalies

### Point Anomaly
A single data point is anomalous compared to the rest.
```
Normal transactions: $20, $35, $50, $28, $42
Anomaly:            $50,000  ← obvious outlier
```

### Contextual Anomaly
A data point is anomalous in a specific context but not globally.
```
Temperature 35°C:
  → In summer: NORMAL ✅
  → In winter: ANOMALY 🚨  (contextual!)
```

### Collective Anomaly
A group of data points is anomalous together, even if individual points look normal.
```
Individual network packet: looks normal
Same packet repeated 10,000 times in 1 second: ANOMALY (DDoS attack)
```

---

## 3. Approaches Overview

```
Anomaly Detection Methods
├── Statistical / Density-Based
│   ├── Gaussian Density Estimation     ← Andrew Ng's approach
│   ├── Multivariate Gaussian
│   └── Elliptic Envelope (sklearn)
│
├── Proximity-Based
│   ├── Local Outlier Factor (LOF)
│   └── KNN-based outlier detection
│
├── Tree-Based
│   └── Isolation Forest               ← Most popular in practice
│
├── Boundary-Based
│   └── One-Class SVM
│
└── Deep Learning
    ├── Autoencoders
    └── Variational Autoencoders (VAE)
```

---

## 4. Gaussian (Normal) Distribution — The Foundation

### Univariate Gaussian

$$p(x; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$

**Parameters:**
- $\mu$ = mean (center of distribution)
- $\sigma^2$ = variance (spread)
- $\sigma$ = standard deviation

### Parameter Estimation from Data

Given training set $\{x^{(1)}, x^{(2)}, \ldots, x^{(m)}\}$:

$$\mu = \frac{1}{m} \sum_{i=1}^{m} x^{(i)}$$

$$\sigma^2 = \frac{1}{m} \sum_{i=1}^{m} (x^{(i)} - \mu)^2$$

**Note:** Using $\frac{1}{m}$ (MLE) not $\frac{1}{m-1}$ (unbiased) — both work fine for anomaly detection.

### Probability Density Intuition

```
High p(x) → x is near the mean → NORMAL
Low  p(x) → x is far from mean → ANOMALY

         p(x)
          │     ╭───╮
          │    ╱     ╲
          │   ╱       ╲
          │──╱─────────╲──────────── x
         μ-3σ    μ    μ+3σ

Points beyond ±3σ have very low p(x) → likely anomalies
```

---

## 5. Density Estimation Algorithm — Full Math

### The Algorithm (Andrew Ng's Approach)

**Training phase** — fit Gaussian to each feature independently:

For each feature $j = 1, 2, \ldots, n$:

$$\mu_j = \frac{1}{m} \sum_{i=1}^{m} x_j^{(i)}$$

$$\sigma_j^2 = \frac{1}{m} \sum_{i=1}^{m} \left(x_j^{(i)} - \mu_j\right)^2$$

**Prediction phase** — compute joint probability (independence assumption):

$$p(\mathbf{x}) = \prod_{j=1}^{n} p(x_j; \mu_j, \sigma_j^2) = \prod_{j=1}^{n} \frac{1}{\sqrt{2\pi\sigma_j^2}} \exp\left(-\frac{(x_j - \mu_j)^2}{2\sigma_j^2}\right)$$

**Decision:**

$$\hat{y} = \begin{cases} 1 \text{ (anomaly)} & \text{if } p(\mathbf{x}) < \varepsilon \\ 0 \text{ (normal)}  & \text{if } p(\mathbf{x}) \geq \varepsilon \end{cases}$$

### Log-Probability (Numerical Stability)

Products of small numbers → underflow. Use log:

$$\log p(\mathbf{x}) = \sum_{j=1}^{n} \left[ -\frac{1}{2}\log(2\pi\sigma_j^2) - \frac{(x_j - \mu_j)^2}{2\sigma_j^2} \right]$$

### Worked Example

```
Feature: server latency (ms)
Training data (normal servers): [10, 12, 11, 13, 10, 11, 12, 10]

μ  = (10+12+11+13+10+11+12+10) / 8 = 11.125
σ² = mean of (x - μ)² = 0.859
σ  = 0.927

New server latency = 25ms:
p(25) = (1/√(2π × 0.859)) × exp(-(25-11.125)²/(2×0.859))
      = 0.430 × exp(-111.8)
      ≈ 0.0  ← VERY low → ANOMALY 🚨

New server latency = 12ms:
p(12) = 0.430 × exp(-0.384) ≈ 0.289  ← HIGH → NORMAL ✅
```

---

## 6. Multivariate Gaussian Distribution

### Why Multivariate?

Univariate Gaussian models each feature **independently** → misses correlations between features.

```
Example: CPU usage and Memory usage are correlated.
  CPU=0.5, Memory=0.5  → NORMAL (both moderate) ✅
  CPU=0.5, Memory=0.9  → ANOMALY (memory high given CPU is moderate) 🚨

Univariate Gaussian CANNOT catch this — it only looks at each feature alone.
Multivariate Gaussian CAN — it models the joint distribution.
```

### Multivariate Gaussian PDF

$$p(\mathbf{x}; \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{(2\pi)^{n/2} |\boldsymbol{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)$$

**Parameters:**
- $\boldsymbol{\mu} \in \mathbb{R}^n$ = mean vector
- $\boldsymbol{\Sigma} \in \mathbb{R}^{n \times n}$ = covariance matrix
- $|\boldsymbol{\Sigma}|$ = determinant of covariance matrix
- $\boldsymbol{\Sigma}^{-1}$ = precision matrix (inverse of covariance)

### Parameter Estimation

$$\boldsymbol{\mu} = \frac{1}{m} \sum_{i=1}^{m} \mathbf{x}^{(i)}$$

$$\boldsymbol{\Sigma} = \frac{1}{m} \sum_{i=1}^{m} (\mathbf{x}^{(i)} - \boldsymbol{\mu})(\mathbf{x}^{(i)} - \boldsymbol{\mu})^T$$

### Mahalanobis Distance

The exponent in the multivariate Gaussian is the **Mahalanobis distance** squared:

$$D_M(\mathbf{x}) = \sqrt{(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})}$$

- Accounts for correlations between features
- Scale-invariant (unlike Euclidean distance)
- $D_M$ large → point is far from the distribution → anomaly

```python
import numpy as np
from scipy.spatial.distance import mahalanobis

def mahalanobis_distance(x, mu, sigma):
    diff = x - mu
    inv_sigma = np.linalg.inv(sigma)
    return np.sqrt(diff @ inv_sigma @ diff)
```

---

## 7. Univariate vs Multivariate Gaussian

| Property | Univariate (per-feature) | Multivariate |
|----------|--------------------------|--------------|
| **Features modeled** | Independently | Jointly (with correlations) |
| **Parameters** | $\mu_j, \sigma_j^2$ per feature | $\boldsymbol{\mu}$, $\boldsymbol{\Sigma}$ |
| **Num parameters** | $2n$ | $n + n(n+1)/2$ |
| **Correlation captured** | ❌ No | ✅ Yes |
| **Computational cost** | Low | High ($O(n^3)$ for matrix inversion) |
| **Works when $m < n$** | ✅ Yes | ❌ No ($\boldsymbol{\Sigma}$ not invertible) |
| **sklearn equivalent** | `GaussianNB` style | `EllipticEnvelope` |

**Rule of thumb (Andrew Ng):**
- Use **Univariate** when $m \gg n$ is not guaranteed, or as first approach
- Use **Multivariate** when $m \gg n$ (at least $m > 10n$) and correlations matter

---

## 8. Choosing the Threshold (Epsilon)

### The Threshold $\varepsilon$ Problem

$\varepsilon$ is the probability threshold below which a point is flagged as anomalous. Choosing it is critical:

- **Too high $\varepsilon$** → many false positives (normal points flagged as anomalies)
- **Too low $\varepsilon$** → many false negatives (anomalies missed)

### How to Choose $\varepsilon$

**Use a labeled validation set** (even a small one with a few anomalies):

```
Validation set: 1000 normal examples + 20 labeled anomalies

For each candidate ε:
  1. Compute p(x) for all validation examples
  2. Flag as anomaly if p(x) < ε
  3. Compute F1 score against true labels
  4. Pick ε with highest F1

Why F1 (not accuracy)?
  → With 1000 normal + 20 anomaly, predicting all normal = 98% accuracy!
  → F1 penalizes missing the rare anomalies
```

### $\varepsilon$ Selection Algorithm

```python
def select_threshold(y_val, p_val):
    """
    Select best epsilon by maximizing F1 on validation set.
    y_val:  true labels (1=anomaly, 0=normal)
    p_val:  model probabilities p(x) for validation set
    """
    best_eps = 0
    best_f1  = 0

    # Try all values between min and max p(x)
    for eps in np.linspace(p_val.min(), p_val.max(), 1000):
        predictions = (p_val < eps).astype(int)

        tp = np.sum((predictions == 1) & (y_val == 1))
        fp = np.sum((predictions == 1) & (y_val == 0))
        fn = np.sum((predictions == 0) & (y_val == 1))

        precision = tp / (tp + fp + 1e-10)
        recall    = tp / (tp + fn + 1e-10)
        f1        = 2 * precision * recall / (precision + recall + 1e-10)

        if f1 > best_f1:
            best_f1  = f1
            best_eps = eps

    return best_eps, best_f1
```

---

## 9. Isolation Forest — Algorithm & Math

### Intuition

**Simple statement:** Anomalies are easier to isolate than normal points. Randomly partition the data — anomalies get isolated in fewer splits.

```
Normal point:   needs many random splits to isolate → deep in tree → long path
Anomaly point:  isolated quickly by random splits   → near root    → short path
```

### Algorithm

```
BUILD Isolation Tree (iTree):
  1. Randomly select a feature j
  2. Randomly select a split threshold t between [min(x_j), max(x_j)]
  3. Split data: left = x_j ≤ t, right = x_j > t
  4. Recurse until: node has 1 sample OR max tree depth reached

SCORE a point x:
  1. Build forest of T isolation trees on subsamples
  2. For each tree, find path length h(x) to isolate x
  3. Anomaly score = average path length across all trees
```

### Anomaly Score Formula

$$s(x, n) = 2^{-\frac{E[h(x)]}{c(n)}}$$

where:
- $E[h(x)]$ = average path length across all trees
- $c(n)$ = expected path length for a random BST with $n$ nodes (normalizing constant):

$$c(n) = 2H(n-1) - \frac{2(n-1)}{n}$$

$$H(i) = \ln(i) + 0.5772 \quad \text{(Euler-Mascheroni constant)}$$

### Score Interpretation

| Score | Interpretation |
|-------|---------------|
| $s \approx 1$ | Very likely anomaly (short path) |
| $s \approx 0.5$ | Ambiguous — not distinguishable |
| $s \ll 0.5$ | Very likely normal (long path) |

```python
from sklearn.ensemble import IsolationForest

iso = IsolationForest(
    n_estimators=100,      # Number of trees
    max_samples='auto',    # Subsample size (default: min(256, n_samples))
    contamination=0.05,    # Expected proportion of anomalies
    max_features=1.0,      # Features per tree
    random_state=42
)
iso.fit(X_train)

scores      = iso.decision_function(X_test)  # higher = more normal
predictions = iso.predict(X_test)            # 1=normal, -1=anomaly
```

---

## 10. Local Outlier Factor (LOF)

### Intuition

**Simple statement:** A point is an anomaly if its local density is much lower than its neighbors' densities. It compares density locally — not globally.

```
Dense cluster: points close together → high local density
Anomaly:       isolated point         → low local density compared to neighbors
```

### Mathematics

**Step 1: k-distance** — distance to k-th nearest neighbor of point $p$:

$$k\text{-dist}(p) = d(p, o_k)$$

**Step 2: Reachability distance** — smoothed distance:

$$\text{reach-dist}_k(p, o) = \max\{k\text{-dist}(o),\ d(p, o)\}$$ 

**Step 3: Local Reachability Density (LRD)**:

$$\text{lrd}_k(p) = \frac{|N_k(p)|}{\sum_{o \in N_k(p)} \text{reach-dist}_k(p, o)}$$

**Step 4: Local Outlier Factor**:

$$\text{LOF}_k(p) = \frac{\sum_{o \in N_k(p)} \frac{\text{lrd}_k(o)}{\text{lrd}_k(p)}}{|N_k(p)|}$$

### LOF Score Interpretation

| LOF Score | Meaning |
|-----------|---------|
| $\approx 1$ | Point has similar density to neighbors → NORMAL |
| $\gg 1$ | Point has much lower density than neighbors → ANOMALY |

```python
from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(
    n_neighbors=20,        # k — number of neighbors
    algorithm='auto',      # 'ball_tree', 'kd_tree', 'brute'
    leaf_size=30,
    metric='minkowski',
    p=2,
    contamination=0.05,    # Expected fraction of outliers
    novelty=False          # True for predict on new data
)

# novelty=False: fit_predict only (transductive)
labels = lof.fit_predict(X)          # 1=normal, -1=anomaly
scores = lof.negative_outlier_factor_ # more negative = more anomalous

# novelty=True: can predict on new data (inductive)
lof_novelty = LocalOutlierFactor(novelty=True, n_neighbors=20)
lof_novelty.fit(X_train)
labels_test = lof_novelty.predict(X_test)
```

---

## 11. One-Class SVM

### Intuition

**Simple statement:** Learn a tight boundary around normal training data in a high-dimensional feature space. Points outside this boundary are anomalies.

### Mathematics (Nu-SVC formulation)

Solve the optimization problem:

$$\min_{w, \xi, \rho} \quad \frac{1}{2}\|w\|^2 - \rho + \frac{1}{\nu m} \sum_{i=1}^{m} \xi_i$$

Subject to:
$$w \cdot \phi(x^{(i)}) \geq \rho - \xi_i, \quad \xi_i \geq 0$$

where:
- $\phi(x)$ = kernel mapping to feature space
- $\rho$ = offset (decision boundary distance from origin)
- $\xi_i$ = slack variables (allow some points outside boundary)
- $\nu \in (0, 1]$ = upper bound on fraction of outliers (hyperparameter)

**Decision function:**

$$f(x) = \text{sign}(w \cdot \phi(x) - \rho)$$

- $f(x) = +1$ → NORMAL (inside boundary)
- $f(x) = -1$ → ANOMALY (outside boundary)

```python
from sklearn.svm import OneClassSVM

ocsvm = OneClassSVM(
    kernel='rbf',     # 'linear', 'poly', 'rbf', 'sigmoid'
    nu=0.05,          # upper bound on fraction of anomalies
    gamma='scale',    # kernel coefficient
    tol=1e-3,
    shrinking=True
)
ocsvm.fit(X_train)
predictions = ocsvm.predict(X_test)   # 1=normal, -1=anomaly
scores      = ocsvm.decision_function(X_test)  # distance from boundary
```

**Note:** One-Class SVM is sensitive to scaling — always use `StandardScaler`.

---

## 12. Elliptic Envelope

### Intuition

Fits a multivariate Gaussian to the data and flags points that are far from the center (high Mahalanobis distance) as anomalies. Robust version handles outliers during fitting.

### Math

Uses **Minimum Covariance Determinant (MCD)** estimator — finds the subset of $h$ observations whose covariance matrix has the lowest determinant:

$$(\hat{\boldsymbol{\mu}}_{\text{MCD}}, \hat{\boldsymbol{\Sigma}}_{\text{MCD}}) = \arg\min_{|H|=h} \det\left(\boldsymbol{\Sigma}(H)\right)$$

Decision based on Mahalanobis distance:

$$D_M^2(\mathbf{x}) = (\mathbf{x} - \hat{\boldsymbol{\mu}})^T \hat{\boldsymbol{\Sigma}}^{-1} (\mathbf{x} - \hat{\boldsymbol{\mu}}) \leq \chi^2_{n, \alpha}$$

```python
from sklearn.covariance import EllipticEnvelope

ee = EllipticEnvelope(
    contamination=0.05,    # Expected fraction of outliers
    support_fraction=None, # Fraction used for MCD (default: (n_features+1)/(2*n_samples))
    random_state=42
)
ee.fit(X_train)
predictions    = ee.predict(X_test)           # 1=normal, -1=anomaly
mahal_distances = ee.mahalanobis(X_test)      # Mahalanobis distances
scores         = ee.decision_function(X_test) # shifted Mahalanobis
```

---

## 13. Bias-Variance & Evaluation Challenge

### The Skewed Class Problem

```
Dataset: 10,000 transactions
  Normal:  9,950  (99.5%)
  Fraud:      50  (0.5%)

Naive classifier (predict all normal):
  Accuracy = 99.5%  ← looks great but useless!
  Recall   = 0%     ← catches zero fraud
```

**Always use F1, Precision-Recall, and ROC-AUC — never raw accuracy for anomaly detection.**

### No Gradient Descent

| Algorithm | Optimization | Method |
|-----------|-------------|--------|
| Gaussian Density | Closed-form MLE | Compute mean/variance analytically |
| Isolation Forest | Random partitioning | No optimization at all |
| LOF | Density computation | Distance-based, no optimization |
| One-Class SVM | Quadratic Programming | Convex optimization (not gradient descent) |
| Elliptic Envelope | MCD estimation | Iterative reweighted least squares |
| Autoencoder | Backpropagation | **Gradient descent** (DL method) |

---

## 14. Hyperparameters — Complete Guide

### Isolation Forest

```python
IsolationForest(
    n_estimators=100,       # Number of trees. More = stable scores. Default 100.
    max_samples='auto',     # Subsample per tree. 'auto'=min(256,n). int or float.
    contamination='auto',   # Fraction of outliers. 'auto' or float [0, 0.5].
                            # MOST IMPORTANT: set to expected anomaly rate.
    max_features=1.0,       # Features per tree. 1.0=all, 'sqrt', float.
    bootstrap=False,        # Sample with replacement. False=without.
    n_jobs=-1,              # Parallel jobs.
    random_state=42,        # Reproducibility.
    warm_start=False,       # Add trees to existing forest.
)
```

| Param | Tune When | Effect |
|-------|-----------|--------|
| `n_estimators` | Always | More trees → more stable (100-500) |
| `max_samples` | Large datasets | Smaller → faster, may miss patterns |
| `contamination` | Always | Must match actual anomaly rate in data |
| `max_features` | High-dim data | Lower → more randomness |

### Local Outlier Factor

```python
LocalOutlierFactor(
    n_neighbors=20,         # k neighbors. MOST IMPORTANT. Try 10-50.
    algorithm='auto',       # Search algorithm: 'ball_tree','kd_tree','brute'
    leaf_size=30,           # For ball/kd tree
    metric='minkowski',     # Distance metric
    p=2,                    # Minkowski p (2=Euclidean)
    metric_params=None,
    contamination=0.05,     # Expected outlier fraction
    novelty=False,          # False=transductive, True=inductive
    n_jobs=-1,
)
```

| Param | Effect |
|-------|--------|
| `n_neighbors` | Too small → noisy; too large → misses local patterns |
| `contamination` | Threshold for decision; set from domain knowledge |
| `novelty` | Set True to use predict() on new data |

### One-Class SVM

```python
OneClassSVM(
    kernel='rbf',           # Kernel: 'linear','poly','rbf','sigmoid'
    degree=3,               # Degree for poly kernel
    gamma='scale',          # Kernel coeff: 'scale'=1/(n_features*X.var()), 'auto'=1/n_features
    coef0=0.0,              # Independent term in poly/sigmoid
    tol=1e-3,               # Stopping tolerance
    nu=0.5,                 # MOST IMPORTANT: upper bound on outlier fraction [0,1]
    shrinking=True,         # Use shrinking heuristic
    cache_size=200,         # Cache size (MB)
    max_iter=-1,            # Max iterations (-1=unlimited)
)
```

| Param | Effect |
|-------|--------|
| `nu` | Fraction of outliers. Set from domain knowledge. |
| `kernel` | `rbf` works best in most cases |
| `gamma` | Controls RBF width. Use `'scale'` as default. |

### Elliptic Envelope

```python
EllipticEnvelope(
    store_precision=True,   # Store precision matrix
    assume_centered=False,  # True if data is centered
    support_fraction=None,  # Fraction for MCD (default: (n+1)/(2n))
    contamination=0.1,      # Expected fraction of outliers
    random_state=42,
)
```

### Gaussian Density (Manual)

```python
# No sklearn class — tune epsilon (ε) via F1 on validation set
# Parameters:
#   var_smoothing: add small value to variance to avoid division by zero
var_smoothing = 1e-9   # (same concept as GaussianNB)
epsilon       = ?      # tune via select_threshold() using F1 score
```

---

## 15. Python from Scratch — Gaussian Anomaly Detection

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

class GaussianAnomalyDetector:
    """
    Anomaly Detection using Gaussian Density Estimation.
    Implements Andrew Ng's ML Specialization approach.

    Training:   O(m * n) — compute mean/variance
    Prediction: O(n)     — compute product of Gaussians
    """

    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing
        self.mu_    = None
        self.sigma2_ = None
        self.epsilon_ = None

    # ── Fit: estimate parameters ─────────────────────────────
    def fit(self, X):
        """
        Estimate mu and sigma^2 for each feature using MLE.
        X: (m, n) — only NORMAL training examples
        """
        X = np.array(X)
        self.mu_     = X.mean(axis=0)                        # (n,)
        self.sigma2_ = X.var(axis=0) + self.var_smoothing    # (n,)
        return self

    # ── Gaussian PDF per feature ─────────────────────────────
    def _gaussian_pdf(self, X):
        """
        Compute p(x_j | mu_j, sigma2_j) for each feature.
        Returns log-probability for numerical stability.
        """
        X = np.array(X)                                      # (m, n)
        log_coeff = -0.5 * np.log(2 * np.pi * self.sigma2_) # (n,)
        log_exp   = -0.5 * ((X - self.mu_)**2) / self.sigma2_ # (m, n)
        return log_coeff + log_exp                           # (m, n)

    # ── Joint log-probability (independence assumption) ──────
    def log_probability(self, X):
        """
        log p(x) = sum_j log p(x_j | mu_j, sigma2_j)
        Returns log-probability for each sample.
        """
        return self._gaussian_pdf(X).sum(axis=1)             # (m,)

    def probability(self, X):
        """p(x) = exp(log p(x))"""
        return np.exp(self.log_probability(X))

    # ── Threshold selection via F1 ───────────────────────────
    def select_threshold(self, X_val, y_val):
        """
        Find optimal epsilon by maximizing F1 on validation set.
        y_val: 1=anomaly, 0=normal
        """
        p_val    = self.probability(X_val)
        best_eps = 0
        best_f1  = 0

        epsilons = np.linspace(p_val.min(), p_val.max(), 1000)

        for eps in epsilons:
            preds = (p_val < eps).astype(int)

            tp = np.sum((preds == 1) & (y_val == 1))
            fp = np.sum((preds == 1) & (y_val == 0))
            fn = np.sum((preds == 0) & (y_val == 1))

            prec = tp / (tp + fp + 1e-10)
            rec  = tp / (tp + fn + 1e-10)
            f1   = 2 * prec * rec / (prec + rec + 1e-10)

            if f1 > best_f1:
                best_f1  = f1
                best_eps = eps

        self.epsilon_ = best_eps
        return best_eps, best_f1

    # ── Predict ──────────────────────────────────────────────
    def predict(self, X, epsilon=None):
        """
        Returns 1 for anomaly, 0 for normal.
        Uses self.epsilon_ if epsilon not provided.
        """
        eps = epsilon if epsilon is not None else self.epsilon_
        if eps is None:
            raise ValueError("Call select_threshold() first or provide epsilon.")
        p = self.probability(X)
        return (p < eps).astype(int)

    def score(self, X, y):
        return f1_score(y, self.predict(X))


# ════════════════════════════════════════════════════════════
# Demo: Server Monitoring Anomaly Detection
# ════════════════════════════════════════════════════════════
np.random.seed(42)

# Generate synthetic server data
n_normal  = 1000
n_anomaly = 20

# Normal: CPU~[0.4-0.8], Memory~[0.3-0.7]  (correlated)
X_normal = np.random.multivariate_normal(
    mean=[0.6, 0.5],
    cov=[[0.01, 0.008], [0.008, 0.01]],
    size=n_normal
)

# Anomalies: unusual combinations
X_anomaly = np.array([
    [0.95, 0.95], [0.98, 0.2], [0.1, 0.98],
    [0.99, 0.85], [0.05, 0.05], [0.92, 0.91],
    *np.random.uniform(0.85, 1.0, size=(14, 2))
])

# Split normal data: train / val / test
X_train = X_normal[:800]
X_val   = np.vstack([X_normal[800:], X_anomaly[:10]])
y_val   = np.array([0]*200 + [1]*10)
X_test  = np.vstack([X_normal[800:], X_anomaly[10:]])
y_test  = np.array([0]*200 + [1]*10)

# Fit on NORMAL training data only
detector = GaussianAnomalyDetector(var_smoothing=1e-9)
detector.fit(X_train)

# Find optimal threshold
best_eps, best_f1 = detector.select_threshold(X_val, y_val)
print(f"Optimal epsilon: {best_eps:.6f}")
print(f"Best Val F1:     {best_f1:.4f}")

# Evaluate on test set
y_pred = detector.predict(X_test)
print(f"\nTest F1:        {f1_score(y_test, y_pred):.4f}")
print(f"Test Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Test Recall:    {recall_score(y_test, y_pred):.4f}")

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: scatter plot with anomaly labels
ax1.scatter(X_test[y_test==0, 0], X_test[y_test==0, 1],
            c='steelblue', alpha=0.5, label='Normal')
ax1.scatter(X_test[y_test==1, 0], X_test[y_test==1, 1],
            c='red', s=100, marker='x', label='True Anomaly')
ax1.set_title('Ground Truth')
ax1.legend()

# Right: predicted anomalies
ax2.scatter(X_test[y_pred==0, 0], X_test[y_pred==0, 1],
            c='steelblue', alpha=0.5, label='Predicted Normal')
ax2.scatter(X_test[y_pred==1, 0], X_test[y_pred==1, 1],
            c='red', s=100, marker='x', label='Predicted Anomaly')
ax2.set_title(f'Predictions (ε={best_eps:.4f})')
ax2.legend()

plt.suptitle('Gaussian Anomaly Detection — Server Monitoring', fontsize=13)
plt.tight_layout()
plt.show()
```

### Multivariate Gaussian from Scratch

```python
class MultivariateGaussianAnomalyDetector:
    """
    Multivariate Gaussian Anomaly Detector.
    Captures correlations between features.
    Requires m >> n (at least m > 10*n).
    """

    def __init__(self, epsilon=None):
        self.mu_      = None
        self.sigma_   = None  # Covariance matrix
        self.epsilon_ = epsilon

    def fit(self, X):
        X = np.array(X)
        m = X.shape[0]
        self.mu_    = X.mean(axis=0)
        diff        = X - self.mu_
        self.sigma_ = (diff.T @ diff) / m   # (n, n) covariance matrix
        return self

    def _log_pdf(self, X):
        """Multivariate Gaussian log-PDF."""
        X    = np.array(X)
        n    = X.shape[1]
        diff = X - self.mu_                          # (m, n)
        inv_sigma  = np.linalg.inv(self.sigma_)      # (n, n)
        sign, logdet = np.linalg.slogdet(self.sigma_)

        # Mahalanobis distance squared for each sample
        mahal  = np.einsum('mi,ij,mj->m', diff, inv_sigma, diff)  # (m,)

        log_p  = -0.5 * (n * np.log(2 * np.pi) + logdet + mahal)
        return log_p

    def probability(self, X):
        return np.exp(self._log_pdf(np.array(X).reshape(-1, self.mu_.shape[0])))

    def predict(self, X):
        p = self.probability(X)
        return (p < self.epsilon_).astype(int)

    def mahalanobis_distance(self, X):
        X    = np.array(X)
        diff = X - self.mu_
        inv_sigma = np.linalg.inv(self.sigma_)
        return np.sqrt(np.einsum('mi,ij,mj->m', diff, inv_sigma, diff))


# Test
mv_detector = MultivariateGaussianAnomalyDetector(epsilon=1e-10)
mv_detector.fit(X_train)
mahal = mv_detector.mahalanobis_distance(X_test)
print(f"Mahalanobis distances (first 5 normal): {mahal[:5].round(2)}")
print(f"Mahalanobis distances (anomalies):      {mahal[-5:].round(2)}")
```

---

## 16. Scikit-learn Implementation

### All Four Methods Side by Side

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from sklearn.metrics import classification_report

# ── Generate Data ──────────────────────────────────────────────
np.random.seed(42)
X_normal, _  = make_blobs(n_samples=300, centers=1,
                           cluster_std=0.5, random_state=42)
X_anomaly    = np.random.uniform(-4, 4, size=(15, 2))
X            = np.vstack([X_normal, X_anomaly])
y_true       = np.array([1]*300 + [-1]*15)   # sklearn: 1=normal, -1=anomaly

# ── Preprocessing ─────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Define Models ──────────────────────────────────────────────
models = {
    'Isolation Forest': IsolationForest(
        contamination=0.05, n_estimators=100, random_state=42
    ),
    'Local Outlier Factor': LocalOutlierFactor(
        n_neighbors=20, contamination=0.05
    ),
    'One-Class SVM': OneClassSVM(
        kernel='rbf', nu=0.05, gamma='scale'
    ),
    'Elliptic Envelope': EllipticEnvelope(
        contamination=0.05, random_state=42
    ),
}

# ── Fit & Predict ──────────────────────────────────────────────
results = {}
for name, model in models.items():
    if name == 'Local Outlier Factor':
        preds = model.fit_predict(X_scaled)        # transductive
    else:
        model.fit(X_scaled)
        preds = model.predict(X_scaled)
    results[name] = preds
    print(f"\n{name}:")
    print(classification_report(y_true, preds,
                                 target_names=['Anomaly (-1)', 'Normal (1)']))

# ── Visual Comparison ──────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

for ax, (name, preds) in zip(axes, results.items()):
    mask_normal  = preds == 1
    mask_anomaly = preds == -1
    ax.scatter(X_scaled[mask_normal,  0], X_scaled[mask_normal,  1],
               c='steelblue', alpha=0.5, label='Normal', s=20)
    ax.scatter(X_scaled[mask_anomaly, 0], X_scaled[mask_anomaly, 1],
               c='red', s=100, marker='x', label='Anomaly', linewidths=2)
    ax.set_title(name, fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle('Anomaly Detection Methods Comparison', fontsize=14)
plt.tight_layout()
plt.show()
```

### Real-World: Credit Card Fraud Detection

```python
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                              precision_recall_curve, average_precision_score)

# Load dataset (use: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
# df = pd.read_csv('creditcard.csv')
# X  = df.drop('Class', axis=1).values
# y  = df['Class'].values   # 0=normal, 1=fraud

# Simulate for demo
np.random.seed(42)
X_normal  = np.random.randn(9850, 10)
X_fraud   = np.random.randn(150, 10) * 3 + 4
X = np.vstack([X_normal, X_fraud])
y = np.array([0]*9850 + [1]*150)   # 0=normal, 1=fraud

# Convert to sklearn convention: 1=normal, -1=anomaly
y_sklearn = np.where(y == 0, 1, -1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_sklearn, test_size=0.2, random_state=42, stratify=y_sklearn
)

# Scale
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Isolation Forest
iso = IsolationForest(
    contamination=150/10000,   # actual fraud rate
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
iso.fit(X_train)
y_pred   = iso.predict(X_test)
y_scores = iso.decision_function(X_test)  # higher = more normal

# Evaluate
print(classification_report(
    y_test, y_pred,
    target_names=['Fraud (-1)', 'Normal (1)']
))

# Precision-Recall curve (best metric for imbalanced)
# Convert to binary: fraud=1, normal=0
y_test_bin  = (y_test == -1).astype(int)
y_score_bin = -y_scores  # flip: higher = more anomalous

precision, recall, thresholds = precision_recall_curve(y_test_bin, y_score_bin)
ap = average_precision_score(y_test_bin, y_score_bin)

plt.figure(figsize=(8, 5))
plt.plot(recall, precision, linewidth=2, label=f'AP={ap:.3f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve — Fraud Detection')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 17. Full Pipeline with Best Practices

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score
import numpy as np

# ── Pipeline ─────────────────────────────────────────────────
# RobustScaler: better than StandardScaler when data has outliers
pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('detector', IsolationForest(random_state=42))
])

# ── Custom scorer (F1 for anomaly detection) ─────────────────
def anomaly_f1(y_true, y_pred):
    # sklearn uses 1=normal, -1=anomaly
    # convert -1 → 1 (anomaly=positive class) for F1
    y_true_bin = (y_true == -1).astype(int)
    y_pred_bin = (y_pred == -1).astype(int)
    return f1_score(y_true_bin, y_pred_bin, zero_division=0)

scorer = make_scorer(anomaly_f1)

# ── Grid Search ───────────────────────────────────────────────
param_grid = {
    'scaler': [StandardScaler(), RobustScaler()],
    'detector__n_estimators':  [50, 100, 200],
    'detector__max_samples':   ['auto', 128, 256],
    'detector__contamination': [0.01, 0.05, 0.1],
    'detector__max_features':  [1.0, 0.5, 'sqrt'],
}

grid = GridSearchCV(
    pipeline, param_grid,
    scoring=scorer, cv=5,
    n_jobs=-1, verbose=1,
    refit=True
)
grid.fit(X_train, y_train)

print(f"Best params: {grid.best_params_}")
print(f"Best F1:     {grid.best_score_:.4f}")
print(f"Test F1:     {anomaly_f1(y_test, grid.predict(X_test)):.4f}")
```

---

## 18. Anomaly Detection vs Supervised Learning

| Factor | Anomaly Detection | Supervised Learning |
|--------|------------------|---------------------|
| **Labeled anomalies** | Very few or none needed | Many needed |
| **Class balance** | Extremely imbalanced | Works with balanced |
| **New anomaly types** | Can detect unseen types | Only detects trained types |
| **Training data** | Only normal data needed | Both classes needed |
| **Use case** | Unknown/novel anomalies | Known fraud patterns |

### When to Use Anomaly Detection vs Supervised Classification

```
Use ANOMALY DETECTION when:
  ✅ Very few anomaly examples (< 20 total)
  ✅ Many different types of anomalies (hard to enumerate)
  ✅ Future anomalies may be different from historical ones
  ✅ Examples: novel cyberattacks, new fraud schemes

Use SUPERVISED CLASSIFICATION when:
  ✅ Enough positive examples (> 50-100 per class)
  ✅ Anomaly types are well-defined and stable
  ✅ Examples: known malware signatures, specific fraud patterns
```

---

## 19. Feature Engineering for Anomaly Detection

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ── Time-based features (for time series anomaly detection) ──
def add_time_features(df, timestamp_col):
    df['hour']        = df[timestamp_col].dt.hour
    df['day_of_week'] = df[timestamp_col].dt.dayofweek
    df['is_weekend']  = (df['day_of_week'] >= 5).astype(int)
    df['month']       = df[timestamp_col].dt.month
    return df

# ── Rolling statistics (detect sudden changes) ───────────────
def add_rolling_features(df, col, windows=[5, 10, 30]):
    for w in windows:
        df[f'{col}_rolling_mean_{w}'] = df[col].rolling(w).mean()
        df[f'{col}_rolling_std_{w}']  = df[col].rolling(w).std()
        df[f'{col}_rolling_max_{w}']  = df[col].rolling(w).max()
        # Z-score relative to rolling window
        df[f'{col}_zscore_{w}'] = (
            (df[col] - df[f'{col}_rolling_mean_{w}']) /
            (df[f'{col}_rolling_std_{w}'] + 1e-8)
        )
    return df

# ── Ratio features (for fraud detection) ─────────────────────
def add_ratio_features(df):
    # Transaction amount vs user's historical average
    user_avg = df.groupby('user_id')['amount'].transform('mean')
    df['amount_vs_user_avg'] = df['amount'] / (user_avg + 1e-8)

    # Transaction frequency per hour
    df['tx_count_1h'] = df.groupby('user_id')['timestamp'] \
                          .transform(lambda x: x.diff().dt.seconds < 3600)
    return df

# ── Log-transform right-skewed features ──────────────────────
# Many financial/network features are heavily skewed
def log_transform(X, features):
    X = X.copy()
    for f in features:
        X[f] = np.log1p(X[f].clip(lower=0))
    return X
```

---

## 20. Evaluation Metrics

```python
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, roc_curve
)
import matplotlib.pyplot as plt

# Convert sklearn convention (-1/1) to binary (1/0) for metrics
y_true_bin = (y_test  == -1).astype(int)   # 1=anomaly
y_pred_bin = (y_pred  == -1).astype(int)   # 1=anomaly
y_scores   = -iso.decision_function(X_test) # higher = more anomalous

# ── Core Metrics ──────────────────────────────────────────────
print(f"Precision:         {precision_score(y_true_bin, y_pred_bin):.4f}")
print(f"Recall:            {recall_score(y_true_bin, y_pred_bin):.4f}")
print(f"F1 Score:          {f1_score(y_true_bin, y_pred_bin):.4f}")
print(f"ROC AUC:           {roc_auc_score(y_true_bin, y_scores):.4f}")
print(f"Average Precision: {average_precision_score(y_true_bin, y_scores):.4f}")

# ── Confusion Matrix ──────────────────────────────────────────
cm = confusion_matrix(y_true_bin, y_pred_bin)
print(f"\nTN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")

# ── Precision-Recall Curve (use this for imbalanced data!) ────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

precision, recall, _ = precision_recall_curve(y_true_bin, y_scores)
ap = average_precision_score(y_true_bin, y_scores)
ax1.plot(recall, precision, linewidth=2, color='darkorange',
         label=f'AP = {ap:.3f}')
ax1.set_xlabel('Recall (Anomaly Detected / Total Anomalies)')
ax1.set_ylabel('Precision (Correct Anomaly / Flagged)')
ax1.set_title('Precision-Recall Curve')
ax1.legend()
ax1.grid(True, alpha=0.3)

# ── ROC Curve ─────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_true_bin, y_scores)
auc = roc_auc_score(y_true_bin, y_scores)
ax2.plot(fpr, tpr, linewidth=2, color='steelblue',
         label=f'AUC = {auc:.3f}')
ax2.plot([0,1],[0,1],'k--')
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC Curve')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ── Metric Choice Guide ──────────────────────────────────────
# Metric       | Use When
# -------------|----------------------------------------------
# Precision    | Cost of false alarm is high (don't cry wolf)
# Recall       | Cost of missing anomaly is high (fraud/security)
# F1           | Balance between precision and recall
# ROC-AUC      | Overall ranking quality
# Avg Precision| Best for severely imbalanced classes
# Accuracy     | NEVER use for anomaly detection!
```

---

## 21. Pros and Cons of Each Method

| Method | Pros | Cons | Best For |
|--------|------|------|---------|
| **Gaussian Density** | Simple, interpretable, fast | Assumes Gaussian, misses correlations | Low-dim continuous data |
| **Multivariate Gaussian** | Captures correlations | Needs m >> n, fails high-dim | Low-dim with correlated features |
| **Isolation Forest** | Fast, scalable, robust | Black-box, needs contamination param | General purpose, large datasets |
| **LOF** | Local context, no distributional assumption | Slow O(n²), needs k tuning | Varying density clusters |
| **One-Class SVM** | Kernel tricks, non-linear boundary | Slow, sensitive to scaling/hyperparams | Small-medium datasets |
| **Elliptic Envelope** | Robust to outliers during fitting | Assumes elliptic distribution, needs m >> n | Gaussian-like distributions |

---

## 22. Interview Questions

**Q1: What is anomaly detection and how does it differ from classification?**
> Anomaly detection identifies rare deviations from normal behavior, typically with few or no labeled anomaly examples. Classification requires labeled examples of both classes. Anomaly detection learns only from normal data.

**Q2: Explain Andrew Ng's Gaussian density estimation approach.**
> Fit a Gaussian distribution to each feature using MLE (compute mean and variance). The joint probability is the product of individual Gaussians (independence assumption). Flag a point as anomaly if $p(x) < \varepsilon$. Choose $\varepsilon$ by maximizing F1 on a small labeled validation set.

**Q3: Why use F1 score instead of accuracy for anomaly detection?**
> Anomalies are extremely rare. A classifier predicting "all normal" gets 99%+ accuracy but catches zero anomalies. F1 balances precision and recall, penalizing both false positives and false negatives equally.

**Q4: When would you use multivariate Gaussian over univariate?**
> When features are correlated and you have $m \gg n$ (at least 10× more samples than features). Multivariate Gaussian captures feature correlations via the covariance matrix. Univariate treats each feature independently.

**Q5: How does Isolation Forest work?**
> Random feature selection + random threshold creates partitions. Anomalies are isolated in fewer splits (shorter path length in the tree). The anomaly score is the normalized average path length across many trees. Short path = more anomalous.

**Q6: What is the contamination parameter and why is it important?**
> It specifies the expected proportion of anomalies in the dataset. Used to set the decision threshold automatically. Setting it too high → too many false positives. Too low → misses anomalies. Should reflect actual domain knowledge about anomaly rate.

**Q7: What is the difference between LOF `novelty=False` and `novelty=True`?**
> `novelty=False` (default): transductive — only fit_predict() on training data, can't predict on new test data. `novelty=True`: inductive — fit() on training, then predict() on new data. Use `novelty=True` for production deployment.

**Q8: Does anomaly detection use gradient descent?**
> Gaussian density estimation: No — closed-form MLE. Isolation Forest: No — random partitioning. LOF: No — distance computation. One-Class SVM: No — quadratic programming. Autoencoders: Yes — gradient descent to minimize reconstruction error.

**Q9: How do you handle the threshold epsilon in Gaussian anomaly detection?**
> Use a small labeled validation set containing some anomaly examples. Try many values of ε from $p_{\min}$ to $p_{\max}$. For each ε, compute F1 score. Select the ε that maximizes F1.

**Q10: What feature transformations help anomaly detection?**
> Log-transform skewed features (financial amounts, network bytes). Add rolling statistics for time-series data. Create ratio features (amount vs user average). Z-score normalization for Gaussian methods. These make distributions more Gaussian and highlight deviations.

---

## 23. Resources

### 📘 Andrew Ng — ML Specialization (Coursera)
- **Course 3, Week 1**: Anomaly Detection — full dedicated week
  - Density estimation with Gaussian distribution
  - Parameter estimation (MLE for μ and σ²)
  - Algorithm for anomaly detection
  - Developing and evaluating an anomaly detection system
  - Anomaly detection vs supervised learning
  - Choosing features for anomaly detection
  - Multivariate Gaussian distribution
- 🔗 https://www.coursera.org/specializations/machine-learning-introduction

### 📗 Hands-On Machine Learning (Aurélien Géron)
- **Chapter 9**: Unsupervised Learning Techniques
  - Gaussian Mixtures for anomaly detection
  - Isolation Forest
  - PCA-based anomaly detection
  - Autoencoders for anomaly detection
- **Chapter 3**: Classification — imbalanced classes, precision/recall tradeoff
- 🔗 https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/

### 🎬 StatQuest with Josh Starmer (YouTube)
- **"Anomaly Detection, Clearly Explained"** → Core concepts
- **"Isolation Forest, Clearly Explained"** → Isolation Forest algorithm
- **"Local Outlier Factor, Clearly Explained"** → LOF intuition and math
- **"Normal Distribution, Clearly Explained"** → Gaussian foundation
- 🔗 https://www.youtube.com/@statquest

### 📙 Introduction to Statistical Learning (ISLP — Python Edition)
- **Chapter 12.4**: Outlier Detection
- **Chapter 4**: Classification — density estimation basics
- **Chapter 6**: Regularization — related to model complexity
- Free PDF: 🔗 https://www.statlearning.com/

### 📜 Scikit-learn Documentation
- Isolation Forest: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
- LOF: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html
- One-Class SVM: https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html
- Elliptic Envelope: https://scikit-learn.org/stable/modules/generated/sklearn.covariance.EllipticEnvelope.html
- Novelty & Outlier Detection Guide: https://scikit-learn.org/stable/modules/outlier_detection.html
- Comparing Methods: https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_anomaly_comparison.html

### 📺 Additional Resources
- **Original Isolation Forest paper**: Liu et al. (2008) — "Isolation Forest"
- **LOF paper**: Breunig et al. (2000) — "LOF: Identifying Density-Based Local Outliers"
- **Kaggle**: Credit Card Fraud Detection dataset (best real-world practice)
- **PyOD library**: https://pyod.readthedocs.io — 40+ anomaly detection algorithms

---

## 🎯 Quick Cheat Sheet

```
Core idea:     Learn p(x) from NORMAL data. Flag x as anomaly if p(x) < ε.

Methods:
  Gaussian       → p(x) = Π Gaussian(x_j; μ_j, σ²_j)   closed-form, fast
  Multivariate   → p(x) = multivariate Gaussian          captures correlations
  Isolation Forest → short path = anomaly                fast, scalable ✅
  LOF            → low local density = anomaly           local context
  One-Class SVM  → outside boundary = anomaly            kernel tricks
  Elliptic Env   → high Mahalanobis distance = anomaly   robust MCD

No gradient descent in: Gaussian, Isolation Forest, LOF, One-Class SVM
Gradient descent in:    Autoencoders (DL approach only)

Threshold ε:   Tune by maximizing F1 on small labeled validation set
               NEVER use accuracy (skewed classes!)

Evaluation:    F1 Score, Precision, Recall, PR-AUC, ROC-AUC
               Never: plain accuracy

Contamination: Set to actual expected anomaly rate in data
               Most important hyperparameter across all sklearn methods

Scaling:       Required for: One-Class SVM, LOF, Elliptic Envelope
               Not required for: Isolation Forest (tree-based)
               Use RobustScaler when data contains outliers

Anomaly Detection vs Classification:
  < 20 anomaly examples    → Anomaly Detection
  Unknown future anomalies → Anomaly Detection
  100+ labeled anomalies   → Supervised Classification
```

---

*Notes compiled from: Andrew Ng ML Specialization (Course 3, Week 1) · Hands-On ML Ch.9 (Géron) · StatQuest Anomaly Detection series · ISLP Ch.12 (James et al.) · Scikit-learn Outlier Detection Guide*
