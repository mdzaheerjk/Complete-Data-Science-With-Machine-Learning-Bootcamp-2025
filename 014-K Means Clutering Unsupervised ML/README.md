# 🔵 K-Means Clustering — Complete Job-Ready Notes

> **References:** Andrew Ng ML Specialization (Coursera) · Hands-On ML with Scikit-Learn (Aurélien Géron, 3rd ed.) · StatQuest with Josh Starmer (YouTube) · Scikit-Learn Documentation · Introduction to Statistical Learning with Python (ISLP, James et al.)

---

## 📚 Table of Contents

1. [What is K-Means? (Simple Statement)](#1-what-is-k-means-simple-statement)
2. [Why K-Means? Motivation](#2-why-k-means-motivation)
3. [The Math Behind K-Means](#3-the-math-behind-k-means)
4. [Step-by-Step K-Means Algorithm](#4-step-by-step-k-means-algorithm)
5. [K-Means as Optimization (Gradient Descent View)](#5-k-means-as-optimization-gradient-descent-view)
6. [K-Means from Scratch in Python (NumPy)](#6-k-means-from-scratch-in-python-numpy)
7. [K-Means with Scikit-Learn](#7-k-means-with-scikit-learn)
8. [Hyperparameters & Tuning](#8-hyperparameters--tuning)
9. [Choosing K — The Right Number of Clusters](#9-choosing-k--the-right-number-of-clusters)
10. [Evaluation Metrics](#10-evaluation-metrics)
11. [Variants of K-Means](#11-variants-of-k-means)
12. [K-Means in ML Pipelines](#12-k-means-in-ml-pipelines)
13. [Limitations & When NOT to Use K-Means](#13-limitations--when-not-to-use-k-means)
14. [Common Interview Questions](#14-common-interview-questions)
15. [Resources](#15-resources)

---

## 1. What is K-Means? (Simple Statement)

> **K-Means partitions `n` data points into `K` groups (clusters) by minimizing the total squared distance from each point to its cluster's center (centroid). Points in the same cluster are similar; points in different clusters are dissimilar.**

**Analogy (StatQuest style):**
Imagine 100 students in a large hall. You want to form `K` study groups so each student is as close as possible to their group leader. K-Means finds the best positions for `K` group leaders so the total distance walked by all students is minimized.

- **Type:** Unsupervised learning (no labels needed)
- **Input:** Data matrix `X` of shape `(n_samples, n_features)` + integer `K`
- **Output:** Cluster labels `[0, 1, ..., K-1]` for each point + `K` centroid vectors
- **Key idea:** Iteratively assign points to nearest centroid, then recompute centroids, until convergence.

---

## 2. Why K-Means? Motivation

| Use Case | Example |
|---|---|
| Customer segmentation | Group customers by purchase behavior |
| Image compression | Reduce colors by clustering pixel values |
| Document clustering | Group similar news articles |
| Anomaly detection | Points far from all centroids = outliers |
| Feature engineering | Cluster membership as new feature |
| Semi-supervised learning | Label a few, propagate to clusters |
| Data summarization | Centroids represent cluster prototypes |

**K-Means vs Supervised Learning:**
- No labels required — learns structure purely from data distribution
- Objective is internal cohesion, not prediction accuracy
- Evaluated by cluster quality metrics (inertia, silhouette), not accuracy

---

## 3. The Math Behind K-Means

### 3.1 Notation

```
X         : Data matrix, shape (n, p) — n samples, p features
K         : Number of clusters (hyperparameter)
μₖ        : Centroid of cluster k, shape (p,)
cᵢ        : Cluster assignment of point xᵢ ∈ {1, ..., K}
Cₖ        : Set of points assigned to cluster k
rᵢₖ       : Binary indicator, rᵢₖ = 1 if xᵢ ∈ cluster k, else 0
```

### 3.2 Objective Function — Within-Cluster Sum of Squares (WCSS)

This is what K-Means minimizes:

```
J = Σₖ Σ_{xᵢ ∈ Cₖ} ||xᵢ - μₖ||²

Equivalently:

J = Σᵢ Σₖ rᵢₖ ||xᵢ - μₖ||²

where rᵢₖ ∈ {0, 1},  Σₖ rᵢₖ = 1  (each point belongs to exactly one cluster)
```

Also called **inertia** in scikit-learn.

### 3.3 Assignment Step (E-step analog)

For each point, assign to the nearest centroid:

```
cᵢ = argmin_k ||xᵢ - μₖ||²        for i = 1, ..., n

Equivalent to argmin_k ||xᵢ - μₖ||  (squared doesn't change argmin)
```

This is a Voronoi partition — each centroid defines a region.

### 3.4 Update Step (M-step analog)

Recompute each centroid as the mean of its assigned points:

```
μₖ = (1 / |Cₖ|) Σ_{xᵢ ∈ Cₖ} xᵢ

Derivation: minimize J w.r.t. μₖ
  ∂J/∂μₖ = -2 Σ_{xᵢ ∈ Cₖ} (xᵢ - μₖ) = 0
  => μₖ = mean of points in Cₖ  ✓
```

### 3.5 Distance Metric

Standard K-Means uses **squared Euclidean distance**:

```
d(xᵢ, μₖ)² = Σⱼ (xᵢⱼ - μₖⱼ)²  =  ||xᵢ - μₖ||²₂
```

This is why **scaling matters** — features with large ranges dominate the distance.

### 3.6 Convergence

Each iteration of K-Means cannot increase `J`:
- Assignment step: reassigning each point to a closer centroid → J decreases or stays same
- Update step: setting centroid to mean is the unique minimizer of squared distances → J decreases or stays same

Since `J ≥ 0` and is non-increasing, K-Means **always converges** (to a local minimum, not necessarily global).

### 3.7 Connection to Gaussian Mixture Models

K-Means is a **hard** version of the **Expectation-Maximization (EM)** algorithm for Gaussian Mixture Models:

| | K-Means | GMM (EM) |
|---|---|---|
| Assignment | Hard (0 or 1) | Soft (probability) |
| Cluster shape | Spherical | Any shape (covariance) |
| Step 1 | Assign to nearest centroid (E-step) | Compute responsibilities |
| Step 2 | Recompute means (M-step) | Update μ, Σ, π |
| Objective | WCSS (inertia) | Log-likelihood |

K-Means = EM with spherical, equal-variance Gaussians in the limit of zero variance.

---

## 4. Step-by-Step K-Means Algorithm

```
Input: X (n×p), K (number of clusters), max_iter, tol

Step 1: Initialize K centroids μ₁, μ₂, ..., μₖ
        (random, k-means++, or user-provided)

Step 2: REPEAT until convergence:

    ── Assignment Step ──────────────────────────────────────
    For each point xᵢ (i = 1 to n):
        cᵢ = argmin_k ||xᵢ - μₖ||²
        (assign to nearest centroid)

    ── Update Step ──────────────────────────────────────────
    For each cluster k (k = 1 to K):
        μₖ = mean({ xᵢ : cᵢ = k })
        (recompute centroid as cluster mean)

    ── Check Convergence ────────────────────────────────────
    If max centroid shift < tol:  STOP
    If iteration count ≥ max_iter: STOP
    If no reassignments occurred:  STOP

Output: {c₁, ..., cₙ} cluster labels, {μ₁, ..., μₖ} centroids, J (inertia)
```

### Initialization Strategies

**Random Initialization (naive):**
- Pick K random data points as initial centroids
- Problem: can converge to poor local minima

**K-Means++ (default in sklearn):**
```
1. Pick first centroid μ₁ uniformly at random from X
2. For k = 2, ..., K:
   - For each point xᵢ, compute D(xᵢ) = min distance to nearest chosen centroid
   - Pick next centroid with probability proportional to D(xᵢ)²
3. Run standard K-Means with these K initial centroids
```

K-Means++ ensures initial centroids are spread out → better convergence, lower final inertia.

---

## 5. K-Means as Optimization (Gradient Descent View)

### 5.1 Why K-Means is NOT Gradient Descent

K-Means minimizes a **non-convex, non-differentiable** objective `J` using coordinate descent (not gradient descent):

```
Coordinate Descent on J:
  - Fix μ, optimize r  → Assignment step (closed-form argmin)
  - Fix r, optimize μ  → Update step    (closed-form mean)
```

J is non-convex → multiple local minima → initialization matters.

### 5.2 Soft K-Means via Gradient Descent (Andrew Ng framing)

Andrew Ng (ML Specialization, Course 3) presents K-Means as directly minimizing WCSS:

```python
import numpy as np

def compute_inertia(X, centroids, labels):
    """J = Σᵢ ||xᵢ - μ_{cᵢ}||²"""
    return sum(np.sum((X[labels == k] - centroids[k])**2)
               for k in range(len(centroids)))

# The two steps ARE the coordinate descent:
# Step 1 (fix μ, solve for r):  labels = argmin_k ||x - μₖ||²
# Step 2 (fix r, solve for μ):  μₖ = mean(X[labels==k])
```

### 5.3 Gradient of Inertia w.r.t. Centroids

For fixed assignments, inertia is differentiable w.r.t. centroids:

```
J = Σₖ Σ_{i: cᵢ=k} ||xᵢ - μₖ||²

∂J/∂μₖ = -2 Σ_{i: cᵢ=k} (xᵢ - μₖ)

Setting ∂J/∂μₖ = 0:
  μₖ* = (1/|Cₖ|) Σ_{i: cᵢ=k} xᵢ   ← The centroid update IS gradient descent with exact step!
```

The update step = **one Newton step** (exact minimizer), not iterative gradient descent.

### 5.4 Online / Stochastic K-Means (Mini-Batch Gradient View)

Mini-Batch K-Means approximates the update with stochastic steps:

```
For each mini-batch B:
    1. Assign each xᵢ ∈ B to nearest centroid
    2. Update: μₖ ← μₖ - η * ∂J_B/∂μₖ
             = μₖ + (η / |B∩Cₖ|) * Σ_{i∈B∩Cₖ} (xᵢ - μₖ)

Learning rate η decreases over time (similar to SGD)
```

This is `MiniBatchKMeans` in sklearn — much faster, slightly less accurate.

---

## 6. K-Means from Scratch in Python (NumPy)

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class KMeansFromScratch:
    """
    K-Means clustering with K-Means++ initialization.
    Mirrors sklearn KMeans API.
    """

    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4,
                 init='k-means++', n_init=10, random_state=None):
        self.n_clusters   = n_clusters
        self.max_iter     = max_iter
        self.tol          = tol
        self.init         = init
        self.n_init       = n_init
        self.random_state = random_state

    # ── Initialization ──────────────────────────────────────────

    def _init_random(self, X, rng):
        """Pick K random data points as centroids."""
        idx = rng.choice(len(X), self.n_clusters, replace=False)
        return X[idx].copy()

    def _init_kmeanspp(self, X, rng):
        """K-Means++ initialization (Arthur & Vassilvitskii, 2007)."""
        n = len(X)
        # 1. Pick first centroid uniformly
        idx = rng.integers(0, n)
        centroids = [X[idx].copy()]

        for _ in range(1, self.n_clusters):
            # 2. Compute D(x)² = min squared dist to nearest chosen centroid
            dists = np.array([
                min(np.sum((x - c)**2) for c in centroids)
                for x in X
            ])
            # 3. Sample next centroid proportional to D(x)²
            probs = dists / dists.sum()
            idx = rng.choice(n, p=probs)
            centroids.append(X[idx].copy())

        return np.array(centroids)

    # ── Core Steps ──────────────────────────────────────────────

    def _assign(self, X, centroids):
        """Assign each point to nearest centroid. Returns labels array."""
        # distances: shape (n, K)
        diffs = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]  # (n, K, p)
        sq_dists = np.sum(diffs**2, axis=2)                         # (n, K)
        return np.argmin(sq_dists, axis=1)                          # (n,)

    def _update(self, X, labels):
        """Recompute centroids as cluster means."""
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            mask = labels == k
            if mask.sum() > 0:
                centroids[k] = X[mask].mean(axis=0)
            else:
                # Empty cluster: reinitialize to random point
                centroids[k] = X[np.random.randint(len(X))]
        return centroids

    def _inertia(self, X, labels, centroids):
        """WCSS = Σᵢ ||xᵢ - μ_{cᵢ}||²"""
        total = 0.0
        for k in range(self.n_clusters):
            pts = X[labels == k]
            if len(pts) > 0:
                total += np.sum((pts - centroids[k])**2)
        return total

    # ── Single Run ──────────────────────────────────────────────

    def _fit_single(self, X, rng):
        """One full K-Means run from a single initialization."""
        # Initialize
        if self.init == 'k-means++':
            centroids = self._init_kmeanspp(X, rng)
        else:
            centroids = self._init_random(X, rng)

        labels = None
        for iteration in range(self.max_iter):
            # Assignment step
            new_labels = self._assign(X, centroids)

            # Update step
            new_centroids = self._update(X, new_labels)

            # Check convergence: max centroid shift
            shift = np.max(np.linalg.norm(new_centroids - centroids, axis=1))

            centroids = new_centroids
            labels    = new_labels

            if shift < self.tol:
                break

        inertia = self._inertia(X, labels, centroids)
        return labels, centroids, inertia

    # ── Public API ──────────────────────────────────────────────

    def fit(self, X):
        X = np.array(X, dtype=float)
        rng = np.random.default_rng(self.random_state)

        best_inertia   = np.inf
        best_labels    = None
        best_centroids = None

        # Run n_init times, keep best result
        for _ in range(self.n_init):
            labels, centroids, inertia = self._fit_single(X, rng)
            if inertia < best_inertia:
                best_inertia   = inertia
                best_labels    = labels
                best_centroids = centroids

        self.labels_    = best_labels
        self.cluster_centers_ = best_centroids
        self.inertia_   = best_inertia
        self.n_iter_    = self.max_iter   # simplified
        return self

    def predict(self, X):
        X = np.array(X, dtype=float)
        return self._assign(X, self.cluster_centers_)

    def fit_predict(self, X):
        return self.fit(X).labels_


# ──────────────────────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────────────────────
from sklearn.datasets import make_blobs

X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.8,
                        random_state=42)

km = KMeansFromScratch(n_clusters=4, n_init=10, random_state=42)
labels = km.fit_predict(X)

print(f"Inertia : {km.inertia_:.2f}")
print(f"Centers :\n{km.cluster_centers_}")

# Plot
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
plt.figure(figsize=(8, 6))
for k in range(4):
    mask = labels == k
    plt.scatter(X[mask, 0], X[mask, 1], c=colors[k], alpha=0.6, s=30)
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
            c='black', marker='X', s=200, zorder=5, label='Centroids')
plt.title("K-Means from Scratch")
plt.legend()
plt.tight_layout()
plt.show()
```

---

## 7. K-Means with Scikit-Learn

### 7.1 Basic Usage

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

X, _ = make_blobs(n_samples=500, centers=4, cluster_std=0.9, random_state=42)

# ALWAYS scale before K-Means
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit K-Means
km = KMeans(n_clusters=4, init='k-means++', n_init=10,
            max_iter=300, tol=1e-4, random_state=42)
km.fit(X_scaled)

# Key outputs
print("Labels       :", km.labels_[:10])
print("Inertia      :", km.inertia_)
print("Centers shape:", km.cluster_centers_.shape)
print("Iterations   :", km.n_iter_)
```

### 7.2 Key Attributes After Fitting

```python
km.labels_           # Cluster label for each training point, shape (n,)
km.cluster_centers_  # Centroid coordinates, shape (K, p)
km.inertia_          # WCSS (total within-cluster sum of squares)
km.n_iter_           # Number of iterations run
km.n_features_in_    # Number of features seen during fit
```

### 7.3 Predict on New Data

```python
# Assign new points to nearest centroid
X_new = np.array([[1.0, 2.0], [-1.5, 0.5]])
labels_new = km.predict(scaler.transform(X_new))
print("New point labels:", labels_new)

# Distance to each centroid
distances = km.transform(X_scaled)  # shape (n, K)
# distances[i, k] = Euclidean distance from point i to centroid k
```

### 7.4 Full Workflow with Visualization

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

# Data
X, y = make_blobs(n_samples=400, centers=4, random_state=42, cluster_std=0.7)
X_s = StandardScaler().fit_transform(X)

# Fit
km = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42)
km.fit(X_s)

# Decision boundary (Voronoi regions)
h = 0.02
x_min, x_max = X_s[:, 0].min() - 0.5, X_s[:, 0].max() + 0.5
y_min, y_max = X_s[:, 1].min() - 0.5, X_s[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                      np.arange(y_min, y_max, h))
Z = km.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(9, 7))
plt.contourf(xx, yy, Z, alpha=0.25, cmap='Set1')
scatter = plt.scatter(X_s[:, 0], X_s[:, 1], c=km.labels_,
                       cmap='Set1', s=20, alpha=0.8, edgecolors='k', lw=0.2)
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
            c='yellow', s=250, marker='*', edgecolors='black',
            linewidths=1.5, zorder=10, label='Centroids')
plt.title(f"K-Means (K=4) — Inertia: {km.inertia_:.2f}")
plt.legend()
plt.tight_layout()
plt.show()
```

### 7.5 Elbow Method

```python
inertias = []
K_range = range(1, 12)

for k in K_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    km.fit(X_s)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, inertias, 'bo-', markersize=8)
plt.xlabel("Number of Clusters K")
plt.ylabel("Inertia (WCSS)")
plt.title("Elbow Method for Optimal K")
plt.xticks(K_range)
plt.grid(True, alpha=0.3)
# Mark elbow visually
plt.annotate('Elbow ≈ K=4', xy=(4, inertias[3]),
             xytext=(6, inertias[3]*1.1),
             arrowprops=dict(arrowstyle='->', color='red'),
             fontsize=12, color='red')
plt.tight_layout()
plt.show()
```

### 7.6 Silhouette Analysis

```python
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.cm as cm

silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = km.fit_predict(X_s)
    score = silhouette_score(X_s, labels)
    silhouette_scores.append(score)
    print(f"K={k}: Silhouette = {score:.4f}")

best_k = K_range[np.argmax(silhouette_scores)]
print(f"\nBest K by silhouette: {best_k}")

# Silhouette plot for best K
km_best = KMeans(n_clusters=best_k, n_init=10, random_state=42)
labels_best = km_best.fit_predict(X_s)
sample_sils = silhouette_samples(X_s, labels_best)

fig, ax = plt.subplots(figsize=(8, 5))
y_lower = 10
for k in range(best_k):
    vals = np.sort(sample_sils[labels_best == k])
    y_upper = y_lower + len(vals)
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals,
                     alpha=0.7, label=f'Cluster {k}')
    y_lower = y_upper + 10
ax.axvline(x=silhouette_score(X_s, labels_best), color='red',
           linestyle='--', label='Mean silhouette')
ax.set_xlabel("Silhouette Coefficient")
ax.set_title(f"Silhouette Plot (K={best_k})")
ax.legend(loc='lower right')
plt.tight_layout()
plt.show()
```

---

## 8. Hyperparameters & Tuning

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_clusters` | int | 8 | **K** — number of clusters. Most important hyperparameter. |
| `init` | str or array | `'k-means++'` | Initialization method. `'k-means++'` (smart), `'random'`, or array of shape `(K, p)` |
| `n_init` | int or `'auto'` | `'auto'` (≥1.4) or 10 | Number of times to run K-Means with different seeds. Best result kept. |
| `max_iter` | int | 300 | Max iterations per run |
| `tol` | float | 1e-4 | Convergence tolerance (relative change in inertia) |
| `algorithm` | str | `'lloyd'` | Algorithm: `'lloyd'` (standard), `'elkan'` (faster for dense with many clusters) |
| `random_state` | int | None | Seed for reproducibility |
| `verbose` | int | 0 | Verbosity of output |
| `copy_x` | bool | True | Pre-center data; if False, modifies X in-place |

### Detailed Notes

```python
# init = 'k-means++'  (default, recommended)
# Much better than 'random' — reduces iterations needed, avoids bad local minima
km = KMeans(n_clusters=5, init='k-means++', random_state=42)

# n_init: more runs = more likely to find global minimum, but slower
# sklearn >= 1.4: n_init='auto' sets n_init=10 for 'random', 1 for 'k-means++'
km = KMeans(n_clusters=5, n_init=20)  # 20 independent runs

# algorithm = 'elkan' uses triangle inequality — faster for low-dimensional dense data
# algorithm = 'lloyd' is standard — better for sparse data
km = KMeans(n_clusters=5, algorithm='elkan')

# Warm starting with custom centroids
initial_centers = np.array([[0, 0], [1, 1], [-1, 1]])
km = KMeans(n_clusters=3, init=initial_centers, n_init=1)
```

---

## 9. Choosing K — The Right Number of Clusters

### Method 1: Elbow Method (Inertia vs K)

```python
inertias = []
for k in range(1, 15):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X_s)
    inertias.append(km.inertia_)

# Look for the "elbow" — point of diminishing returns
# No mathematical definition; visual judgment
```

**Limitation:** Often ambiguous — the elbow isn't always clear.

### Method 2: Silhouette Score (Best mathematical criterion)

```python
from sklearn.metrics import silhouette_score

scores = {}
for k in range(2, 12):
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X_s)
    scores[k] = silhouette_score(X_s, labels)

best_k = max(scores, key=scores.get)
# Silhouette ∈ [-1, 1]; higher is better
# silhouette ≈ 1: dense, well-separated clusters
# silhouette ≈ 0: overlapping clusters
# silhouette < 0: misassigned points
```

### Method 3: Gap Statistic

```python
# Gap(K) = E*[log(Wₖ)] - log(Wₖ)
# where E* = expected value under null reference distribution
# Choose K where Gap(K) ≥ Gap(K+1) - std(K+1)
# sklearn doesn't implement this natively; use gap-statistic or yellowbrick

# pip install gap-stat
from gap_statistic import OptimalK
opt_k = OptimalK(n_jobs=1)
n_clusters = opt_k(X_s, cluster_array=np.arange(1, 15))
print(f"Optimal K: {n_clusters}")
```

### Method 4: Davies-Bouldin Index

```python
from sklearn.metrics import davies_bouldin_score

# Lower is better (0 = perfect)
db_scores = {}
for k in range(2, 12):
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X_s)
    db_scores[k] = davies_bouldin_score(X_s, labels)

best_k_db = min(db_scores, key=db_scores.get)
```

### Method 5: Calinski-Harabasz Index (Variance Ratio Criterion)

```python
from sklearn.metrics import calinski_harabasz_score

# Higher is better (ratio of between-cluster to within-cluster dispersion)
ch_scores = {}
for k in range(2, 12):
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X_s)
    ch_scores[k] = calinski_harabasz_score(X_s, labels)

best_k_ch = max(ch_scores, key=ch_scores.get)
```

### Method 6: Domain Knowledge

Always validate K against business logic:
- Customer segments: 3–7 meaningful groups
- Image colors: 8–256 colors for compression
- Topic modeling: check topic coherence

---

## 10. Evaluation Metrics

### 10.1 Internal Metrics (No Labels Needed)

| Metric | Formula | Best Value | sklearn function |
|---|---|---|---|
| **Inertia (WCSS)** | `Σᵢ ||xᵢ - μ_{cᵢ}||²` | Lower → better (minimize) | `km.inertia_` |
| **Silhouette Score** | `(b-a)/max(a,b)` per point | `[-1, 1]`, higher better | `silhouette_score(X, labels)` |
| **Davies-Bouldin** | Avg max cluster similarity | Lower → better (0 = best) | `davies_bouldin_score(X, labels)` |
| **Calinski-Harabasz** | Between/within cluster ratio | Higher → better | `calinski_harabasz_score(X, labels)` |

**Silhouette math:**
```
For each point i:
  a(i) = mean distance to other points in same cluster    (cohesion)
  b(i) = mean distance to points in nearest other cluster (separation)

  s(i) = (b(i) - a(i)) / max(a(i), b(i))

Overall silhouette = mean(s(i)) for all i
```

### 10.2 External Metrics (When True Labels Available — for validation)

```python
from sklearn.metrics import (
    adjusted_rand_score,           # ARI: -1 to 1, 1=perfect, 0=random
    adjusted_mutual_info_score,    # AMI: accounts for chance
    normalized_mutual_info_score,  # NMI: 0 to 1
    homogeneity_score,             # Each cluster contains only one class
    completeness_score,            # Each class is in only one cluster
    v_measure_score,               # Harmonic mean of homogeneity & completeness
    fowlkes_mallows_score,         # Geometric mean of precision/recall
)

# Example
ari  = adjusted_rand_score(y_true, km.labels_)
ami  = adjusted_mutual_info_score(y_true, km.labels_)
nmi  = normalized_mutual_info_score(y_true, km.labels_)
hom  = homogeneity_score(y_true, km.labels_)
comp = completeness_score(y_true, km.labels_)
vm   = v_measure_score(y_true, km.labels_)

print(f"ARI={ari:.3f}, AMI={ami:.3f}, NMI={nmi:.3f}")
print(f"Homogeneity={hom:.3f}, Completeness={comp:.3f}, V-measure={vm:.3f}")
```

---

## 11. Variants of K-Means

### 11.1 Mini-Batch K-Means — Fast, Scalable

```python
from sklearn.cluster import MiniBatchKMeans

# Uses mini-batches instead of full dataset per iteration
# Much faster for large datasets; slightly higher inertia
mbkm = MiniBatchKMeans(
    n_clusters=4,
    batch_size=100,         # Mini-batch size (default: 1024 in sklearn >= 1.0)
    max_iter=100,
    n_init=3,
    init='k-means++',
    reassignment_ratio=0.01,  # Fraction of centers reassigned per step
    random_state=42
)
labels = mbkm.fit_predict(X_s)
print(f"Mini-Batch Inertia: {mbkm.inertia_:.2f}")
```

### 11.2 K-Medoids — Robust to Outliers

```python
# pip install scikit-learn-extra
from sklearn_extra.cluster import KMedoids

# Centroids are actual data points (medoids), not means
# Less sensitive to outliers than K-Means
kmed = KMedoids(n_clusters=4, metric='euclidean',
                method='pam',         # 'pam' (exact), 'alternate' (fast)
                init='k-medoids++',
                random_state=42)
labels = kmed.fit_predict(X_s)
print(f"Medoid indices: {kmed.medoid_indices_}")
```

### 11.3 Fuzzy C-Means (Soft K-Means)

```python
# pip install scikit-fuzzy
import skfuzzy as fuzz

# Each point has a membership degree to each cluster ∈ [0, 1]
# Useful when points belong to multiple clusters
X_T = X_s.T  # skfuzzy expects (features, samples)
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    X_T,
    c=4,      # number of clusters
    m=2.0,    # fuzziness exponent (m>1; m→1 = hard K-Means, m→∞ = uniform)
    error=0.005,
    maxiter=1000,
    init=None
)
# u[k, i] = degree of membership of point i in cluster k
labels_fuzzy = np.argmax(u, axis=0)
```

### 11.4 K-Means++ Only (Better Initialization)

```python
from sklearn.cluster import kmeans_plusplus

# Get K-Means++ centers without running full K-Means
centers, indices = kmeans_plusplus(X_s, n_clusters=4, random_state=42)
print("Initial centers:\n", centers)
```

### 11.5 Bisecting K-Means

```python
from sklearn.cluster import BisectingKMeans

# Divisive hierarchical approach using K-Means
# Better inertia than standard K-Means, especially for large K
bkm = BisectingKMeans(
    n_clusters=4,
    init='k-means++',
    n_init=1,
    bisecting_strategy='biggest_inertia',  # or 'largest_cluster'
    random_state=42
)
labels = bkm.fit_predict(X_s)
```

### 11.6 Comparison Table

| Method | Outlier Robust | Soft Membership | Scalability | Shape Assumption |
|---|---|---|---|---|
| K-Means | No | No (hard) | High | Spherical, equal-size |
| Mini-Batch K-Means | No | No | Very High | Spherical |
| K-Medoids | Yes | No | Medium | Any metric |
| Fuzzy C-Means | Partial | Yes | Medium | Spherical |
| Bisecting K-Means | No | No | High | Spherical |
| GMM (EM) | No | Yes | Medium | Elliptical |
| DBSCAN | Yes | No | Medium | Arbitrary |
| Agglomerative | Partial | No | Low | Any linkage |

---

## 12. K-Means in ML Pipelines

### 12.1 K-Means + PCA Visualization

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits

X, y = load_digits(return_X_y=True)

# Scale → PCA → K-Means
scaler = StandardScaler()
X_s = scaler.fit_transform(X)

pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_s)

km = KMeans(n_clusters=10, n_init=10, random_state=42)
labels = km.fit_predict(X_2d)

plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels,
                       cmap='tab10', alpha=0.5, s=10)
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
            c='black', marker='X', s=200, zorder=5, label='Centroids')
plt.colorbar(scatter, label='Cluster')
plt.title("K-Means (K=10) on PCA-reduced Digits")
plt.legend()
plt.tight_layout()
plt.show()
```

### 12.2 K-Means as Feature Engineering (Cluster Features)

```python
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np

# Use cluster distances as features (semi-supervised trick)
class KMeansFeaturizer:
    """Transform X into cluster distance features."""
    def __init__(self, k=50):
        self.km = KMeans(n_clusters=k, n_init=3, random_state=42)
    def fit(self, X, y=None):
        self.km.fit(X)
        return self
    def transform(self, X):
        return self.km.transform(X)   # shape (n, k): distances to each centroid

# Pipeline: Scale → KMeansFeaturizer → Classifier
from sklearn.datasets import make_classification
X_c, y_c = make_classification(n_samples=1000, n_features=20, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X_c, y_c, test_size=0.2, random_state=42)

pipe = Pipeline([
    ('scaler',   StandardScaler()),
    ('km_feat',  KMeansFeaturizer(k=20)),
    ('clf',      LogisticRegression(max_iter=500))
])
pipe.fit(X_tr, y_tr)
print(f"Test Accuracy (KMeans features): {pipe.score(X_te, y_te):.4f}")
```

### 12.3 Image Compression with K-Means

```python
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

# Download a sample image or use any RGB image
# image = io.imread('photo.jpg')

# Simulate with random pixels
np.random.seed(42)
image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

# Reshape to (n_pixels, 3)
h, w, c = image.shape
pixels = image.reshape(-1, 3).astype(float)

# Compress to K colors
K = 16
km = KMeans(n_clusters=K, n_init=3, random_state=42)
km.fit(pixels)
compressed_pixels = km.cluster_centers_[km.labels_]
compressed_image = compressed_pixels.reshape(h, w, c).astype(np.uint8)

# Compare
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(image);         axes[0].set_title("Original")
axes[1].imshow(compressed_image); axes[1].set_title(f"K-Means K={K} colors")
for ax in axes: ax.axis('off')
plt.tight_layout()
plt.show()

orig_size = h * w * 24      # bits: 3 channels × 8 bits each
comp_size = h * w * np.log2(K) + K * 3 * 8   # label bits + palette
print(f"Compression ratio: {orig_size/comp_size:.1f}x")
```

### 12.4 K-Means for Anomaly Detection

```python
# Points far from all centroids → potential anomalies
km = KMeans(n_clusters=5, n_init=10, random_state=42)
km.fit(X_s)

# Distance to nearest centroid
distances = km.transform(X_s)     # (n, K)
min_dist  = distances.min(axis=1) # (n,) distance to assigned centroid

# Threshold: top 5% distances = anomalies
threshold = np.percentile(min_dist, 95)
anomalies = min_dist > threshold
print(f"Anomalies detected: {anomalies.sum()}")

plt.figure(figsize=(8, 5))
plt.scatter(X_s[~anomalies, 0], X_s[~anomalies, 1],
            alpha=0.4, s=10, label='Normal')
plt.scatter(X_s[anomalies, 0], X_s[anomalies, 1],
            c='red', s=40, marker='x', label='Anomaly')
plt.title("K-Means Anomaly Detection")
plt.legend()
plt.tight_layout()
plt.show()
```

---

## 13. Limitations & When NOT to Use K-Means

| Limitation | Problem | Alternative |
|---|---|---|
| Assumes spherical clusters | Elongated or crescent shapes | DBSCAN, GMM, Spectral Clustering |
| Assumes equal cluster size | Very unequal cluster sizes | GMM, Agglomerative |
| Sensitive to outliers | Outliers pull centroids | K-Medoids, DBSCAN |
| Must specify K | Don't know K in advance | DBSCAN, HDBSCAN (auto K) |
| Sensitive to scale | Large-scale features dominate | Always `StandardScaler` first |
| Local minima | Different runs → different results | Multiple restarts (`n_init`) |
| Non-convex clusters | Can't cluster interlocking rings | Spectral, DBSCAN |
| Categorical data | Distance undefined | K-Modes, K-Prototypes |

```python
# K-Means FAILS on non-convex clusters:
from sklearn.datasets import make_moons, make_circles

X_moons, _ = make_moons(n_samples=300, noise=0.05, random_state=42)
km_moons = KMeans(n_clusters=2, n_init=10, random_state=42)
labels_moons = km_moons.fit_predict(X_moons)
# Result: terrible clusters — use DBSCAN or SpectralClustering instead

from sklearn.cluster import DBSCAN, SpectralClustering
labels_db = DBSCAN(eps=0.2, min_samples=5).fit_predict(X_moons)       # Works!
labels_sp = SpectralClustering(n_clusters=2, random_state=42).fit_predict(X_moons)  # Works!
```

---

## 14. Common Interview Questions

### Q1: What does K-Means minimize?
> K-Means minimizes the **Within-Cluster Sum of Squares (WCSS)** or **inertia**: `J = Σₖ Σ_{xᵢ∈Cₖ} ||xᵢ - μₖ||²`. This is equivalent to finding cluster assignments and centroids such that each point is as close as possible to its cluster center.

### Q2: Does K-Means always converge?
> Yes, K-Means always converges because each step (assignment and update) either decreases or maintains inertia, and inertia is bounded below by 0. However, it converges to a **local minimum**, not necessarily the global one. Running multiple times (`n_init`) mitigates this.

### Q3: How does K-Means++ differ from random init?
> K-Means++ selects initial centroids with probability proportional to their squared distance from already-chosen centroids. This ensures spread-out initialization, leading to faster convergence and lower final inertia. Mathematically, it gives an `O(log K)` approximation guarantee vs random init.

### Q4: Why must you scale data before K-Means?
> K-Means uses Euclidean distance. Features with large ranges (e.g., income in $100K) dominate the distance metric over small-range features (e.g., age). StandardScaler equalizes feature contributions.

### Q5: How do you choose K?
> Multiple methods: (1) **Elbow method** — plot inertia vs K, find the bend. (2) **Silhouette score** — maximize average silhouette. (3) **Domain knowledge** — business or scientific context. (4) **Gap statistic**, **Davies-Bouldin**, **Calinski-Harabasz**. In practice, combine several methods.

### Q6: K-Means vs GMM — what's the difference?
> K-Means does **hard** assignment (each point to exactly one cluster) and assumes **spherical** clusters. GMM does **soft** assignment (probabilistic membership) and models clusters as elliptical Gaussians with arbitrary covariance. GMM is more flexible but harder to fit. K-Means is a special case of GMM.

### Q7: What is inertia and what are its limitations?
> Inertia = WCSS = total squared distance from each point to its assigned centroid. Limitation: **always decreases as K increases**, so you can't just minimize inertia to pick K. Use the elbow or silhouette score instead.

### Q8: Is K-Means supervised or unsupervised?
> **Unsupervised** — no labels are used. K-Means discovers structure in unlabeled data. However, if labels exist, you can use them for external validation (ARI, NMI) but not for training.

### Q9: Can K-Means handle non-numeric/categorical data?
> No. Standard K-Means requires numeric data because it computes means and Euclidean distances. For categorical data, use **K-Modes**. For mixed data, use **K-Prototypes**.

### Q10: What is the time complexity of K-Means?
```
Time complexity: O(n × K × p × I)

where:
  n = number of samples
  K = number of clusters
  p = number of features
  I = number of iterations

Space complexity: O(n × K)  (for distance matrix)

Mini-Batch KMeans: O(b × K × p × I)  where b << n (batch size)
```

---

## 15. Resources

### 📖 Books

| Book | Chapter/Section | Topic |
|---|---|---|
| **Hands-On ML** (Géron, 3rd ed.) | Chapter 9 | K-Means, DBSCAN, GMM — complete coverage |
| **ISLP** (James et al.) | Chapter 12.4 | K-Means Clustering, hierarchical clustering |
| **ESL** (Hastie et al.) | Chapter 14.3 | K-Means and vector quantization |
| **Pattern Recognition** (Bishop) | Chapter 9.1 | K-Means as EM limit |

### 🎓 Andrew Ng — ML Specialization (Coursera)

- **Course 3: Unsupervised Learning, Recommenders, Reinforcement Learning**
  - Week 1: K-Means Algorithm, optimization objective, random initialization
  - Key lectures: "K-Means intuition", "Optimization objective", "Initializing K-Means", "Choosing the number of clusters"
- **Course 1: Supervised Machine Learning** — No K-Means, but covers clustering motivation

### 📺 StatQuest with Josh Starmer (YouTube)

| Video | Search Query |
|---|---|
| K-Means Clustering | `StatQuest K-means clustering` |
| K-Means++ | `StatQuest K-means++` |
| Silhouette | `StatQuest silhouette` |
| Hierarchical Clustering | `StatQuest hierarchical clustering` |
| DBSCAN | `StatQuest DBSCAN` |

### 📚 Scikit-Learn Documentation

| Resource | URL Suffix |
|---|---|
| `sklearn.cluster.KMeans` | `/stable/modules/generated/sklearn.cluster.KMeans` |
| `sklearn.cluster.MiniBatchKMeans` | `/stable/modules/generated/sklearn.cluster.MiniBatchKMeans` |
| Clustering User Guide | `/stable/modules/clustering.html` |
| Clustering comparison | `/stable/auto_examples/cluster/plot_cluster_comparison` |
| Silhouette analysis | `/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis` |

> Base URL: `https://scikit-learn.org`

### 📄 Key Papers

| Paper | Note |
|---|---|
| Lloyd (1982) | Original K-Means algorithm |
| Arthur & Vassilvitskii (2007) | K-Means++ initialization |
| Sculley (2010) | Web-scale K-Means (mini-batch) |
| Elkan (2003) | Accelerated K-Means with triangle inequality |

---

## 🗂️ Quick Reference Cheat Sheet

```python
# ── Minimal K-Means Workflow ──────────────────────────────────
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 1. Scale (MANDATORY)
X_s = StandardScaler().fit_transform(X)

# 2. Fit
km = KMeans(
    n_clusters   = 4,           # K: number of clusters
    init         = 'k-means++', # smart initialization (default)
    n_init       = 10,          # number of independent runs
    max_iter     = 300,         # max iterations per run
    tol          = 1e-4,        # convergence threshold
    algorithm    = 'lloyd',     # 'lloyd' or 'elkan'
    random_state = 42
)
km.fit(X_s)

# 3. Results
labels   = km.labels_            # (n,) cluster assignments
centers  = km.cluster_centers_   # (K, p) centroid coordinates
inertia  = km.inertia_           # WCSS (lower = more compact)
sil      = silhouette_score(X_s, labels)   # [-1,1] higher = better

# 4. Predict new data
new_labels = km.predict(X_new_scaled)

# ── Choose K ─────────────────────────────────────────────────
# Elbow: plot inertia vs K, look for bend
# Silhouette: argmax silhouette_score over K range
# Davies-Bouldin: argmin davies_bouldin_score over K range

# ── Key Metrics ──────────────────────────────────────────────
from sklearn.metrics import (
    silhouette_score,        # internal, no labels needed [-1,1] ↑
    davies_bouldin_score,    # internal, no labels needed [0,∞] ↓
    calinski_harabasz_score, # internal, no labels needed [0,∞] ↑
    adjusted_rand_score,     # external, needs true labels [-1,1] ↑
    normalized_mutual_info_score,  # external [0,1] ↑
)

# ── Variants ─────────────────────────────────────────────────
from sklearn.cluster import (
    MiniBatchKMeans,    # large datasets, approximate
    BisectingKMeans,    # hierarchical splits, better inertia
)
# K-Medoids: pip install scikit-learn-extra
# Fuzzy C-Means: pip install scikit-fuzzy
```

---

*Notes compiled for ML/DL job readiness. Covers K-Means theory, WCSS objective, coordinate descent formulation, K-Means++ math, NumPy implementation, full sklearn API, all hyperparameters, choosing K (5 methods), evaluation metrics (internal + external), variants, real-world pipelines, and limitations.*
