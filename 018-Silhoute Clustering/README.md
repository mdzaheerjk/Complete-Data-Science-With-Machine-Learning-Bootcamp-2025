# 📊 Silhouette Analysis for Clustering — Complete ML/DL Job-Ready Notes

> **"How well does each point fit its own cluster compared to its nearest rival cluster?"**
> The Silhouette Score is a **cluster evaluation metric** that measures the quality of any clustering result — without needing ground-truth labels. It quantifies both **cohesion** (how tight a cluster is) and **separation** (how far apart clusters are).

---

## 📌 Table of Contents

1. [Intuition](#1-intuition)
2. [Core Definitions](#2-core-definitions)
3. [Mathematics — Full Derivation](#3-mathematics--full-derivation)
4. [Interpreting Silhouette Scores](#4-interpreting-silhouette-scores)
5. [Silhouette for Choosing K](#5-silhouette-for-choosing-k)
6. [Silhouette Plot — Deep Dive](#6-silhouette-plot--deep-dive)
7. [Silhouette vs Other Clustering Metrics](#7-silhouette-vs-other-clustering-metrics)
8. [No Gradient Descent — Why](#8-no-gradient-descent--why)
9. [Hyperparameters and Configuration](#9-hyperparameters-and-configuration)
10. [Python from Scratch](#10-python-from-scratch)
11. [Scikit-learn Implementation](#11-scikit-learn-implementation)
12. [Full Pipeline — K-Means + Silhouette](#12-full-pipeline--k-means--silhouette)
13. [Full Pipeline — DBSCAN + Silhouette](#13-full-pipeline--dbscan--silhouette)
14. [Silhouette with All Clustering Algorithms](#14-silhouette-with-all-clustering-algorithms)
15. [Pros and Cons](#15-pros-and-cons)
16. [Interview Questions](#16-interview-questions)
17. [Resources](#17-resources)

---

## 1. Intuition

**Simple statement:** For every data point, ask two questions:
1. How similar is it to its **own cluster**? (cohesion — want this SMALL)
2. How different is it from the **nearest other cluster**? (separation — want this LARGE)

Silhouette score = how much larger the separation is compared to the cohesion.

**Real-world analogy:**

```
Imagine students grouped by major:

Student A (CS major, assigned to CS group):
  → Average distance to CS classmates:        2.0  (tight group = good)
  → Average distance to nearest other group:  8.0  (far from Math group = good)
  → Silhouette = (8.0 - 2.0) / max(8.0, 2.0) = 6/8 = 0.75  ✅ well placed

Student B (CS major, but assigned to Math group by mistake):
  → Average distance to Math classmates:      7.0  (loose = bad)
  → Average distance to CS group:             2.0  (closer to CS = bad assignment)
  → Silhouette = (2.0 - 7.0) / max(7.0, 2.0) = -5/7 = -0.71  ❌ wrongly placed
```

**Visual intuition:**

```
Good clustering (high silhouette):    Bad clustering (low silhouette):

  ●●●        ■■■                         ●■●        ■●■
  ●●●        ■■■                         ■●■        ●■●
  ●●          ■■                         ●●■        ■●●
  (tight,     (tight,                    (mixed,    (mixed,
  well-sep.)  well-sep.)                 overlapping)
  s ≈ 0.8                                s ≈ 0.1
```

---

## 2. Core Definitions

### For a single point $i$ in cluster $C_I$:

| Symbol | Name | Definition |
|--------|------|-----------|
| $a(i)$ | **Intra-cluster distance** (cohesion) | Mean distance from $i$ to all other points in its own cluster $C_I$ |
| $b(i)$ | **Nearest-cluster distance** (separation) | Mean distance from $i$ to all points in the nearest **other** cluster |
| $s(i)$ | **Silhouette coefficient** of point $i$ | Combines $a(i)$ and $b(i)$ into a single score |

### $a(i)$ — Cohesion

$$a(i) = \frac{1}{|C_I| - 1} \sum_{j \in C_I,\, j \neq i} d(i, j)$$

- Small $a(i)$ → point $i$ is close to its cluster members → **good**
- $|C_I| - 1$ because we exclude the point itself

### $b(i)$ — Separation

$$b(i) = \min_{J \neq I} \frac{1}{|C_J|} \sum_{j \in C_J} d(i, j)$$

- For each **other** cluster $C_J$, compute mean distance from $i$ to all points in $C_J$
- $b(i)$ = the **minimum** of these mean distances (nearest rival cluster)
- Large $b(i)$ → point $i$ is far from neighboring clusters → **good**

---

## 3. Mathematics — Full Derivation

### Silhouette Coefficient for Point $i$

$$s(i) = \frac{b(i) - a(i)}{\max(a(i),\, b(i))}$$

**Equivalently written as:**

$$s(i) = \begin{cases} 1 - \dfrac{a(i)}{b(i)} & \text{if } a(i) < b(i) \\[8pt] 0 & \text{if } a(i) = b(i) \\[8pt] \dfrac{b(i)}{a(i)} - 1 & \text{if } a(i) > b(i) \end{cases}$$

**Range:** $s(i) \in [-1, +1]$

### Why This Formula Works

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

The denominator $\max(a(i), b(i))$ **normalizes** to $[-1, +1]$:

- If $a(i) \ll b(i)$: tight cluster, far from neighbors → $s(i) \to +1$
- If $a(i) = b(i)$: on the boundary → $s(i) = 0$
- If $a(i) \gg b(i)$: point closer to another cluster → $s(i) \to -1$

### Mean Silhouette Score (Overall)

$$\bar{s} = \frac{1}{N} \sum_{i=1}^{N} s(i)$$

### Per-Cluster Silhouette Score

$$\bar{s}_k = \frac{1}{|C_k|} \sum_{i \in C_k} s(i)$$

### Worked Example

```
Dataset: 6 points in 2 clusters

Cluster A: p1=(1,1), p2=(1,2), p3=(2,1)
Cluster B: p4=(8,8), p5=(8,9), p6=(9,8)

For point p1=(1,1) in Cluster A:

a(p1) = mean distance to p2, p3
      = (d(p1,p2) + d(p1,p3)) / 2
      = (√((1-1)²+(1-2)²) + √((1-2)²+(1-1)²)) / 2
      = (1.0 + 1.0) / 2
      = 1.0

b(p1) = mean distance to Cluster B
      = (d(p1,p4) + d(p1,p5) + d(p1,p6)) / 3
      = (√98 + √113 + √113) / 3
      ≈ (9.90 + 10.63 + 10.63) / 3
      ≈ 10.39

s(p1) = (10.39 - 1.0) / max(10.39, 1.0)
      = 9.39 / 10.39
      ≈ 0.904   ← excellent!
```

### Edge Case: Single-Point Cluster

$$s(i) = 0 \quad \text{if } |C_I| = 1$$

A cluster with one point has no other members to compute $a(i)$, so the silhouette is undefined — sklearn sets it to 0.

---

## 4. Interpreting Silhouette Scores

### Score Interpretation Table

| Score Range | Interpretation | Action |
|-------------|---------------|--------|
| **0.71 – 1.00** | Strong structure — excellent clustering | Use this clustering |
| **0.51 – 0.70** | Reasonable structure — good clustering | Acceptable for most use cases |
| **0.26 – 0.50** | Weak structure — clusters may overlap | Consider different K or algorithm |
| **< 0.25** | No substantial structure found | Re-examine data, features, algorithm |
| **< 0** | Points likely in wrong cluster | Clustering is worse than random |

> Reference: Kaufman & Rousseeuw (1990) — original silhouette paper

### Per-Point Score Meaning

```
s(i) close to +1.0  → Point is deep inside its correct cluster
s(i) close to  0.0  → Point is on the boundary between two clusters
s(i) close to -1.0  → Point is closer to another cluster — possibly misassigned
```

### Effect of K on Silhouette

```
K too small:  Clusters merged that should be separate
              → low b(i) for many points (clusters close together)
              → low silhouette

K too large:  Clusters split that should be together
              → low b(i) for many points (nearest rival is a fragment of same cluster)
              → low silhouette

K optimal:    Natural cluster structure captured
              → high a(i) cohesion + high b(i) separation
              → high silhouette
```

---

## 5. Silhouette for Choosing K

**Simple statement:** Run clustering for K = 2, 3, 4, ..., N−1. Pick the K with the highest mean silhouette score.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

# ── Data ──────────────────────────────────────────────────────
X, _ = make_blobs(n_samples=500, centers=4,
                  cluster_std=0.8, random_state=42)
X_scaled = StandardScaler().fit_transform(X)

# ── Silhouette scores for K = 2..10 ──────────────────────────
K_range  = range(2, 11)
sil_scores = []
inertias   = []

for k in K_range:
    km     = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    sil    = silhouette_score(X_scaled, labels)
    sil_scores.append(sil)
    inertias.append(km.inertia_)
    print(f"K={k:2d} | Silhouette: {sil:.4f} | Inertia: {km.inertia_:.1f}")

# ── Plot ──────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(K_range, sil_scores, 'bo-', linewidth=2, markersize=8)
ax1.axvline(x=K_range[np.argmax(sil_scores)],
            color='red', linestyle='--',
            label=f"Best K = {K_range[np.argmax(sil_scores)]}")
ax1.set_xlabel('Number of Clusters K', fontsize=12)
ax1.set_ylabel('Mean Silhouette Score',  fontsize=12)
ax1.set_title('Silhouette Method for Optimal K', fontsize=13)
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(K_range, inertias, 'gs-', linewidth=2, markersize=8)
ax2.set_xlabel('Number of Clusters K', fontsize=12)
ax2.set_ylabel('Inertia (Within-SS)', fontsize=12)
ax2.set_title('Elbow Method (comparison)', fontsize=13)
ax2.grid(True, alpha=0.3)

plt.suptitle('Silhouette vs Elbow: Finding Optimal K', fontsize=14)
plt.tight_layout()
plt.show()

optimal_k = K_range[np.argmax(sil_scores)]
print(f"\nOptimal K (Silhouette): {optimal_k}")
print(f"Best Silhouette Score:  {max(sil_scores):.4f}")
```

---

## 6. Silhouette Plot — Deep Dive

**Simple statement:** A silhouette plot shows the silhouette score for every single point, grouped by cluster. Wide, uniform bands → good clusters. Thin, uneven bands with negative scores → bad clusters.

```
Silhouette Plot Anatomy:

Y-axis: clusters (sorted by size)
X-axis: silhouette coefficient [-1, +1]

Good clustering:                    Bad clustering:
  ████████████████ ←Cluster 1         ████ ←Cluster 1 (thin)
  ████████████████ ←Cluster 2         ███|█ ←has negatives
  ██████████████   ←Cluster 3       █████████ ←Cluster 2 (uneven)
  ─────────────────                  ─────────────────
  All wide, > avg line               Thin, below avg, some negative
```

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

def plot_silhouette(X, n_clusters, ax=None, title=None):
    """
    Full silhouette plot for a given K.
    Shows per-point and per-cluster silhouette analysis.
    """
    km     = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = km.fit_predict(X)

    # Compute silhouette scores for all samples
    sil_vals = silhouette_samples(X, labels)
    mean_sil = silhouette_score(X, labels)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    colors  = cm.tab10(np.linspace(0, 1, n_clusters))
    y_lower = 10

    for k in range(n_clusters):
        # Silhouette values for cluster k, sorted
        k_sil  = np.sort(sil_vals[labels == k])
        k_size = len(k_sil)
        y_upper = y_lower + k_size

        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0, k_sil,
            facecolor=colors[k],
            edgecolor=colors[k],
            alpha=0.7
        )
        # Cluster label in center of band
        ax.text(-0.05, y_lower + k_size / 2, str(k), fontsize=10)
        y_lower = y_upper + 10

    # Mean silhouette vertical line
    ax.axvline(x=mean_sil, color='red', linestyle='--', linewidth=2,
               label=f'Mean = {mean_sil:.3f}')

    ax.set_xlabel('Silhouette Coefficient', fontsize=11)
    ax.set_ylabel('Cluster', fontsize=11)
    ax.set_title(title or f'Silhouette Plot (K={n_clusters})', fontsize=12)
    ax.set_xlim([-0.2, 1.0])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)

    return mean_sil, labels


# ── Plot silhouette for K=2,3,4,5 side by side ───────────────
X, _ = make_blobs(n_samples=500, centers=4,
                  cluster_std=0.8, random_state=42)
X_sc  = StandardScaler().fit_transform(X)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for idx, k in enumerate([2, 3, 4, 5]):
    score, labels = plot_silhouette(X_sc, k, ax=axes[idx],
                                    title=f'K={k} (mean={0:.3f})')

plt.suptitle('Silhouette Plots for Different K Values', fontsize=15)
plt.tight_layout()
plt.show()
```

### Reading a Silhouette Plot

```
Signs of a GOOD silhouette plot:
  ✅ All clusters wider than the mean silhouette line
  ✅ No (or very few) negative values
  ✅ Bands relatively uniform in thickness
  ✅ Mean silhouette score > 0.5
  ✅ Smooth, filled bands (not jagged)

Signs of a BAD silhouette plot:
  ❌ Some clusters thinner than mean line
  ❌ Negative values present (points misassigned)
  ❌ Very unequal band sizes (huge size imbalance)
  ❌ Mean silhouette score < 0.25
  ❌ Jagged, irregular band shapes
```

---

## 7. Silhouette vs Other Clustering Metrics

| Metric | Formula Summary | Range | Better | Ground Truth Needed | Notes |
|--------|----------------|-------|--------|---------------------|-------|
| **Silhouette** | $(b-a)/\max(a,b)$ | $[-1, +1]$ | Higher | ❌ No | Best general-purpose metric |
| **Davies-Bouldin** | $\frac{1}{K}\sum\max\frac{s_i+s_j}{d_{ij}}$ | $[0, \infty)$ | Lower | ❌ No | Simple, fast |
| **Calinski-Harabasz** | $\frac{SS_B/(K-1)}{SS_W/(N-K)}$ | $[0, \infty)$ | Higher | ❌ No | Fast, biased toward convex |
| **Adjusted Rand** | $(RI - E[RI])/(max-E[RI])$ | $[-1, +1]$ | Higher | ✅ Yes | Best when labels known |
| **NMI** | $MI(U,V)/H(U,V)$ | $[0, 1]$ | Higher | ✅ Yes | Info-theoretic |
| **Inertia** | $\sum\|x_i - \mu_k\|^2$ | $[0, \infty)$ | Lower | ❌ No | K-Means only, elbow method |
| **V-Measure** | $2hc/(h+c)$ | $[0, 1]$ | Higher | ✅ Yes | Harmonic mean of h+c |

### When to Use What

```python
# No ground truth (typical unsupervised case):
from sklearn.metrics import (
    silhouette_score,        # best general metric
    davies_bouldin_score,    # fast alternative
    calinski_harabasz_score  # biased but fast
)

# Ground truth available (benchmarking):
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    v_measure_score
)
```

### Davies-Bouldin Index (formula)

$$DB = \frac{1}{K} \sum_{i=1}^{K} \max_{j \neq i} \left( \frac{s_i + s_j}{d_{ij}} \right)$$

where $s_i$ = average distance from points in cluster $i$ to its centroid,
$d_{ij}$ = distance between centroids $i$ and $j$.

### Calinski-Harabasz Score (formula)

$$CH = \frac{SS_B / (K-1)}{SS_W / (N-K)}$$

where $SS_B$ = between-cluster sum of squares, $SS_W$ = within-cluster sum of squares.

---

## 8. No Gradient Descent — Why

**Simple statement:** Silhouette is a **measurement tool**, not an optimization algorithm. You run clustering first, then measure quality with silhouette. There is nothing to "train."

```
Silhouette is PURELY a metric:

  Step 1: Run any clustering algorithm (K-Means, DBSCAN, etc.)
  Step 2: Compute silhouette scores
  Step 3: Report / compare

No parameters → no gradients → no optimization loop

It's like asking: "How good is this clustering?"
Not: "Make the clustering better."

If you want to USE silhouette to SELECT K:
  → You loop over K values and pick the best
  → This is a hyperparameter search, NOT gradient descent
  → Each K runs a fresh clustering — no parameter updates
```

**The only "optimization" using silhouette is:**
```python
# Outer loop: silhouette-guided K search
best_k = max(range(2, 11),
             key=lambda k: silhouette_score(
                 X, KMeans(n_clusters=k).fit_predict(X)
             ))
# This is evaluation-based selection, not gradient optimization
```

---

## 9. Hyperparameters and Configuration

### `silhouette_score` Parameters

```python
from sklearn.metrics import silhouette_score

silhouette_score(
    X,                      # Feature matrix (n_samples, n_features)
                            # OR precomputed distance matrix
    labels,                 # Cluster labels (n_samples,) — integers
    metric='euclidean',     # Distance metric (same as used in clustering!)
    sample_size=None,       # Subsample for speed (None = use all)
    random_state=None,      # Seed for subsampling reproducibility
    **kwds                  # Extra kwargs passed to pairwise_distances
)
```

### `silhouette_samples` Parameters

```python
from sklearn.metrics import silhouette_samples

silhouette_samples(
    X,                      # Feature matrix or precomputed distances
    labels,                 # Cluster labels
    metric='euclidean',     # Distance metric
    **kwds
)
# Returns: array of shape (n_samples,) — one score per point
```

### Key Configuration Decisions

#### `metric` — Must Match Your Clustering

```python
# ⚠️ CRITICAL: Use the same metric as your clustering algorithm!

# K-Means uses Euclidean → use euclidean silhouette
sil = silhouette_score(X, labels, metric='euclidean')

# DBSCAN with cosine → use cosine silhouette
sil = silhouette_score(X, labels, metric='cosine')

# DBSCAN with precomputed distances
sil = silhouette_score(dist_matrix, labels, metric='precomputed')
```

#### `sample_size` — For Large Datasets

```python
# Full computation: O(N²) time and space — slow for large N
# Use sample_size for approximate but fast silhouette

# Exact (slow for N > 10,000)
sil_exact = silhouette_score(X, labels, sample_size=None)

# Approximate (fast for large datasets)
sil_approx = silhouette_score(X, labels,
                               sample_size=5000,
                               random_state=42)
# Usually within 0.01-0.02 of exact value for large N
```

#### Supported Metrics

```python
metrics = [
    'euclidean',     # default — L2 distance
    'manhattan',     # L1 distance — robust to outliers
    'cosine',        # angle-based — for text/NLP
    'l1', 'l2',      # aliases
    'minkowski',     # general Lp norm
    'chebyshev',     # L∞ norm
    'haversine',     # GPS coordinates
    'precomputed',   # supply your own distance matrix
]
```

---

## 10. Python from Scratch

```python
import numpy as np

def euclidean_distance(p, q):
    return np.sqrt(np.sum((p - q) ** 2))


def silhouette_single(X, labels, i):
    """
    Compute silhouette coefficient for a single point i.

    a(i) = mean intra-cluster distance
    b(i) = mean distance to nearest other cluster
    s(i) = (b(i) - a(i)) / max(a(i), b(i))
    """
    own_label = labels[i]
    own_cluster = [j for j in range(len(X))
                   if labels[j] == own_label and j != i]

    # Edge case: only one point in cluster
    if len(own_cluster) == 0:
        return 0.0

    # ── a(i): mean distance to own cluster ────────────────────
    a_i = np.mean([euclidean_distance(X[i], X[j])
                   for j in own_cluster])

    # ── b(i): mean distance to each other cluster, take min ───
    unique_labels = set(labels) - {own_label}

    if len(unique_labels) == 0:
        return 0.0   # only one cluster total

    mean_dists = []
    for label in unique_labels:
        other_cluster = [j for j in range(len(X))
                         if labels[j] == label]
        if len(other_cluster) == 0:
            continue
        mean_d = np.mean([euclidean_distance(X[i], X[j])
                          for j in other_cluster])
        mean_dists.append(mean_d)

    b_i = min(mean_dists)

    # ── s(i) ──────────────────────────────────────────────────
    s_i = (b_i - a_i) / max(a_i, b_i)
    return s_i


def silhouette_samples_scratch(X, labels):
    """Silhouette coefficient for every sample."""
    X      = np.array(X)
    labels = np.array(labels)
    return np.array([silhouette_single(X, labels, i)
                     for i in range(len(X))])


def silhouette_score_scratch(X, labels):
    """Mean silhouette score — overall clustering quality."""
    return np.mean(silhouette_samples_scratch(X, labels))


def silhouette_per_cluster(X, labels):
    """Mean silhouette score per cluster."""
    samples = silhouette_samples_scratch(X, labels)
    unique  = np.unique(labels)
    return {k: np.mean(samples[labels == k]) for k in unique}


# ── Vectorized version (faster — avoids nested loops) ─────────

def silhouette_score_vectorized(X, labels):
    """
    Vectorized silhouette using distance matrix.
    Time: O(N²) — same as sklearn.
    """
    from sklearn.metrics.pairwise import euclidean_distances

    X      = np.array(X)
    labels = np.array(labels)
    n      = len(X)

    # Full pairwise distance matrix
    D = euclidean_distances(X)

    unique_labels = np.unique(labels)
    a = np.zeros(n)
    b = np.full(n, np.inf)

    for label in unique_labels:
        mask = labels == label
        indices = np.where(mask)[0]

        # a(i): mean intra-cluster distances
        for i in indices:
            others = indices[indices != i]
            if len(others) > 0:
                a[i] = D[i, others].mean()
            else:
                a[i] = 0.0

        # b(i): mean distance to each other cluster
        for other_label in unique_labels:
            if other_label == label:
                continue
            other_mask    = labels == other_label
            other_indices = np.where(other_mask)[0]
            for i in indices:
                mean_d = D[i, other_indices].mean()
                if mean_d < b[i]:
                    b[i] = mean_d

    # Handle single-point clusters
    mask_single = a == 0
    s = np.where(mask_single, 0.0,
                 (b - a) / np.maximum(a, b))

    return s.mean(), s


# ── Test ────────────────────────────────────────────────────────
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score as sk_sil

X, _ = make_blobs(n_samples=100, centers=3,
                  cluster_std=0.5, random_state=42)
X_sc  = StandardScaler().fit_transform(X)

km     = KMeans(n_clusters=3, n_init=10, random_state=42)
labels = km.fit_predict(X_sc)

# Compare scratch vs sklearn
score_scratch = silhouette_score_scratch(X_sc, labels)
score_vec, _  = silhouette_score_vectorized(X_sc, labels)
score_sklearn = sk_sil(X_sc, labels)

print(f"Scratch (loop):       {score_scratch:.6f}")
print(f"Scratch (vectorized): {score_vec:.6f}")
print(f"Sklearn:              {score_sklearn:.6f}")
print(f"Match: {abs(score_scratch - score_sklearn) < 1e-6}")
```

---

## 11. Scikit-learn Implementation

### Basic Silhouette Score

```python
import numpy as np
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

# ── Data ──────────────────────────────────────────────────────
X, y_true = make_blobs(n_samples=500, centers=4,
                       cluster_std=0.8, random_state=42)
X_scaled  = StandardScaler().fit_transform(X)

# ── Cluster ───────────────────────────────────────────────────
km     = KMeans(n_clusters=4, n_init=10, random_state=42)
labels = km.fit_predict(X_scaled)

# ── Overall silhouette score ──────────────────────────────────
score = silhouette_score(X_scaled, labels, metric='euclidean')
print(f"Mean Silhouette Score: {score:.4f}")

# ── Per-sample silhouette values ──────────────────────────────
sample_scores = silhouette_samples(X_scaled, labels)
print(f"Min per-sample:  {sample_scores.min():.4f}")
print(f"Max per-sample:  {sample_scores.max():.4f}")
print(f"Mean per-sample: {sample_scores.mean():.4f}")

# ── Per-cluster silhouette ────────────────────────────────────
for k in range(4):
    k_scores = sample_scores[labels == k]
    print(f"Cluster {k}: n={len(k_scores):3d}, "
          f"mean_sil={k_scores.mean():.4f}, "
          f"min={k_scores.min():.4f}")

# ── Find misassigned points ───────────────────────────────────
negative_mask = sample_scores < 0
print(f"\nMisassigned points (s < 0): {negative_mask.sum()}")
print(f"Their indices: {np.where(negative_mask)[0]}")
```

### Silhouette with Precomputed Distance Matrix

```python
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances
import numpy as np

# For text/NLP clustering where cosine distance is used
# X_tfidf = TF-IDF matrix

dist_matrix = cosine_distances(X_tfidf)

# Use precomputed metric
score = silhouette_score(
    dist_matrix,
    labels,
    metric='precomputed'  # tell sklearn the matrix is already distances
)
print(f"Cosine Silhouette: {score:.4f}")
```

### Large Dataset — Subsampled Silhouette

```python
from sklearn.metrics import silhouette_score

# For N > 50,000 — full O(N²) is too slow
# Use sample_size for approximation

score_exact  = silhouette_score(X_large, labels)           # slow
score_approx = silhouette_score(X_large, labels,
                                 sample_size=10000,
                                 random_state=42)          # fast

print(f"Exact:       {score_exact:.4f}")
print(f"Approximate: {score_approx:.4f}")
# Usually differ by < 0.02
```

---

## 12. Full Pipeline — K-Means + Silhouette

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

# ── Data ──────────────────────────────────────────────────────
X, _ = make_blobs(n_samples=500, centers=4,
                  cluster_std=0.6, random_state=42)
X_sc  = StandardScaler().fit_transform(X)

# ── Step 1: Try K = 2 to 10, record silhouette + inertia ────
K_range    = range(2, 11)
sil_scores = []
inertias   = []
all_labels = {}

for k in K_range:
    km     = KMeans(n_clusters=k, n_init=20, random_state=42)
    labels = km.fit_predict(X_sc)
    sil    = silhouette_score(X_sc, labels)
    sil_scores.append(sil)
    inertias.append(km.inertia_)
    all_labels[k] = labels

optimal_k = K_range[np.argmax(sil_scores)]
print(f"Optimal K: {optimal_k}")
print(f"Best Silhouette: {max(sil_scores):.4f}")

# ── Step 2: Detailed silhouette plot for top 3 K values ──────
top_k = sorted(zip(sil_scores, K_range), reverse=True)[:3]

for sil_val, k in top_k:
    labels   = all_labels[k]
    sil_vals = silhouette_samples(X_sc, labels)
    mean_sil = silhouette_score(X_sc, labels)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = cm.tab10(np.linspace(0, 1, k))

    # Left: silhouette plot
    y_lower = 10
    for ki in range(k):
        ki_vals  = np.sort(sil_vals[labels == ki])
        y_upper  = y_lower + len(ki_vals)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ki_vals,
                          facecolor=colors[ki], alpha=0.7)
        ax1.text(-0.05, y_lower + len(ki_vals)/2, str(ki))
        y_lower  = y_upper + 10

    ax1.axvline(x=mean_sil, color='red', linestyle='--',
                label=f'Mean={mean_sil:.3f}')
    ax1.set_xlim([-0.2, 1.0])
    ax1.set_xlabel('Silhouette Coefficient')
    ax1.set_ylabel('Cluster')
    ax1.set_title(f'Silhouette Plot (K={k})')
    ax1.legend()

    # Right: cluster scatter
    scatter = ax2.scatter(X_sc[:, 0], X_sc[:, 1],
                          c=labels, cmap='tab10',
                          s=20, alpha=0.7)
    centers = KMeans(n_clusters=k, n_init=20,
                     random_state=42).fit(X_sc).cluster_centers_
    ax2.scatter(centers[:, 0], centers[:, 1],
                c='black', marker='X', s=200, zorder=5,
                label='Centroids')
    ax2.set_title(f'K-Means Clusters (K={k})')
    ax2.legend()

    plt.suptitle(f'K={k} | Mean Silhouette={mean_sil:.4f}',
                 fontsize=13)
    plt.tight_layout()
    plt.show()
```

---

## 13. Full Pipeline — DBSCAN + Silhouette

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons
from sklearn.neighbors import NearestNeighbors

X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)
X_sc  = StandardScaler().fit_transform(X)

# ── K-Distance plot to find eps ───────────────────────────────
min_samples = 5
nbrs = NearestNeighbors(n_neighbors=min_samples-1).fit(X_sc)
dists, _ = nbrs.kneighbors(X_sc)
k_dists  = np.sort(dists[:, -1])[::-1]

plt.figure(figsize=(8, 3))
plt.plot(k_dists, linewidth=2)
plt.axhline(y=0.3, color='red', linestyle='--', label='eps candidate = 0.3')
plt.title('K-Distance Graph')
plt.xlabel('Points (sorted)')
plt.ylabel('k-NN Distance')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ── Grid search eps + min_samples ─────────────────────────────
results = []
for eps in np.arange(0.1, 1.0, 0.05):
    for ms in range(3, 12):
        db     = DBSCAN(eps=eps, min_samples=ms)
        labels = db.fit_predict(X_sc)
        n_clus = len(set(labels)) - (1 if -1 in labels else 0)
        n_nois = (labels == -1).sum()
        valid  = labels != -1

        if n_clus < 2 or valid.sum() <= n_clus:
            continue

        try:
            sil = silhouette_score(X_sc[valid], labels[valid])
            results.append({
                'eps': round(eps, 3), 'min_samples': ms,
                'n_clusters': n_clus, 'n_noise': n_nois,
                'silhouette': sil
            })
        except Exception:
            continue

results.sort(key=lambda x: x['silhouette'], reverse=True)
print("Top 5 DBSCAN configs:")
for r in results[:5]:
    print(f"  eps={r['eps']:.2f}, min_samples={r['min_samples']:2d} → "
          f"K={r['n_clusters']}, noise={r['n_noise']:3d}, "
          f"sil={r['silhouette']:.4f}")

# ── Final DBSCAN with best params ────────────────────────────
best = results[0]
db_final = DBSCAN(eps=best['eps'], min_samples=best['min_samples'])
labels   = db_final.fit_predict(X_sc)
valid    = labels != -1

sil_vals = silhouette_samples(X_sc[valid], labels[valid])
mean_sil = silhouette_score(X_sc[valid], labels[valid])

print(f"\nFinal: eps={best['eps']}, min_samples={best['min_samples']}")
print(f"Clusters: {best['n_clusters']}, Noise: {best['n_noise']}")
print(f"Mean Silhouette: {mean_sil:.4f}")
```

---

## 14. Silhouette with All Clustering Algorithms

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import (
    KMeans, DBSCAN, AgglomerativeClustering,
    SpectralClustering, GaussianMixture, Birch
)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons

X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)
X_sc  = StandardScaler().fit_transform(X)

algorithms = {
    'KMeans (K=2)':
        KMeans(n_clusters=2, n_init=10, random_state=42),
    'KMeans (K=3)':
        KMeans(n_clusters=3, n_init=10, random_state=42),
    'DBSCAN (eps=0.3)':
        DBSCAN(eps=0.3, min_samples=5),
    'Agglomerative (K=2)':
        AgglomerativeClustering(n_clusters=2),
    'Spectral (K=2)':
        SpectralClustering(n_clusters=2, random_state=42,
                           affinity='nearest_neighbors'),
    'Birch (K=2)':
        Birch(n_clusters=2),
}

print(f"{'Algorithm':<25} {'N_Clusters':>10} {'Silhouette':>12}")
print("-" * 50)

for name, algo in algorithms.items():
    labels = algo.fit_predict(X_sc)
    valid  = labels != -1
    n_clus = len(set(labels[valid]))

    if n_clus < 2 or valid.sum() <= n_clus:
        print(f"{name:<25} {'N/A':>10} {'N/A':>12}")
        continue

    sil = silhouette_score(X_sc[valid], labels[valid])
    print(f"{name:<25} {n_clus:>10} {sil:>12.4f}")
```

---

## 15. Pros and Cons

### ✅ Advantages

| Advantage | Explanation |
|-----------|-------------|
| **No ground truth needed** | Pure unsupervised evaluation |
| **Interpretable range** | −1 to +1 with clear meaning |
| **Per-point insight** | Diagnose individual misassigned points |
| **Algorithm-agnostic** | Works with K-Means, DBSCAN, hierarchical, etc. |
| **Detects wrong K** | Low score clearly signals wrong number of clusters |
| **Visual (silhouette plot)** | Rich diagnostic — not just a single number |
| **Metric-flexible** | Works with any distance metric |

### ❌ Disadvantages

| Disadvantage | Explanation |
|--------------|-------------|
| **$O(N^2)$ complexity** | Slow for large datasets (use `sample_size`) |
| **Biased toward convex clusters** | Penalizes non-convex (e.g., crescent) shapes |
| **Ignores noise labels** | Must filter DBSCAN noise (label=−1) before computing |
| **Single-point clusters** | Score = 0, distorts mean |
| **Not absolute** | Score 0.6 in one domain ≠ score 0.6 in another |
| **Same metric required** | Must use same distance as clustering algorithm |

### When to Use Silhouette

```
✅ Choosing optimal K for K-Means
✅ Comparing different clustering algorithms
✅ Tuning eps and min_samples for DBSCAN
✅ Diagnosing cluster quality (silhouette plot)
✅ Finding misassigned points for re-labeling
✅ Validation without ground truth labels

❌ Very large datasets without subsampling (use sample_size=)
❌ Non-Euclidean clusters with Euclidean metric (use correct metric!)
❌ When clusters are known to be non-convex (use DBI or domain knowledge)
```

---

## 16. Interview Questions

**Q1: What does the silhouette score measure?**
> It measures clustering quality by combining cohesion (how close a point is to its own cluster, $a(i)$) and separation (how far it is from the nearest other cluster, $b(i)$). Formula: $s(i) = (b(i) - a(i)) / \max(a(i), b(i))$.

**Q2: What is $a(i)$ and $b(i)$?**
> $a(i)$ = mean distance from point $i$ to all other points in its own cluster (intra-cluster distance — want small). $b(i)$ = mean distance from point $i$ to all points in the nearest neighboring cluster (inter-cluster distance — want large).

**Q3: What does a silhouette score of −0.4 mean for a point?**
> The point is closer to a neighboring cluster than to its own cluster — it is likely misassigned. It would fit better in the neighboring cluster.

**Q4: Does silhouette score use gradient descent?**
> No. Silhouette is a pure measurement metric — it computes distances and averages. There is no optimization, no loss function, no parameter updates. It is computed after clustering, not during.

**Q5: What is the time complexity of silhouette score?**
> $O(N^2 \cdot d)$ — requires computing all pairwise distances. For large N, use `sample_size` parameter for an $O(S \cdot N \cdot d)$ approximation where $S$ is the subsample size.

**Q6: How do you use silhouette to choose K?**
> Run clustering for K = 2, 3, ..., max_K. Compute mean silhouette for each K. Choose the K with the highest mean silhouette score. Use a silhouette plot to visually confirm.

**Q7: Why must you exclude noise points (label=−1) before computing silhouette for DBSCAN?**
> Silhouette requires points to belong to a cluster. Noise points have no cluster, so $a(i)$ and $b(i)$ are not meaningful. Including them distorts the score. Filter: `X[labels != -1]`, `labels[labels != -1]`.

**Q8: What is a silhouette plot and what does it tell you?**
> A silhouette plot shows the silhouette coefficient of every point, grouped by cluster. Wide, uniform bands above the mean line indicate good clusters. Thin bands, bands below the mean, and negative values indicate poor clusters or wrong K.

**Q9: What are the differences between silhouette, Davies-Bouldin, and Calinski-Harabasz?**
> All are unsupervised metrics. Silhouette: $[-1,+1]$, higher better, most interpretable, $O(N^2)$. Davies-Bouldin: $[0, \infty)$, lower better, fast $O(N \cdot K)$. Calinski-Harabasz: $[0, \infty)$, higher better, fast but biased toward convex spherical clusters.

**Q10: Can silhouette score compare clusters from different algorithms?**
> Yes — it is algorithm-agnostic. You can compare K-Means vs DBSCAN vs hierarchical clustering on the same dataset using silhouette. Just ensure you use the same distance metric for fair comparison.

---

## 17. Resources

### 📘 Andrew Ng — ML Specialization (Coursera)
- **Course 3, Week 1**: Unsupervised Learning
  - K-Means algorithm and choosing K
  - Elbow method vs silhouette (comparison discussed)
  - Cluster quality evaluation overview
- 🔗 https://www.coursera.org/specializations/machine-learning-introduction

### 📗 Hands-On Machine Learning (Aurélien Géron)
- **Chapter 9**: Unsupervised Learning Techniques
  - Section 9.1: K-Means — silhouette score for K selection
  - Silhouette plot visualization with matplotlib
  - Comparison with inertia / elbow method
  - Code: full silhouette pipeline with K-Means
- 🔗 https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/

### 🎬 StatQuest with Josh Starmer (YouTube)
- **"K-means Clustering"** → silhouette mentioned for K selection
- **"Hierarchical Clustering"** → evaluation of clustering quality
- **"Principal Component Analysis (PCA)"** → dimensionality reduction before clustering
- 🔗 https://www.youtube.com/@statquest

### 📙 Introduction to Statistical Learning (ISLP — Python Edition)
- **Chapter 12**: Unsupervised Learning
  - 12.4.1: K-Means Clustering — quality measures
  - 12.4.3: Practical issues in clustering
  - Discussion of within-cluster variation and between-cluster separation
- Free PDF: 🔗 https://www.statlearning.com/

### 📜 Scikit-learn Documentation
- `silhouette_score`: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html
- `silhouette_samples`: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_samples.html
- Clustering metrics guide: https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation
- Silhouette plot example: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html

### 📄 Original Paper
- **Rousseeuw, P.J. (1987)** — "Silhouettes: A graphical aid to the interpretation and validation of cluster analysis" — Journal of Computational and Applied Mathematics
- This is the foundational paper defining the silhouette coefficient

---

## 🎯 Quick Cheat Sheet

```
What it is:    Cluster quality metric — no ground truth needed
Range:         [-1, +1]   higher = better
Complexity:    O(N² · d)  use sample_size for large N

Key formulas:
  a(i) = (1/|C_I|-1) Σ_{j∈C_I, j≠i} d(i,j)   ← intra-cluster (small = good)
  b(i) = min_{J≠I} (1/|C_J|) Σ_{j∈C_J} d(i,j) ← nearest other cluster (large = good)
  s(i) = (b(i) - a(i)) / max(a(i), b(i))        ← per-point score
  s̄   = (1/N) Σ s(i)                            ← overall mean score

Score interpretation:
  0.71–1.00 → Strong structure   ✅
  0.51–0.70 → Reasonable         ✅
  0.26–0.50 → Weak               ⚠️
  < 0.25    → No structure       ❌
  < 0       → Misassigned point  ❌

sklearn usage:
  silhouette_score(X, labels)           # mean score
  silhouette_samples(X, labels)         # per-point scores
  silhouette_score(X, labels,
                   metric='cosine')     # custom metric
  silhouette_score(X, labels,
                   sample_size=5000)    # fast approximation

Critical rules:
  → Scale features BEFORE clustering and silhouette
  → Use SAME metric as clustering algorithm
  → EXCLUDE noise points (label=-1) for DBSCAN
  → Single-point clusters get score = 0
  → No gradient descent — pure measurement tool
```

---

*Notes compiled from: Andrew Ng ML Specialization (Course 3) · Hands-On ML Ch.9 (Géron) · StatQuest clustering series · ISLP Ch.12 (James et al.) · Scikit-learn Docs · Rousseeuw (1987) original paper*
