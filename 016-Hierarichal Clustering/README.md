# 🌳 Hierarchical Clustering — Complete Job-Ready Notes

> **References:** Andrew Ng ML Specialization (Coursera) · Hands-On ML with Scikit-Learn (Aurélien Géron, 3rd ed.) · StatQuest with Josh Starmer (YouTube) · Scikit-Learn Documentation · Introduction to Statistical Learning with Python (ISLP, James et al.)

---

## 📚 Table of Contents

1. [What is Hierarchical Clustering? (Simple Statement)](#1-what-is-hierarchical-clustering-simple-statement)
2. [Why Hierarchical Clustering? Motivation](#2-why-hierarchical-clustering-motivation)
3. [The Math Behind Hierarchical Clustering](#3-the-math-behind-hierarchical-clustering)
4. [Agglomerative vs Divisive](#4-agglomerative-vs-divisive)
5. [Linkage Criteria — Deep Dive](#5-linkage-criteria--deep-dive)
6. [The Dendrogram](#6-the-dendrogram)
7. [Optimization View (No Gradient Descent — Here's Why)](#7-optimization-view-no-gradient-descent--heres-why)
8. [Hierarchical Clustering from Scratch (NumPy)](#8-hierarchical-clustering-from-scratch-numpy)
9. [Hierarchical Clustering with Scikit-Learn](#9-hierarchical-clustering-with-scikit-learn)
10. [Hierarchical Clustering with SciPy (Dendrogram)](#10-hierarchical-clustering-with-scipy-dendrogram)
11. [Hyperparameters & Tuning](#11-hyperparameters--tuning)
12. [Choosing the Number of Clusters](#12-choosing-the-number-of-clusters)
13. [Evaluation Metrics](#13-evaluation-metrics)
14. [Variants of Hierarchical Clustering](#14-variants-of-hierarchical-clustering)
15. [Hierarchical Clustering in ML Pipelines](#15-hierarchical-clustering-in-ml-pipelines)
16. [Comparison: Hierarchical vs K-Means vs DBSCAN](#16-comparison-hierarchical-vs-k-means-vs-dbscan)
17. [Limitations & When NOT to Use](#17-limitations--when-not-to-use)
18. [Common Interview Questions](#18-common-interview-questions)
19. [Resources](#19-resources)

---

## 1. What is Hierarchical Clustering? (Simple Statement)

> **Hierarchical Clustering builds a tree of clusters (called a dendrogram) by either merging the most similar pairs of clusters bottom-up (agglomerative) or splitting the most dissimilar clusters top-down (divisive). You can cut this tree at any height to get any number of flat clusters — no need to specify K in advance.**

**Analogy (StatQuest style):**
Think of sorting a deck of cards by similarity. Agglomerative clustering is like starting with every card in its own pile, then repeatedly combining the two most similar piles until everything is in one pile. Divisive is the opposite — start with one big pile and keep splitting. At the end you have a tree showing every merge/split decision made.

- **Type:** Unsupervised learning
- **Input:** Data matrix `X` of shape `(n, p)` OR a pairwise distance matrix `(n, n)`
- **Output:** A **dendrogram** (tree structure) + flat cluster labels at any chosen cut height
- **Key advantage:** No need to pre-specify K; the tree reveals all cluster structures at once

---

## 2. Why Hierarchical Clustering? Motivation

| Use Case | Example |
|---|---|
| Gene expression analysis | Group genes by expression similarity across conditions |
| Phylogenetics | Build evolutionary trees of species |
| Document taxonomy | Create hierarchical topic trees |
| Customer journey | Multi-level customer segmentation (segment → sub-segment) |
| Image segmentation | Hierarchical region merging |
| Social network analysis | Community detection at multiple resolutions |
| No fixed K needed | Explore data without committing to K upfront |

**Hierarchical vs K-Means:**

| | Hierarchical | K-Means |
|---|---|---|
| Specify K? | No (choose after) | Yes (before) |
| Result | Tree (dendrogram) | Flat partition |
| Deterministic? | Yes (given linkage) | No (random init) |
| Scalability | O(n³) or O(n²log n) | O(n·K·p·I) |
| Shape assumption | Depends on linkage | Spherical only |
| Interpretability | High (tree structure) | Medium |

---

## 3. The Math Behind Hierarchical Clustering

### 3.1 Notation

```
X         : Data matrix, shape (n, p)
d(xᵢ, xⱼ) : Distance between points i and j (scalar)
D         : Pairwise distance matrix, shape (n, n)
Cₐ, C_b   : Two clusters being merged
d(Cₐ, C_b): Distance between two clusters (depends on linkage)
h         : Height in dendrogram = distance at which merge occurred
```

### 3.2 Distance Metrics

The choice of distance metric is **independent** of the linkage criterion:

```
Euclidean distance (default, L2):
  d(xᵢ, xⱼ) = √(Σₖ (xᵢₖ - xⱼₖ)²) = ||xᵢ - xⱼ||₂

Manhattan distance (L1, robust to outliers):
  d(xᵢ, xⱼ) = Σₖ |xᵢₖ - xⱼₖ|

Cosine distance (for text/high-dimensional):
  d(xᵢ, xⱼ) = 1 - (xᵢ · xⱼ) / (||xᵢ||₂ ||xⱼ||₂)

Correlation distance:
  d(xᵢ, xⱼ) = 1 - corr(xᵢ, xⱼ)

Minkowski (generalized Lp):
  d(xᵢ, xⱼ) = (Σₖ |xᵢₖ - xⱼₖ|^p)^(1/p)
  p=1 → Manhattan, p=2 → Euclidean, p→∞ → Chebyshev
```

### 3.3 The Pairwise Distance Matrix

```
D = [d(xᵢ, xⱼ)] for all i,j

D is:
  - Symmetric:        D[i,j] = D[j,i]
  - Zero diagonal:    D[i,i] = 0
  - Non-negative:     D[i,j] ≥ 0

Shape: (n, n)   →   n(n-1)/2 unique values (upper triangle)
```

### 3.4 Lance-Williams Update Formula

When two clusters `Cₐ` and `C_b` are merged into `C_{a∪b}`, the distance to any other cluster `Cₓ` can be updated efficiently using the **Lance-Williams formula**:

```
d(C_{a∪b}, Cₓ) = αₐ · d(Cₐ, Cₓ) + α_b · d(C_b, Cₓ)
                + β · d(Cₐ, C_b) + γ · |d(Cₐ, Cₓ) - d(C_b, Cₓ)|

Parameters for each linkage:
┌─────────────┬────────────────────┬──────────────────────┬───────┬──────┐
│ Linkage     │ αₐ                 │ α_b                  │ β     │ γ    │
├─────────────┼────────────────────┼──────────────────────┼───────┼──────┤
│ Single      │ 1/2                │ 1/2                  │ 0     │ -1/2 │
│ Complete    │ 1/2                │ 1/2                  │ 0     │ +1/2 │
│ Average     │ nₐ/(nₐ+n_b)       │ n_b/(nₐ+n_b)         │ 0     │  0   │
│ Ward        │ (nₓ+nₐ)/(nₓ+nT)  │ (nₓ+n_b)/(nₓ+nT)    │-nₓ/nT │  0   │
│ Centroid    │ nₐ/(nₐ+n_b)       │ n_b/(nₐ+n_b)         │-nₐn_b/│  0   │
│             │                    │                      │(nₐ+n_b│      │
└─────────────┴────────────────────┴──────────────────────┴───────┴──────┘

where nₐ, n_b = cluster sizes, nT = nₐ + n_b, nₓ = size of Cₓ
```

This avoids recomputing all pairwise distances from scratch after each merge.

---

## 5. Linkage Criteria — Deep Dive

### 5.1 Single Linkage (Minimum / Nearest Neighbor)

```
d(Cₐ, C_b) = min { d(xᵢ, xⱼ) : xᵢ ∈ Cₐ, xⱼ ∈ C_b }
           = distance between closest points in the two clusters
```

- **Produces:** Long, chain-like clusters (chaining effect)
- **Strengths:** Detects elongated or irregular shapes; finds outliers
- **Weaknesses:** Very sensitive to noise/outliers; chaining effect
- **When to use:** Non-convex, elongated clusters

### 5.2 Complete Linkage (Maximum / Furthest Neighbor)

```
d(Cₐ, C_b) = max { d(xᵢ, xⱼ) : xᵢ ∈ Cₐ, xⱼ ∈ C_b }
           = distance between the furthest points in the two clusters
```

- **Produces:** Compact, spherical clusters of similar diameter
- **Strengths:** Robust to outliers; well-defined cluster boundaries
- **Weaknesses:** Sensitive to outliers in clusters; tends to break large clusters
- **When to use:** When you want compact, equally sized clusters

### 5.3 Average Linkage (UPGMA)

```
d(Cₐ, C_b) = (1 / |Cₐ| · |C_b|) Σ_{xᵢ ∈ Cₐ} Σ_{xⱼ ∈ C_b} d(xᵢ, xⱼ)
           = mean pairwise distance between all pairs across the two clusters
```

- **Produces:** Balanced compromise between single and complete
- **Strengths:** Less sensitive to outliers than single; less biased than complete
- **Weaknesses:** Computationally expensive; biased toward merging clusters with small variance
- **When to use:** General-purpose default after Ward

### 5.4 Ward Linkage (Minimum Variance)

```
d(Cₐ, C_b) = increase in total within-cluster variance when Cₐ and C_b are merged

           = (nₐ · n_b) / (nₐ + n_b) · ||μₐ - μ_b||²

where μₐ, μ_b are cluster centroids, nₐ, n_b are cluster sizes
```

Equivalently, Ward minimizes:

```
ΔJ = Σ_{x ∈ Cₐ∪C_b} ||x - μ_{a∪b}||²  -  Σ_{x ∈ Cₐ} ||x - μₐ||²  -  Σ_{x ∈ C_b} ||x - μ_b||²
```

- **Produces:** Compact, spherical clusters of similar size (most like K-Means)
- **Strengths:** Minimizes inertia (same objective as K-Means); best overall performance
- **Weaknesses:** Only valid for Euclidean distance; biased toward equal-size clusters
- **When to use:** **Default choice** for most problems with Euclidean distance

### 5.5 Centroid Linkage (UPGMC)

```
d(Cₐ, C_b) = ||μₐ - μ_b||²
           = squared Euclidean distance between cluster centroids
```

- Can produce **inversions** (non-monotone dendrogram — child merge height > parent)
- Less commonly used

### 5.6 Linkage Comparison Table

| Linkage | Distance Formula | Cluster Shape | Outlier Robust | sklearn default |
|---|---|---|---|---|
| **Ward** | ΔVariance | Spherical, equal | Moderate | ✅ Yes |
| **Complete** | max pairwise | Compact, spherical | Poor | No |
| **Average** | mean pairwise | Intermediate | Good | No |
| **Single** | min pairwise | Elongated, chains | Very poor | No |
| **Centroid** | centroid-centroid | Intermediate | Moderate | No |

---

## 6. The Dendrogram

### 6.1 What is a Dendrogram?

```
A dendrogram is a tree diagram where:
  - Leaves = individual data points
  - Internal nodes = merge events
  - Y-axis (height) = distance/dissimilarity at which merge occurred
  - X-axis = data points (order chosen to minimize branch crossings)
```

### 6.2 Reading a Dendrogram

```
Height 5 ──────────────────────────────── ← Cut here → 2 clusters
             │                    │
Height 3 ──  │         ───────────│──────  ← Cut here → 3 clusters
           ──┤──     ──┤──      ──┤──
Height 1   A  B     C   D      E   F
```

- **Horizontal line at height h** = cut the dendrogram → flat clusters
- **Lower merge** = more similar pair
- **Longer vertical line** = large gap between clusters (natural cluster boundary)
- **Rule:** Look for the longest vertical line that isn't crossed by a horizontal cut

### 6.3 Cophenetic Correlation Coefficient

Measures how faithfully the dendrogram preserves pairwise distances:

```
CPCC = corr(D_original, D_cophenetic)

where D_cophenetic[i,j] = height at which xᵢ and xⱼ first joined the same cluster

CPCC ∈ [-1, 1]:
  CPCC > 0.75 → good dendrogram
  CPCC > 0.90 → excellent
```

---

## 7. Optimization View (No Gradient Descent — Here's Why)

### 7.1 Why There's No Gradient Descent in Hierarchical Clustering

Hierarchical clustering does **not** use gradient descent because:

1. **Discrete optimization:** The merge/split decisions are discrete (which two clusters to merge), not continuous parameters
2. **No differentiable objective:** There's no single global objective function being minimized across all steps simultaneously
3. **Greedy algorithm:** Each step makes the locally optimal merge — this is greedy coordinate-wise optimization

### 7.2 What Ward Linkage DOES Optimize

Ward linkage has the closest connection to optimization:

```
At each step, Ward merges the pair (Cₐ, C_b) that minimizes the increase in total WCSS:

ΔJ(Cₐ, C_b) = (nₐ · n_b)/(nₐ + n_b) · ||μₐ - μ_b||²

Global objective (informal):
  J_total = Σₖ Σ_{xᵢ ∈ Cₖ} ||xᵢ - μₖ||²   (same as K-Means inertia!)
```

But this is **greedy** — each merge is locally optimal. Unlike K-Means (which iterates), hierarchical clustering never revises a merge decision.

### 7.3 Comparison to Gradient Descent

```
Algorithm          | Optimization Type       | Revisits Decisions?
─────────────────────────────────────────────────────────────────
K-Means            | Coordinate descent      | Yes (iterates)
Hierarchical       | Greedy (one-pass)       | No (irreversible)
DBSCAN             | Density-based scan      | No
GMM (EM)           | Gradient-like EM        | Yes (iterates)
Gradient Descent   | First-order continuous  | Yes (iterates)
```

### 7.4 Ward as Divisive GD Analog (Advanced)

For divisive clustering (top-down), one approach is to find the split maximizing:

```
Maximize: B(C) = n_L · n_R / (n_L + n_R) · ||μ_L - μ_R||²

This is a combinatorial optimization.
BisectingKMeans solves it approximately using K-Means(K=2) at each step.
```

---

## 8. Hierarchical Clustering from Scratch (NumPy)

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

class AgglomerativeFromScratch:
    """
    Agglomerative hierarchical clustering from scratch.
    Supports: single, complete, average, ward linkage.
    """

    def __init__(self, n_clusters=2, linkage='ward'):
        self.n_clusters = n_clusters
        self.linkage    = linkage
        self.labels_    = None
        self.linkage_matrix_ = []   # stores merge history (like scipy)

    # ── Distance between two clusters ───────────────────────────

    def _cluster_distance(self, X, cluster_a, cluster_b):
        pts_a = X[list(cluster_a)]
        pts_b = X[list(cluster_b)]
        D = cdist(pts_a, pts_b, metric='euclidean')

        if self.linkage == 'single':
            return D.min()
        elif self.linkage == 'complete':
            return D.max()
        elif self.linkage == 'average':
            return D.mean()
        elif self.linkage == 'ward':
            nA, nB = len(cluster_a), len(cluster_b)
            mu_a = pts_a.mean(axis=0)
            mu_b = pts_b.mean(axis=0)
            return (nA * nB) / (nA + nB) * np.sum((mu_a - mu_b)**2)
        else:
            raise ValueError(f"Unknown linkage: {self.linkage}")

    # ── Main fit ─────────────────────────────────────────────────

    def fit(self, X):
        X = np.array(X, dtype=float)
        n = len(X)

        # Each point starts as its own cluster
        # cluster_id → set of point indices
        clusters = {i: {i} for i in range(n)}
        next_id  = n   # new cluster IDs start after original n points

        # Track merge history: (id_a, id_b, distance, new_size)
        history = []

        active = set(range(n))

        while len(active) > 1:
            best_dist = np.inf
            best_pair = None

            # Find the closest pair of clusters
            active_list = list(active)
            for i in range(len(active_list)):
                for j in range(i + 1, len(active_list)):
                    a, b = active_list[i], active_list[j]
                    d = self._cluster_distance(X, clusters[a], clusters[b])
                    if d < best_dist:
                        best_dist = d
                        best_pair = (a, b)

            # Merge best pair
            a, b = best_pair
            new_cluster = clusters[a] | clusters[b]
            new_size    = len(new_cluster)

            history.append([a, b, best_dist, new_size])
            self.linkage_matrix_.append([a, b, best_dist, new_size])

            # Update clusters
            clusters[next_id] = new_cluster
            active.discard(a)
            active.discard(b)
            active.add(next_id)
            next_id += 1

        self.linkage_matrix_ = np.array(self.linkage_matrix_)
        self._X = X
        self._n = n
        self._history = history

        # Cut tree to get flat labels
        self.labels_ = self._cut_tree(self.n_clusters)
        return self

    def _cut_tree(self, k):
        """Cut dendrogram to get k clusters."""
        n = self._n
        # Start: each point is its own cluster
        clusters = {i: {i} for i in range(n)}
        next_id  = n

        # Replay merges, stop when k clusters remain
        merges_to_do = n - k
        for i, (a, b, dist, size) in enumerate(self._history):
            if i >= merges_to_do:
                break
            a, b = int(a), int(b)
            clusters[next_id] = clusters.pop(a) | clusters.pop(b)
            next_id += 1

        # Assign labels
        labels = np.zeros(n, dtype=int)
        for label, (cid, pts) in enumerate(clusters.items()):
            for idx in pts:
                labels[idx] = label
        return labels

    def fit_predict(self, X):
        return self.fit(X).labels_


# ────────────────────────────────────────────────────────────
# Demo
# ────────────────────────────────────────────────────────────
from sklearn.datasets import make_blobs

X, y_true = make_blobs(n_samples=30, centers=3, cluster_std=0.7,
                        random_state=42)

for linkage in ['single', 'complete', 'average', 'ward']:
    model = AgglomerativeFromScratch(n_clusters=3, linkage=linkage)
    labels = model.fit_predict(X)
    print(f"{linkage:8s}: labels = {labels}")
```

---

## 9. Hierarchical Clustering with Scikit-Learn

### 9.1 Basic Usage — AgglomerativeClustering

```python
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

X, y = make_blobs(n_samples=150, centers=4, cluster_std=0.8, random_state=42)

# Always scale
scaler = StandardScaler()
X_s = scaler.fit_transform(X)

# Fit
agg = AgglomerativeClustering(
    n_clusters  = 4,
    metric      = 'euclidean',   # distance metric
    linkage     = 'ward',        # linkage criterion
    compute_full_tree = 'auto',  # needed for dendrogram
)
labels = agg.fit_predict(X_s)

print("Labels      :", labels[:10])
print("n_clusters_ :", agg.n_clusters_)
print("n_leaves_   :", agg.n_leaves_)
print("n_connected_:", agg.n_connected_components_)
```

### 9.2 Key Attributes After Fitting

```python
agg.labels_                  # (n,) cluster assignment for each point
agg.n_clusters_              # Number of clusters found
agg.n_leaves_                # Number of leaves (= n_samples)
agg.n_connected_components_  # Number of connected components
agg.n_features_in_           # Number of input features

# With compute_full_tree=True:
agg.children_                # (n-1, 2) merge matrix — which nodes merged at each step
agg.distances_               # (n-1,) distances at each merge (needs compute_distances=True)
```

### 9.3 Without Specifying n_clusters (Use distance_threshold)

```python
# Cut dendrogram at a fixed distance instead of fixed K
agg_dist = AgglomerativeClustering(
    n_clusters         = None,   # must be None when using distance_threshold
    distance_threshold = 10.0,   # merge up to this distance
    linkage            = 'ward',
    compute_full_tree  = True,   # required when using distance_threshold
)
labels_dist = agg_dist.fit_predict(X_s)
print(f"Clusters found with threshold: {agg_dist.n_clusters_}")
```

### 9.4 Linkage Comparison — All Four

```python
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_moons, make_blobs
import matplotlib.pyplot as plt

datasets = {
    'Blobs' : make_blobs(n_samples=200, centers=3, cluster_std=0.8, random_state=42)[0],
    'Moons' : make_moons(n_samples=200, noise=0.05, random_state=42)[0],
}

linkages = ['ward', 'complete', 'average', 'single']

fig, axes = plt.subplots(len(datasets), len(linkages),
                          figsize=(16, 8))

for row, (name, X_data) in enumerate(datasets.items()):
    X_sc = StandardScaler().fit_transform(X_data)
    for col, link in enumerate(linkages):
        agg = AgglomerativeClustering(
            n_clusters=2 if name=='Moons' else 3,
            linkage=link,
            metric='euclidean' if link != 'ward' else 'euclidean'
        )
        lbl = agg.fit_predict(X_sc)
        axes[row, col].scatter(X_sc[:, 0], X_sc[:, 1],
                                c=lbl, cmap='Set1', s=15, alpha=0.8)
        axes[row, col].set_title(f"{name} | {link}", fontsize=10)
        axes[row, col].axis('off')

plt.suptitle("Linkage Comparison", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

### 9.5 Using a Connectivity Matrix (Structured Clustering)

```python
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import AgglomerativeClustering

# Connectivity: only merge neighboring points (preserves spatial structure)
connectivity = kneighbors_graph(X_s, n_neighbors=10, include_self=False)

agg_conn = AgglomerativeClustering(
    n_clusters   = 4,
    linkage      = 'ward',
    connectivity = connectivity   # restrict merges to neighbors
)
labels_conn = agg_conn.fit_predict(X_s)
```

### 9.6 Using Precomputed Distance Matrix

```python
from scipy.spatial.distance import pdist, squareform

# Compute custom distance (e.g., correlation-based)
D_condensed = pdist(X_s, metric='correlation')
D_square    = squareform(D_condensed)

agg_pre = AgglomerativeClustering(
    n_clusters = 4,
    metric     = 'precomputed',   # tells sklearn D is already distances
    linkage    = 'complete',      # Ward requires Euclidean — can't use precomputed
)
labels_pre = agg_pre.fit_predict(D_square)
```

> ⚠️ **Ward linkage requires `metric='euclidean'`** — it minimizes variance, which only makes sense in Euclidean space. Use `complete` or `average` with precomputed distances.

---

## 10. Hierarchical Clustering with SciPy (Dendrogram)

### 10.1 Full Dendrogram Workflow

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import (
    linkage, dendrogram, fcluster,
    cut_tree, cophenet, inconsistent
)
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

iris = load_iris()
X = StandardScaler().fit_transform(iris.data)
labels_true = iris.target

# Step 1: Compute condensed distance matrix
dist_condensed = pdist(X, metric='euclidean')   # shape: n*(n-1)/2

# Step 2: Linkage (builds the merge tree)
Z = linkage(X, method='ward', metric='euclidean')
# Z shape: (n-1, 4)
# Z[i] = [cluster_a, cluster_b, distance, new_cluster_size]

print(f"Linkage matrix shape: {Z.shape}")
print("First 5 merges:\n", Z[:5])
```

### 10.2 Drawing the Dendrogram

```python
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Full dendrogram
ax = axes[0]
dendrogram(
    Z,
    ax               = ax,
    leaf_rotation    = 90,
    leaf_font_size   = 6,
    color_threshold  = 0,        # 0 = all black; positive = auto-color
    above_threshold_color = 'gray',
    show_contracted  = True,
)
ax.set_title("Full Dendrogram (Ward Linkage)")
ax.set_xlabel("Sample Index")
ax.set_ylabel("Distance (Ward)")

# Truncated dendrogram (last p merges)
ax = axes[1]
dendrogram(
    Z,
    ax             = ax,
    truncate_mode  = 'lastp',    # show only last p merges
    p              = 10,
    leaf_rotation  = 45,
    leaf_font_size = 10,
    show_leaf_counts = True,     # show count in parentheses
    color_threshold  = Z[-3, 2], # color clusters from cut at 3 clusters
)
ax.axhline(y=Z[-3, 2], color='red', linestyle='--',
           label=f'Cut → 3 clusters (h={Z[-3,2]:.2f})')
ax.set_title("Truncated Dendrogram (last 10 merges)")
ax.legend()

plt.tight_layout()
plt.show()
```

### 10.3 Cutting the Tree to Get Flat Clusters

```python
# Method 1: by number of clusters
labels_3 = fcluster(Z, t=3, criterion='maxclust')
print("3 clusters:", labels_3)  # note: 1-indexed!

# Method 2: by distance threshold
labels_thresh = fcluster(Z, t=5.0, criterion='distance')
print("Distance threshold 5.0:", np.unique(labels_thresh))

# Method 3: by inconsistency threshold
labels_incon = fcluster(Z, t=1.5, criterion='inconsistent', depth=2)

# Method 4: cut_tree gives all possible cuts at once
all_cuts = cut_tree(Z, n_clusters=[2, 3, 4, 5])
print("Cut matrix shape:", all_cuts.shape)   # (n_samples, 4)
```

### 10.4 Cophenetic Correlation

```python
c, coph_dists = cophenet(Z, dist_condensed)
print(f"Cophenetic Correlation Coefficient: {c:.4f}")
# > 0.75 → good; > 0.90 → excellent
```

### 10.5 All Four Linkages — Dendrogram Comparison

```python
from scipy.cluster.hierarchy import linkage, dendrogram

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
methods = ['single', 'complete', 'average', 'ward']
colors  = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']

for ax, method, color in zip(axes, methods, colors):
    Z_m = linkage(X, method=method)
    c, _ = cophenet(Z_m, dist_condensed)
    dendrogram(Z_m, ax=ax, leaf_font_size=0,
               link_color_func=lambda k: color)
    ax.set_title(f"{method.capitalize()}\nCPCC={c:.3f}", fontsize=11)
    ax.set_xlabel("Samples")
    ax.set_ylabel("Distance")

plt.suptitle("Linkage Method Comparison — Iris Dataset", fontsize=13)
plt.tight_layout()
plt.show()
```

---

## 11. Hyperparameters & Tuning

### sklearn `AgglomerativeClustering` Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_clusters` | int or None | 2 | Number of clusters. Set `None` to use `distance_threshold` instead. |
| `metric` | str or callable | `'euclidean'` | Distance metric. Any scipy metric, `'precomputed'` for distance matrix input. Ward requires `'euclidean'`. |
| `linkage` | str | `'ward'` | Linkage criterion: `'ward'`, `'complete'`, `'average'`, `'single'` |
| `distance_threshold` | float or None | None | Threshold to cut dendrogram. Use instead of `n_clusters`. |
| `connectivity` | array or callable | None | Connectivity matrix restricting which clusters can merge. |
| `compute_full_tree` | bool or `'auto'` | `'auto'` | Build full tree (needed for `distance_threshold` or dendrogram export). |
| `compute_distances` | bool | False | Store merge distances in `distances_` attribute. |
| `memory` | str or Memory | None | Cache results (useful for repeated runs). |

### scipy `linkage` Parameters

| Parameter | Options | Description |
|---|---|---|
| `method` | `'single'`, `'complete'`, `'average'`, `'weighted'`, `'centroid'`, `'median'`, `'ward'` | Linkage algorithm |
| `metric` | Any scipy distance metric string or callable | Distance function between observations |
| `optimal_ordering` | bool (default False) | Reorder leaves to minimize distances between adjacent leaves (nicer dendrogram) |

```python
# Tuning example: try all linkages, score with silhouette
from sklearn.metrics import silhouette_score

results = {}
for link in ['ward', 'complete', 'average', 'single']:
    # Ward only works with euclidean
    metric = 'euclidean'
    for k in range(2, 8):
        agg = AgglomerativeClustering(n_clusters=k, linkage=link, metric=metric)
        lbl = agg.fit_predict(X_s)
        sil = silhouette_score(X_s, lbl)
        results[(link, k)] = sil

best = max(results, key=results.get)
print(f"Best: linkage={best[0]}, k={best[1]}, silhouette={results[best]:.4f}")
```

---

## 12. Choosing the Number of Clusters

### Method 1: Dendrogram — Longest Vertical Line

```python
from scipy.cluster.hierarchy import linkage, dendrogram
import numpy as np

Z = linkage(X_s, method='ward')

# Find the largest gap between successive merge heights
heights = Z[:, 2]
gaps    = np.diff(heights)
k_optimal = len(heights) - np.argmax(gaps[::-1])   # count from top
print(f"Suggested K (largest gap): {k_optimal}")

# Visual cut
cut_height = (heights[-k_optimal] + heights[-k_optimal + 1]) / 2

plt.figure(figsize=(10, 5))
dendrogram(Z, leaf_font_size=0)
plt.axhline(y=cut_height, color='red', linestyle='--',
            label=f'Cut → {k_optimal} clusters')
plt.title("Dendrogram — Longest Gap Method")
plt.legend()
plt.tight_layout()
plt.show()
```

### Method 2: Silhouette Score

```python
from sklearn.metrics import silhouette_score

sil_scores = {}
for k in range(2, 12):
    agg = AgglomerativeClustering(n_clusters=k, linkage='ward')
    lbl = agg.fit_predict(X_s)
    sil_scores[k] = silhouette_score(X_s, lbl)

best_k = max(sil_scores, key=sil_scores.get)
print(f"Best K (silhouette): {best_k}, score={sil_scores[best_k]:.4f}")

plt.plot(list(sil_scores.keys()), list(sil_scores.values()), 'bo-')
plt.xlabel("K"); plt.ylabel("Silhouette Score")
plt.title("Silhouette vs K — AgglomerativeClustering")
plt.grid(True, alpha=0.3); plt.show()
```

### Method 3: Inconsistency Criterion

```python
from scipy.cluster.hierarchy import inconsistent

# Measures how different each merge is from its neighborhood merges
# depth=2 means look at merges within 2 levels
incon = inconsistent(Z, depth=2)
# incon[:, 3] = inconsistency coefficient
# High value at a merge → natural cluster boundary
print("Inconsistency for last 5 merges:\n", incon[-5:])
```

### Method 4: Elbow on Within-Cluster Variance

```python
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score

db_scores  = {}
ch_scores  = {}
for k in range(2, 12):
    agg = AgglomerativeClustering(n_clusters=k, linkage='ward')
    lbl = agg.fit_predict(X_s)
    db_scores[k] = davies_bouldin_score(X_s, lbl)
    ch_scores[k] = calinski_harabasz_score(X_s, lbl)

best_db = min(db_scores, key=db_scores.get)
best_ch = max(ch_scores, key=ch_scores.get)
print(f"Davies-Bouldin best K: {best_db}")
print(f"Calinski-Harabasz best K: {best_ch}")
```

---

## 13. Evaluation Metrics

### 13.1 Internal Metrics (No True Labels)

```python
from sklearn.metrics import (
    silhouette_score,        # (b-a)/max(a,b) per point; [-1,1] ↑ better
    davies_bouldin_score,    # avg max(σᵢ+σⱼ)/d(μᵢ,μⱼ); [0,∞] ↓ better
    calinski_harabasz_score, # between/within ratio; [0,∞] ↑ better
)
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

Z = linkage(X_s, method='ward')

# Cophenetic Correlation — dendrogram quality
cpcc, _ = cophenet(Z, pdist(X_s))
print(f"CPCC: {cpcc:.4f}")   # > 0.75 = good

# Flat cluster metrics
agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
lbl = agg.fit_predict(X_s)

sil = silhouette_score(X_s, lbl)
db  = davies_bouldin_score(X_s, lbl)
ch  = calinski_harabasz_score(X_s, lbl)
print(f"Silhouette={sil:.4f}, Davies-Bouldin={db:.4f}, C-H={ch:.4f}")
```

### 13.2 External Metrics (When True Labels Available)

```python
from sklearn.metrics import (
    adjusted_rand_score,          # ARI: [-1,1] ↑ — accounts for chance
    adjusted_mutual_info_score,   # AMI: [0,1] ↑
    normalized_mutual_info_score, # NMI: [0,1] ↑
    homogeneity_score,            # each cluster = 1 class only
    completeness_score,           # each class = 1 cluster only
    v_measure_score,              # harmonic mean of H and C
    fowlkes_mallows_score,        # geometric mean of precision/recall
)

agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
lbl = agg.fit_predict(X_s)

print(f"ARI  : {adjusted_rand_score(y, lbl):.4f}")
print(f"AMI  : {adjusted_mutual_info_score(y, lbl):.4f}")
print(f"NMI  : {normalized_mutual_info_score(y, lbl):.4f}")
print(f"Hom  : {homogeneity_score(y, lbl):.4f}")
print(f"Comp : {completeness_score(y, lbl):.4f}")
print(f"V    : {v_measure_score(y, lbl):.4f}")
```

---

## 14. Variants of Hierarchical Clustering

### 14.1 BIRCH — Scalable Hierarchical

```python
from sklearn.cluster import Birch

# Builds a Clustering Feature Tree (CF-Tree) incrementally
# Good for large datasets
birch = Birch(
    n_clusters    = 4,
    threshold     = 0.5,   # radius threshold for subclusters
    branching_factor = 50, # max children per CF node
)
labels_birch = birch.fit_predict(X_s)
```

### 14.2 OPTICS — Hierarchical Density-Based

```python
from sklearn.cluster import OPTICS

# Generalizes DBSCAN, produces a reachability plot (dendrogram analog)
optics = OPTICS(
    min_samples  = 5,
    xi           = 0.05,     # steepness for cluster extraction
    min_cluster_size = 0.05,
    metric       = 'euclidean',
    algorithm    = 'auto',   # 'kd_tree', 'ball_tree', 'brute'
)
labels_optics = optics.fit_predict(X_s)
```

### 14.3 Feature Agglomeration (Hierarchical for Features)

```python
from sklearn.cluster import FeatureAgglomeration

# Cluster FEATURES (columns) instead of samples
# Useful for dimensionality reduction when p >> n
fa = FeatureAgglomeration(
    n_clusters = 10,      # reduce to 10 feature groups
    linkage    = 'ward',
    metric     = 'euclidean',
)
X_reduced = fa.fit_transform(X_s)     # (n, 10)
print(f"Reduced from {X_s.shape[1]} to {X_reduced.shape[1]} features")
print(f"Feature labels: {fa.labels_}")  # which cluster each feature belongs to
```

### 14.4 Comparison Table

| Method | Type | Scalable | Shape | Key Feature |
|---|---|---|---|---|
| AgglomerativeClustering | Bottom-up | O(n²log n) | Depends on linkage | Dendrogram, flexible |
| BIRCH | Bottom-up | High | Spherical | CF-Tree, streaming |
| OPTICS | Density | Medium | Arbitrary | Reachability plot |
| FeatureAgglomeration | Feature-level | Medium | Any | Reduces features |
| Divisive (DIANA) | Top-down | Low | Any | More natural splits |

---

## 15. Hierarchical Clustering in ML Pipelines

### 15.1 Full Pipeline with sklearn

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.datasets import load_wine

X, y = load_wine(return_X_y=True)

# Build pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca',    PCA(n_clusters=8, random_state=42)),  # reduce dims first
    ('agg',    AgglomerativeClustering(n_clusters=3, linkage='ward'))
])

# Note: fit_predict for clustering pipelines
X_s = StandardScaler().fit_transform(X)
X_pca = PCA(n_components=8, random_state=42).fit_transform(X_s)
labels = AgglomerativeClustering(n_clusters=3, linkage='ward').fit_predict(X_pca)

sil = silhouette_score(X_pca, labels)
ari = adjusted_rand_score(y, labels)
print(f"Silhouette={sil:.4f}, ARI={ari:.4f}")
```

### 15.2 Heatmap with Hierarchical Clustering (Seaborn Clustermap)

```python
import seaborn as sns
import pandas as pd

# Load sample data
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = [iris.target_names[i] for i in iris.target]

# Clustermap = heatmap + dendrogram on both axes
g = sns.clustermap(
    df.drop('species', axis=1),
    method        = 'ward',       # linkage
    metric        = 'euclidean',  # distance
    standard_scale = 1,           # scale columns (0=rows, 1=cols)
    cmap          = 'RdYlBu_r',
    figsize       = (10, 10),
    row_colors    = pd.Categorical(df['species']).codes,  # color rows by class
    dendrogram_ratio = 0.2,
    cbar_pos      = (0.02, 0.8, 0.03, 0.18),
)
g.fig.suptitle("Iris Clustermap (Ward Linkage)", y=1.02)
plt.show()
```

### 15.3 Gene Expression Style Analysis

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Simulate gene expression data: genes × samples
np.random.seed(42)
n_genes, n_samples = 50, 20
expr = np.random.randn(n_genes, n_samples)

# Add some block structure (3 gene groups)
expr[:15] += 2 * np.sin(np.arange(n_samples) / 3)
expr[15:30] -= 2 * np.cos(np.arange(n_samples) / 4)

df_expr = pd.DataFrame(
    expr,
    index   = [f"Gene_{i:03d}" for i in range(n_genes)],
    columns = [f"Sample_{j:02d}" for j in range(n_samples)]
)

g = sns.clustermap(
    df_expr,
    method  = 'average',
    metric  = 'correlation',   # correlation distance for gene expression
    cmap    = 'RdBu_r',
    center  = 0,
    figsize = (12, 10),
    yticklabels = True,
    xticklabels = True,
)
plt.title("Gene Expression Heatmap + Dendrogram")
plt.show()
```

---

## 16. Comparison: Hierarchical vs K-Means vs DBSCAN

| Property | Hierarchical | K-Means | DBSCAN |
|---|---|---|---|
| Specify K? | No (choose after) | Yes (before) | No |
| Deterministic? | Yes | No (random init) | Yes |
| Handles noise/outliers | Partially | No | Yes (labels as -1) |
| Shape assumption | Depends on linkage | Spherical | Arbitrary |
| Scalability | O(n²log n) | O(n·K·p·I) | O(n log n) |
| Memory | O(n²) | O(n·K) | O(n) |
| Gives tree structure? | Yes | No | No |
| Works with any distance? | Yes | Euclidean only | Yes |
| Soft assignments? | No | No | No |
| Best for | Biology, taxonomy, small-medium n | Large n, spherical | Spatial, arbitrary shape |

---

## 17. Limitations & When NOT to Use

| Limitation | Details | Alternative |
|---|---|---|
| **O(n²) memory** | Distance matrix requires n² storage | Mini-Batch K-Means, DBSCAN for large n |
| **O(n³) / O(n²log n) time** | Very slow for n > 10,000 | BIRCH, Mini-Batch K-Means |
| **Irreversible merges** | A bad early merge can't be fixed | Run multiple times? No guarantee |
| **Ward = spherical only** | Same weakness as K-Means | Use single/complete with non-Euclidean metric |
| **Sensitive to outliers** | Single linkage worst; complete better | Complete/Ward linkage; remove outliers first |
| **No predict for new points** | sklearn's AgglomerativeClustering can't assign new points | Use KMeans or fit a KNN classifier on top |
| **Inverted dendrograms** | Centroid/median linkage can produce non-monotone trees | Use Ward, complete, or average |

```python
# Workaround: predict new points using KNN on existing labels
from sklearn.neighbors import KNeighborsClassifier

agg = AgglomerativeClustering(n_clusters=4, linkage='ward')
train_labels = agg.fit_predict(X_s)

# Train KNN to assign new points
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_s, train_labels)

# Predict new points
X_new_s = scaler.transform(X_new)
new_labels = knn.predict(X_new_s)
```

---

## 18. Common Interview Questions

### Q1: What is hierarchical clustering?
> An unsupervised method that builds a tree (dendrogram) of clusters. Agglomerative (bottom-up): each point starts as its own cluster; pairs merge greedily. Divisive (top-down): all points start in one cluster; recursively split. The tree can be cut at any height to produce flat clusters.

### Q2: What is a dendrogram and how do you use it to choose K?
> A dendrogram is a binary tree where leaves = data points, internal nodes = merges, and y-axis height = merge distance. To choose K: look for the longest vertical line that isn't crossed by any horizontal cut — cutting there gives natural clusters. Alternatively, cut at a specific distance threshold.

### Q3: What are the differences between linkage methods?
> **Single:** min pairwise distance → chains, elongated clusters, sensitive to noise. **Complete:** max pairwise distance → compact, spherical clusters. **Average:** mean pairwise distance → balanced. **Ward:** minimizes increase in within-cluster variance → best overall, spherical clusters like K-Means.

### Q4: Why is Ward linkage similar to K-Means?
> Both minimize within-cluster sum of squares (WCSS / inertia). Ward's merge criterion `ΔJ = (nₐ·n_b)/(nₐ+n_b) · ||μₐ-μ_b||²` is exactly the increase in total inertia from merging. The difference: K-Means iteratively reassigns and recomputes; Ward greedily merges without revision.

### Q5: What is the time and space complexity?
> Naive implementation: O(n³) time, O(n²) space. With optimized algorithms (SLINK for single, CLINK for complete): O(n²) time. This makes hierarchical clustering infeasible for n > 10,000–50,000 without approximations (BIRCH, etc.).

### Q6: Can hierarchical clustering predict new data points?
> Not directly — `AgglomerativeClustering` in sklearn has no `predict()` method. Workaround: train a KNN classifier on the cluster labels assigned during fit, then use KNN to assign new points.

### Q7: What is the Cophenetic Correlation Coefficient?
> CPCC measures how well the dendrogram preserves original pairwise distances. It's the correlation between the original distance matrix and the cophenetic distance matrix (height at which each pair first merges). CPCC > 0.75 = good representation.

### Q8: When to use hierarchical over K-Means?
> Use hierarchical when: (1) you don't know K in advance; (2) you want to explore cluster structure at multiple granularities; (3) data is small-medium (n < 10,000); (4) you need interpretable hierarchy (biology, taxonomy); (5) you want to use non-Euclidean distances.

### Q9: What is the Lance-Williams formula?
> A recursive update formula that recomputes the distance between a newly merged cluster and all other clusters without recomputing all pairwise distances from scratch. It generalizes all standard linkage methods with four parameters (αₐ, α_b, β, γ), making hierarchical clustering computationally efficient.

### Q10: Single linkage vs Complete linkage — practical difference?
> Single linkage suffers from "chaining" — one noisy point can bridge two very different clusters. Complete linkage produces more compact clusters but can break large natural clusters. Ward and average are usually better defaults.

---

## 19. Resources

### 📖 Books

| Book | Chapter | Topic |
|---|---|---|
| **Hands-On ML** (Géron, 3rd ed.) | Chapter 9 | Agglomerative clustering, BIRCH, dendrogram |
| **ISLP** (James et al.) | Chapter 12.4.2 | Hierarchical clustering, linkage, dendrogram reading |
| **ESL** (Hastie et al.) | Chapter 14.3.12 | Hierarchical methods, average and Ward linkage |
| **Pattern Recognition** (Bishop) | Chapter 9 | Mixture models and EM |

### 🎓 Andrew Ng — ML Specialization (Coursera)

- **Course 3: Unsupervised Learning, Recommenders, Reinforcement Learning**
  - Week 1: Clustering overview (K-Means is primary focus; hierarchical discussed for comparison)
  - Supplementary: Andrew Ng Stanford CS229 Lecture Notes cover hierarchical clustering math in detail
- Recommended: Stanford CS229 Lecture 13 notes (free PDF) — full Ward linkage derivation

### 📺 StatQuest with Josh Starmer (YouTube)

| Video Title | Search Query |
|---|---|
| Hierarchical Clustering | `StatQuest hierarchical clustering` |
| Dendrogram | `StatQuest dendrogram` |
| Heatmaps and Dendrograms | `StatQuest heatmap dendrogram` |
| Clustering methods comparison | `StatQuest clustering comparison` |

### 📚 Scikit-Learn Documentation

| Resource | URL Suffix |
|---|---|
| `AgglomerativeClustering` | `/stable/modules/generated/sklearn.cluster.AgglomerativeClustering` |
| `FeatureAgglomeration` | `/stable/modules/generated/sklearn.cluster.FeatureAgglomeration` |
| `BIRCH` | `/stable/modules/generated/sklearn.cluster.Birch` |
| Clustering User Guide | `/stable/modules/clustering.html#hierarchical-clustering` |
| Plot dendrogram example | `/stable/auto_examples/cluster/plot_agglomerative_dendrogram` |

> Base URL: `https://scikit-learn.org`

### 📚 SciPy Documentation

| Resource |
|---|
| `scipy.cluster.hierarchy.linkage` |
| `scipy.cluster.hierarchy.dendrogram` |
| `scipy.cluster.hierarchy.fcluster` |
| `scipy.cluster.hierarchy.cophenet` |

> Base URL: `https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html`

### 📄 Key Papers

| Paper | Note |
|---|---|
| Ward (1963) | Original Ward linkage paper |
| Lance & Williams (1967) | General update formula for all linkages |
| Murtagh & Contreras (2012) | Algorithms for hierarchical clustering survey |
| Zhang et al. (1996) | BIRCH: Balanced Iterative Reducing and Clustering |

---

## 🗂️ Quick Reference Cheat Sheet

```python
# ── sklearn AgglomerativeClustering ──────────────────────────
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

X_s = StandardScaler().fit_transform(X)   # ALWAYS scale

agg = AgglomerativeClustering(
    n_clusters          = 3,          # or None (use distance_threshold)
    metric              = 'euclidean',# distance metric; 'precomputed' for D matrix
    linkage             = 'ward',     # 'ward'|'complete'|'average'|'single'
    distance_threshold  = None,       # alternative to n_clusters
    connectivity        = None,       # restrict merges (kneighbors_graph)
    compute_full_tree   = 'auto',     # True for dendrogram export
    compute_distances   = False,      # store merge distances in .distances_
)
labels = agg.fit_predict(X_s)

# ── scipy dendrogram ──────────────────────────────────────────
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from scipy.spatial.distance import pdist

Z = linkage(X_s, method='ward', metric='euclidean')
# Z[i] = [cluster_a, cluster_b, distance, new_size]

# Draw dendrogram
dendrogram(Z, truncate_mode='lastp', p=12, leaf_font_size=10)

# Cut to K clusters
labels_k = fcluster(Z, t=3, criterion='maxclust')   # 1-indexed!
labels_d = fcluster(Z, t=5.0, criterion='distance')

# Quality
cpcc, _ = cophenet(Z, pdist(X_s))

# ── Linkage guide ─────────────────────────────────────────────
# Ward     → minimize ΔVariance; spherical clusters; best default
# Complete → max distance; compact; robust to chaining
# Average  → mean distance; balanced; moderate robustness
# Single   → min distance; chains; good for non-convex shapes

# ── Key metrics ──────────────────────────────────────────────
from sklearn.metrics import (
    silhouette_score,            # internal [-1,1] ↑
    davies_bouldin_score,        # internal [0,∞] ↓
    calinski_harabasz_score,     # internal [0,∞] ↑
    adjusted_rand_score,         # external [-1,1] ↑
    normalized_mutual_info_score # external [0,1] ↑
)

# ── When no predict() needed — use KNN ───────────────────────
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_s, labels)
new_labels = knn.predict(X_new_s)
```

---

*Notes compiled for ML/DL job readiness. Covers hierarchical clustering theory, all linkage math (single/complete/average/Ward), Lance-Williams formula, dendrogram reading, why no gradient descent applies (greedy coordinate optimization), NumPy implementation, full sklearn + scipy API, hyperparameters, choosing K (4 methods), evaluation metrics, variants (BIRCH, OPTICS, FeatureAgglomeration), real-world pipelines, and limitations.*
