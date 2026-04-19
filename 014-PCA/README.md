# 📐 Principal Component Analysis (PCA) — Complete Job-Ready Notes

> **References:** Andrew Ng ML Specialization (Coursera) · Hands-On ML with Scikit-Learn (Aurélien Géron) · StatQuest with Josh Starmer (YouTube) · Scikit-Learn Documentation · Introduction to Statistical Learning with Python (ISLP)

---

## 📚 Table of Contents

1. [What is PCA? (Simple Statement)](#1-what-is-pca-simple-statement)
2. [Why PCA? Motivation](#2-why-pca-motivation)
3. [The Math Behind PCA](#3-the-math-behind-pca)
4. [Step-by-Step PCA Algorithm](#4-step-by-step-pca-algorithm)
5. [PCA via Gradient Descent (Manual)](#5-pca-via-gradient-descent-manual)
6. [PCA from Scratch in Python (NumPy)](#6-pca-from-scratch-in-python-numpy)
7. [PCA with Scikit-Learn](#7-pca-with-scikit-learn)
8. [Hyperparameters & Tuning](#8-hyperparameters--tuning)
9. [Choosing Number of Components](#9-choosing-number-of-components)
10. [Variants of PCA](#10-variants-of-pca)
11. [PCA in ML Pipelines](#11-pca-in-ml-pipelines)
12. [Common Interview Questions](#12-common-interview-questions)
13. [Resources](#13-resources)

---

## 1. What is PCA? (Simple Statement)

> **PCA finds new axes (called principal components) that capture the most variance in your data, letting you represent high-dimensional data in fewer dimensions without losing too much information.**

**Analogy (StatQuest style):**  
Imagine you have data spread in a cloud in 3D space. PCA finds the "best angle" to look at it from, so when you project it onto 2D, the spread (variance) is maximized. You keep the most informative view and discard the least informative direction.

- **Input:** High-dimensional data matrix `X` of shape `(n_samples, n_features)`
- **Output:** Lower-dimensional representation `Z` of shape `(n_samples, k_components)` where `k << n_features`
- **Key idea:** Linear transformation that rotates and scales data into a new coordinate system.

---

## 2. Why PCA? Motivation

| Problem | PCA Solution |
|---|---|
| Too many features (curse of dimensionality) | Reduce to `k` components |
| Multicollinearity between features | Components are orthogonal (uncorrelated) |
| Slow training due to high dimensionality | Fewer features → faster training |
| Hard to visualize data | Reduce to 2D/3D for plotting |
| Noisy features | Low-variance directions (noise) are dropped |

**When NOT to use PCA:**
- When interpretability matters (components are linear combinations, not original features)
- When data is already low-dimensional
- When relationships are non-linear (use Kernel PCA or t-SNE/UMAP instead)

---

## 3. The Math Behind PCA

### 3.1 Notation

```
X         : Data matrix, shape (n, p)  — n samples, p features
μ         : Mean vector, shape (p,)
X_centered: X - μ, zero-mean data
Σ         : Covariance matrix, shape (p, p)
V         : Matrix of eigenvectors (principal components), shape (p, k)
λ         : Eigenvalues (variance explained per component)
Z         : Projected data = X_centered @ V, shape (n, k)
```

### 3.2 Covariance Matrix

The **covariance matrix** captures how features vary together:

```
Σ = (1 / (n-1)) * X_centered^T @ X_centered     shape: (p, p)
```

- Diagonal entries: variance of each feature
- Off-diagonal entries: covariance between features
- Symmetric positive semi-definite matrix

### 3.3 Eigendecomposition

We decompose the covariance matrix:

```
Σ = V Λ V^T

where:
  V  = matrix of eigenvectors (columns = principal components)
  Λ  = diagonal matrix of eigenvalues λ₁ ≥ λ₂ ≥ ... ≥ λₚ
```

Each eigenvector `vᵢ` is a **direction** in feature space.  
Each eigenvalue `λᵢ` is the **variance** of the data along `vᵢ`.

### 3.4 Projection

To reduce from `p` dimensions to `k` dimensions:

```
Z = X_centered @ V_k

where V_k = first k eigenvectors (columns of V), shape (p, k)
Z = projected data, shape (n, k)
```

### 3.5 Explained Variance Ratio

How much information does each component retain?

```
Explained Variance Ratio for component i = λᵢ / Σλⱼ

Cumulative EVR for k components = Σᵢ₌₁ᵏ λᵢ / Σⱼ₌₁ᵖ λⱼ
```

### 3.6 SVD Connection

PCA is numerically computed via **Singular Value Decomposition (SVD)**, not eigendecomposition directly (more stable):

```
X_centered = U Σ V^T

where:
  U  : Left singular vectors, shape (n, n)
  Σ  : Diagonal of singular values σᵢ, shape (n, p)
  V^T: Right singular vectors (= principal components), shape (p, p)

Relationship: λᵢ = σᵢ² / (n-1)
Projection Z = U_k @ Σ_k  OR  X_centered @ V_k
```

> **Scikit-learn uses SVD internally** (via `scipy.linalg.svd` or randomized SVD for large data).

### 3.7 Reconstruction Error

```
X_reconstructed = Z @ V_k^T + μ

Reconstruction Error = ||X - X_reconstructed||²_F  (Frobenius norm)
```

The error equals the sum of discarded eigenvalues: `Σᵢ₌ₖ₊₁ᵖ λᵢ`

---

## 4. Step-by-Step PCA Algorithm

```
Step 1: Standardize / Center the data
        μ = mean(X, axis=0)
        X_c = X - μ
        (Optionally scale: X_c /= std(X, axis=0))

Step 2: Compute covariance matrix
        Σ = X_c^T @ X_c / (n-1)

Step 3: Eigendecomposition of Σ
        eigenvalues, eigenvectors = eig(Σ)

Step 4: Sort eigenvalues in descending order
        idx = argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

Step 5: Select top-k eigenvectors
        V_k = eigenvectors[:, :k]

Step 6: Project data
        Z = X_c @ V_k

Step 7: (Optional) Reconstruct for validation
        X_recon = Z @ V_k.T + μ
```

---

## 5. PCA via Gradient Descent (Manual)

> Andrew Ng (ML Specialization Week 4) frames PCA as an **optimization problem**: find the unit vector `w` that maximizes variance of projections.

### 5.1 Objective Function

```
Maximize: J(w) = (1/n) Σᵢ (xᵢ · w)²   subject to: ||w|| = 1

Equivalent to maximizing: w^T Σ w   subject to: w^T w = 1
```

### 5.2 Lagrangian

```
L(w, λ) = w^T Σ w - λ(w^T w - 1)

∂L/∂w = 2Σw - 2λw = 0
=> Σw = λw    ← This is the eigenvalue equation!
```

So the optimal `w` is the eigenvector of `Σ` corresponding to the largest eigenvalue.

### 5.3 Gradient Descent Implementation (for learning purposes)

```python
import numpy as np

def pca_gradient_descent(X_centered, k=2, lr=0.01, epochs=1000):
    """
    PCA via gradient ascent on variance objective.
    NOTE: This is educational — sklearn's SVD is far more efficient.
    """
    n, p = X_centered.shape
    
    # Initialize k random unit vectors
    W = np.random.randn(p, k)
    # Orthonormalize with QR decomposition
    W, _ = np.linalg.qr(W)
    
    for epoch in range(epochs):
        # Project data onto W
        Z = X_centered @ W           # (n, k)
        
        # Gradient of variance = (2/n) * X^T @ Z (projected back)
        grad = (2 / n) * X_centered.T @ Z  # (p, k)
        
        # Ascent step
        W = W + lr * grad
        
        # Re-orthonormalize (Gram-Schmidt via QR)
        W, _ = np.linalg.qr(W)
    
    return W

# Usage
np.random.seed(42)
X = np.random.randn(100, 5)
X_c = X - X.mean(axis=0)

W = pca_gradient_descent(X_c, k=2, lr=0.001, epochs=2000)
Z = X_c @ W  # Projected data (100, 2)
print("Components shape:", W.shape)   # (5, 2)
print("Projected shape:", Z.shape)    # (100, 2)
```

### 5.4 Why Gradient Descent for PCA?

- Andrew Ng uses this framing to connect PCA to **neural network autoencoders**
- Linear autoencoder (no activation, MSE loss, bottleneck) learns the same subspace as PCA
- Gradient descent scales better than full eigendecomposition for very large `p`

---

## 6. PCA from Scratch in Python (NumPy)

```python
import numpy as np
import matplotlib.pyplot as plt

class PCAFromScratch:
    """
    PCA implementation using NumPy eigendecomposition.
    Mirrors sklearn's PCA API.
    """
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components_ = None          # Principal axes (V_k), shape (k, p)
        self.explained_variance_ = None  # Eigenvalues
        self.explained_variance_ratio_ = None
        self.mean_ = None
    
    def fit(self, X):
        n, p = X.shape
        
        # Step 1: Center
        self.mean_ = X.mean(axis=0)
        X_c = X - self.mean_
        
        # Step 2: Covariance matrix
        cov = X_c.T @ X_c / (n - 1)   # shape (p, p)
        
        # Step 3: Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Step 4: Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Step 5: Store top-k
        self.components_ = eigenvectors[:, :self.n_components].T   # (k, p)
        self.explained_variance_ = eigenvalues[:self.n_components]
        self.explained_variance_ratio_ = (
            eigenvalues[:self.n_components] / eigenvalues.sum()
        )
        return self
    
    def transform(self, X):
        X_c = X - self.mean_
        return X_c @ self.components_.T   # (n, k)
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)
    
    def inverse_transform(self, Z):
        return Z @ self.components_ + self.mean_


# ──────────────────────────────────────────────
# Demo: Iris dataset
# ──────────────────────────────────────────────
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X, y = iris.data, iris.target

# Always scale before PCA!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCAFromScratch(n_components=2)
Z = pca.fit_transform(X_scaled)

print("Explained Variance Ratio:", pca.explained_variance_ratio_)
print("Cumulative EVR:", pca.explained_variance_ratio_.sum())
print("Projected shape:", Z.shape)

# ──────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────
colors = ['red', 'green', 'blue']
for i, name in enumerate(iris.target_names):
    mask = y == i
    plt.scatter(Z[mask, 0], Z[mask, 1], c=colors[i], label=name, alpha=0.7)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA from Scratch — Iris Dataset")
plt.legend()
plt.tight_layout()
plt.show()
```

---

## 7. PCA with Scikit-Learn

### 7.1 Basic Usage

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt

# Load data
digits = load_digits()
X, y = digits.data, digits.target   # (1797, 64)

# ALWAYS scale before PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Fit PCA ──
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print(f"Original shape     : {X_scaled.shape}")
print(f"Reduced shape      : {X_pca.shape}")
print(f"Explained variance : {pca.explained_variance_ratio_}")
print(f"Total variance kept: {pca.explained_variance_ratio_.sum():.3f}")
```

### 7.2 Key Attributes After Fitting

```python
pca.components_              # Principal axes, shape (n_components, n_features)
pca.explained_variance_      # Variance (eigenvalues) per component
pca.explained_variance_ratio_# Fraction of variance per component
pca.singular_values_         # Singular values (σᵢ)
pca.mean_                    # Per-feature mean (computed during fit)
pca.n_components_            # Actual number of components used
pca.noise_variance_          # Estimated noise variance (from remaining eigenvalues)
```

### 7.3 Choosing n_components by Variance Threshold

```python
# Keep components explaining 95% of variance
pca_95 = PCA(n_components=0.95, random_state=42)
X_95 = pca_95.fit_transform(X_scaled)
print(f"Components for 95% variance: {pca_95.n_components_}")
```

### 7.4 Explained Variance Plot (Scree Plot)

```python
# Fit full PCA first
pca_full = PCA(random_state=42)
pca_full.fit(X_scaled)

evr = pca_full.explained_variance_ratio_
cumulative_evr = np.cumsum(evr)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scree plot
axes[0].bar(range(1, 21), evr[:20], color='steelblue', alpha=0.8)
axes[0].set_xlabel("Principal Component")
axes[0].set_ylabel("Explained Variance Ratio")
axes[0].set_title("Scree Plot (Top 20 Components)")

# Cumulative EVR
axes[1].plot(range(1, len(cumulative_evr) + 1), cumulative_evr, 'b-o', markersize=3)
axes[1].axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
axes[1].axhline(y=0.99, color='g', linestyle='--', label='99% threshold')
axes[1].set_xlabel("Number of Components")
axes[1].set_ylabel("Cumulative Explained Variance")
axes[1].set_title("Cumulative Explained Variance Ratio")
axes[1].legend()

plt.tight_layout()
plt.show()

# Find component count for given thresholds
for threshold in [0.80, 0.90, 0.95, 0.99]:
    n = np.argmax(cumulative_evr >= threshold) + 1
    print(f"{threshold*100:.0f}% variance → {n} components")
```

### 7.5 Reconstruction & Error

```python
pca_k = PCA(n_components=20, random_state=42)
X_reduced = pca_k.fit_transform(X_scaled)
X_recon   = pca_k.inverse_transform(X_reduced)

# Reconstruction error
mse = np.mean((X_scaled - X_recon) ** 2)
print(f"Reconstruction MSE: {mse:.4f}")

# Visualize original vs reconstructed (digits)
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i in range(5):
    axes[0, i].imshow(X_scaled[i].reshape(8, 8), cmap='gray')
    axes[0, i].set_title("Original")
    axes[0, i].axis('off')
    axes[1, i].imshow(X_recon[i].reshape(8, 8), cmap='gray')
    axes[1, i].set_title(f"k=20")
    axes[1, i].axis('off')
plt.tight_layout()
plt.show()
```

### 7.6 PCA Biplot (Loadings + Scores)

```python
from sklearn.datasets import load_iris

iris = load_iris()
X_iris = StandardScaler().fit_transform(iris.data)

pca2 = PCA(n_components=2)
scores = pca2.fit_transform(X_iris)
loadings = pca2.components_.T   # shape (n_features, 2)

fig, ax = plt.subplots(figsize=(9, 7))

# Scatter scores
for i, name in enumerate(iris.target_names):
    mask = iris.target == i
    ax.scatter(scores[mask, 0], scores[mask, 1], label=name, alpha=0.6)

# Arrow loadings
scale = 3
for j, feat in enumerate(iris.feature_names):
    ax.annotate('', xy=(loadings[j, 0]*scale, loadings[j, 1]*scale),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.text(loadings[j, 0]*scale*1.1, loadings[j, 1]*scale*1.1,
            feat, color='red', fontsize=9)

ax.set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]:.1%} var)")
ax.set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]:.1%} var)")
ax.set_title("PCA Biplot — Iris")
ax.legend()
ax.axhline(0, color='gray', lw=0.5)
ax.axvline(0, color='gray', lw=0.5)
plt.tight_layout()
plt.show()
```

---

## 8. Hyperparameters & Tuning

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_components` | int, float, None, 'mle' | None | Number of components to keep. `int`: exact count. `float (0–1)`: variance ratio threshold. `'mle'`: auto-select via MLE. `None`: keep all. |
| `whiten` | bool | False | Divides components by their std so each has unit variance. Useful before clustering or distance-based methods. |
| `svd_solver` | str | 'auto' | SVD algorithm (see below) |
| `tol` | float | 0.0 | Tolerance for `arpack` solver |
| `iterated_power` | int or 'auto' | 'auto' | Power iterations for `randomized` solver (more = more accurate) |
| `n_oversamples` | int | 10 | Extra samples for `randomized` solver stability |
| `power_iteration_normalizer` | str | 'auto' | Normalization in randomized SVD |
| `random_state` | int | None | Seed for reproducibility (for `randomized` solver) |

### SVD Solvers

| Solver | When to Use | Speed |
|---|---|---|
| `'auto'` | Default, picks best based on shape | — |
| `'full'` | Small datasets, exact LAPACK SVD | Exact but slow for large |
| `'arpack'` | Sparse or large data with small k | Fast for sparse |
| `'randomized'` | Large dense data (n or p > 500), small k | Very fast, approximate |
| `'covariance_eigh'` | n >> p, compute cov matrix first | Fast when n >> p |

```python
# For large datasets (e.g., images, NLP)
pca_fast = PCA(n_components=50, svd_solver='randomized', random_state=42)

# For exact results on small data
pca_exact = PCA(n_components=10, svd_solver='full')

# Auto-select k via MLE (Minka 2001)
pca_mle = PCA(n_components='mle', svd_solver='full')
```

### `whiten=True` — When to Use

```python
# Whitening makes all components unit variance
# Use before: KMeans, SVM, LDA, ICA, neural networks
pca_white = PCA(n_components=20, whiten=True, random_state=42)
X_white = pca_white.fit_transform(X_scaled)

# Components are now decorrelated AND unit variance
print(np.cov(X_white.T).diagonal()[:5])   # All ~1.0
```

---

## 9. Choosing Number of Components

### Method 1: Explained Variance Threshold (Most Common)

```python
pca = PCA(n_components=0.95)   # Keep 95% of variance
```

### Method 2: Scree Plot — "Elbow Method"

Look for the "elbow" where variance drops sharply. Components after the elbow add little information.

### Method 3: MLE (Minka's Method)

```python
pca = PCA(n_components='mle', svd_solver='full')
pca.fit(X_scaled)
print(f"MLE suggests: {pca.n_components_} components")
```

### Method 4: Cross-Validation with Downstream Task

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

results = {}
for k in [2, 5, 10, 20, 30, 50]:
    pipe = Pipeline([
        ('pca', PCA(n_components=k, random_state=42)),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    score = cross_val_score(pipe, X_scaled, y, cv=5, scoring='accuracy').mean()
    results[k] = score
    print(f"k={k:3d} → CV Accuracy: {score:.4f}")

best_k = max(results, key=results.get)
print(f"\nBest k: {best_k} with accuracy {results[best_k]:.4f}")
```

### Method 5: Reconstruction Error vs k

```python
recon_errors = []
ks = range(1, 65)
for k in ks:
    pca_k = PCA(n_components=k)
    X_r = pca_k.fit_transform(X_scaled)
    X_rec = pca_k.inverse_transform(X_r)
    recon_errors.append(np.mean((X_scaled - X_rec)**2))

plt.plot(ks, recon_errors, 'b-o', markersize=3)
plt.xlabel("k (number of components)")
plt.ylabel("Reconstruction MSE")
plt.title("Reconstruction Error vs k")
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 10. Variants of PCA

### 10.1 Kernel PCA — Non-linear Dimensionality Reduction

```python
from sklearn.decomposition import KernelPCA

# For non-linearly separable data (e.g., Swiss roll, circles)
kpca_rbf = KernelPCA(n_components=2, kernel='rbf', gamma=0.04,
                      fit_inverse_transform=True, random_state=42)
X_kpca = kpca_rbf.fit_transform(X_scaled)

# Kernels: 'linear' (= standard PCA), 'poly', 'rbf', 'sigmoid', 'cosine'
# gamma: RBF kernel coefficient (higher = tighter fit)
# degree: for 'poly' kernel
# coef0: for 'poly' and 'sigmoid'
```

**Math:** Kernel PCA applies PCA in the feature space induced by a kernel function `k(xᵢ, xⱼ)` via the **kernel trick**, without explicitly computing the high-dimensional transformation.

### 10.2 Incremental PCA — Large Datasets (Out-of-Core)

```python
from sklearn.decomposition import IncrementalPCA

# Process data in mini-batches — good for data > RAM
ipca = IncrementalPCA(n_components=20, batch_size=200)

# Option 1: partial_fit batches
for batch in np.array_split(X_scaled, 10):
    ipca.partial_fit(batch)

X_ipca = ipca.transform(X_scaled)

# Option 2: fit directly (splits internally)
ipca2 = IncrementalPCA(n_components=20)
X_ipca2 = ipca2.fit_transform(X_scaled)
```

### 10.3 Randomized PCA — Fast Approximation for Large Data

```python
from sklearn.decomposition import PCA

# svd_solver='randomized' = Halko et al. 2009 algorithm
pca_rand = PCA(n_components=50, svd_solver='randomized',
               n_oversamples=10, iterated_power=4, random_state=42)
X_rand = pca_rand.fit_transform(X_scaled)
# Significantly faster than full SVD for large n or p
```

### 10.4 Sparse PCA

```python
from sklearn.decomposition import SparsePCA

# Components are sparse (few non-zero loadings) — more interpretable
spca = SparsePCA(n_components=5, alpha=1.0, random_state=42, n_jobs=-1)
X_spca = spca.fit_transform(X_scaled)
# alpha: regularization (higher = sparser components)
```

### 10.5 Truncated SVD (LSA) — Works on Sparse Matrices

```python
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import random as sparse_random

# Use for sparse matrices (TF-IDF, document-term matrix)
# Does NOT center data (centering destroys sparsity)
X_sparse = sparse_random(1000, 500, density=0.05, format='csr')

svd = TruncatedSVD(n_components=50, n_iter=7, random_state=42)
X_svd = svd.fit_transform(X_sparse)
```

### 10.6 Comparison Table

| Method | Data Type | Linear? | Scalable | Key Use Case |
|---|---|---|---|---|
| PCA | Dense | Yes | Medium | Standard dim reduction |
| Kernel PCA | Dense | No | Low | Non-linear manifolds |
| Incremental PCA | Dense | Yes | High | Out-of-core (large data) |
| Randomized PCA | Dense | Yes | High | Large n or p |
| Sparse PCA | Dense | Yes | Low | Interpretable components |
| TruncatedSVD | Sparse | Yes | High | Text/NLP (TF-IDF) |

---

## 11. PCA in ML Pipelines

### 11.1 Standard Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_digits

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Pipeline: Scale → PCA → Classifier
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca',    PCA(random_state=42)),
    ('clf',    SVC(kernel='rbf', random_state=42))
])

# Grid search over PCA + SVM hyperparameters together
param_grid = {
    'pca__n_components': [10, 20, 30, 40],
    'clf__C':            [0.1, 1, 10],
    'clf__gamma':        ['scale', 'auto']
}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy',
                    n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

print(f"Best params : {grid.best_params_}")
print(f"Test accuracy: {grid.score(X_test, y_test):.4f}")
```

### 11.2 PCA for Visualization

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 2D visualization
pca2 = PCA(n_components=2, random_state=42)
X_2d = pca2.fit_transform(StandardScaler().fit_transform(X))

plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='tab10',
                       alpha=0.6, s=15)
plt.colorbar(scatter, label='Digit Class')
plt.xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]:.1%})")
plt.ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]:.1%})")
plt.title("2D PCA of Digits Dataset")
plt.tight_layout()
plt.show()

# 3D visualization
pca3 = PCA(n_components=3, random_state=42)
X_3d = pca3.fit_transform(StandardScaler().fit_transform(X))

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2],
                c=y, cmap='tab10', alpha=0.5, s=10)
plt.colorbar(sc, ax=ax, label='Class')
ax.set_title("3D PCA of Digits")
plt.tight_layout()
plt.show()
```

### 11.3 PCA for Denoising

```python
# Compress then reconstruct = noise removal
from sklearn.datasets import load_digits
import numpy as np

X, y = load_digits(return_X_y=True)
scaler = StandardScaler()
X_s = scaler.fit_transform(X.astype(float))

# Add noise
rng = np.random.default_rng(42)
X_noisy = X_s + rng.normal(0, 0.5, X_s.shape)

# Denoise via PCA compression + reconstruction
pca_denoise = PCA(n_components=0.90, random_state=42)
X_compressed = pca_denoise.fit_transform(X_noisy)
X_denoised   = pca_denoise.inverse_transform(X_compressed)
X_denoised   = scaler.inverse_transform(X_denoised)

print(f"Components used: {pca_denoise.n_components_}")
print(f"Noise MSE  : {np.mean((X - X_noisy)**2):.4f}")
print(f"Denoised MSE: {np.mean((X - X_denoised)**2):.4f}")
```

---

## 12. Common Interview Questions

### Q1: What does PCA do?
> PCA finds orthogonal axes (principal components) of maximum variance in the data, and projects data onto the top-k such axes to reduce dimensionality while preserving as much variance as possible.

### Q2: Why must you standardize data before PCA?
> PCA is variance-based. Features with large scales (e.g., income in $100K vs age in years) would dominate the covariance matrix. Standardization puts all features on equal footing. **Always use `StandardScaler` before `PCA`.**

### Q3: What are principal components?
> They are the eigenvectors of the covariance matrix (or equivalently, the right singular vectors of the centered data matrix). They are orthogonal unit vectors pointing in the directions of maximum variance.

### Q4: What is the explained variance ratio?
> `EVR[i] = λᵢ / Σλⱼ` — the fraction of total variance captured by the i-th component. Cumulative EVR tells you how much total variance k components retain.

### Q5: PCA vs LDA — what's the difference?
| | PCA | LDA |
|---|---|---|
| Type | Unsupervised | Supervised |
| Objective | Max variance | Max class separation |
| Output | Principal components | Linear discriminants |
| Max components | min(n-1, p) | n_classes - 1 |

### Q6: When does PCA fail?
- Non-linear structure (use Kernel PCA, t-SNE, UMAP)
- When you need interpretability
- Very few samples (n < p, covariance matrix is rank-deficient)

### Q7: Is PCA affected by outliers?
> Yes. PCA is sensitive to outliers because variance is sensitive to extreme values. Consider **Robust PCA** or outlier removal before applying PCA.

### Q8: What's the difference between PCA and SVD?
> SVD is the numerical algorithm used to compute PCA. PCA is the statistical method; SVD is how sklearn implements it for numerical stability (avoids forming the potentially ill-conditioned covariance matrix explicitly).

### Q9: Can you apply PCA to categorical data?
> Not directly. Standard PCA requires continuous numerical data. Use **MCA (Multiple Correspondence Analysis)** for categorical, or encode carefully with **FAMD** (Factor Analysis of Mixed Data).

### Q10: What is whitening in PCA?
> Setting `whiten=True` divides each component by its standard deviation, resulting in unit variance per component. This is useful before ICA, neural networks, or algorithms sensitive to feature scale.

---

## 13. Resources

### 📖 Books

| Book | Chapter/Section | Topic |
|---|---|---|
| **Hands-On ML** (Géron, 3rd ed.) | Chapter 8 | Dimensionality Reduction, PCA, Kernel PCA, LLE |
| **ISLP** (James et al.) | Chapter 12 | Unsupervised Learning, PCA, Matrix Completion |
| **Pattern Recognition** (Bishop) | Chapter 12 | Probabilistic PCA, EM for PCA |

### 🎓 Andrew Ng — ML Specialization (Coursera)

- **Course 2: Advanced Learning Algorithms** — PCA for visualization
- **Course 3: Unsupervised Learning** — Week 2: PCA, reconstruction, choosing k
- Key lectures: "Reducing the number of features", "PCA algorithm", "Reconstruction from compressed representation"

### 📺 StatQuest with Josh Starmer (YouTube)

| Video | Link |
|---|---|
| StatQuest: PCA Step-by-Step | `youtube.com/watch?v=FgakZw6K1QQ` |
| StatQuest: PCA in Python | `youtube.com/watch?v=Lsue2gEM9D0` |
| SVD Visually Explained | `youtube.com/watch?v=vSFLr4OWZBM` |
| StatQuest: Covariance/Correlation | `youtube.com/watch?v=qtaqvPAeEJY` |

### 📚 Scikit-Learn Documentation

| Link | Description |
|---|---|
| `sklearn.decomposition.PCA` | Main PCA docs with parameters, examples |
| `sklearn.decomposition.KernelPCA` | Kernel PCA docs |
| `sklearn.decomposition.IncrementalPCA` | Out-of-core PCA |
| `sklearn.decomposition.TruncatedSVD` | Sparse matrix SVD |
| User Guide: Decomposing signals | `/stable/modules/decomposition.html` |

> Official URL: `https://scikit-learn.org/stable/modules/decomposition.html#pca`

### 🔗 Papers

| Paper | Note |
|---|---|
| Pearson (1901) | Original PCA paper |
| Halko et al. (2011) | Randomized SVD (used in sklearn) |
| Minka (2001) | Automatic dimensionality selection via MLE |
| Candès et al. (2011) | Robust PCA |

---

## 🗂️ Quick Reference Cheat Sheet

```python
# ── Minimal PCA workflow ──────────────────────────────────────
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Scale (MANDATORY)
X_scaled = StandardScaler().fit_transform(X)

# 2. Fit PCA
pca = PCA(n_components=0.95, random_state=42)  # keep 95% variance
X_reduced = pca.fit_transform(X_scaled)

# 3. Check
print(pca.n_components_)             # how many components selected
print(pca.explained_variance_ratio_) # variance per component
print(np.cumsum(pca.explained_variance_ratio_)[-1]) # total variance kept

# 4. Reconstruct
X_back = pca.inverse_transform(X_reduced)

# ── Key hyperparameters ──────────────────────────────────────
PCA(
    n_components  = 0.95,          # float: % variance | int: exact | 'mle': auto
    whiten        = False,         # True for clustering/ICA
    svd_solver    = 'auto',        # 'full'|'arpack'|'randomized'|'covariance_eigh'
    tol           = 0.0,           # arpack convergence
    iterated_power= 'auto',        # randomized: more = accurate
    n_oversamples = 10,            # randomized: extra samples
    random_state  = 42             # reproducibility
)

# ── Variants ─────────────────────────────────────────────────
from sklearn.decomposition import (
    KernelPCA,      # non-linear kernel trick
    IncrementalPCA, # mini-batch, out-of-core
    SparsePCA,      # sparse/interpretable components
    TruncatedSVD,   # sparse matrices (NLP/TF-IDF)
)
```

---

*Notes compiled for ML/DL job readiness. Cover PCA theory, mathematics (covariance, eigendecomposition, SVD), gradient descent formulation, NumPy implementation, full sklearn API, hyperparameter tuning, variants, and pipelines.*
