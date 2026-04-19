# 🔵 K-Nearest Neighbors (KNN) — Complete ML/DL Job-Ready Notes

> **"Tell me who your neighbors are, and I'll tell you who you are."**
> KNN is a **non-parametric, lazy learning** algorithm — it memorizes training data and makes predictions at query time.

---

## 📌 Table of Contents

1. [Intuition](#1-intuition)
2. [How KNN Works — Step by Step](#2-how-knn-works--step-by-step)
3. [Mathematics](#3-mathematics)
4. [Distance Metrics](#4-distance-metrics)
5. [KNN for Classification vs Regression](#5-knn-for-classification-vs-regression)
6. [Decision Boundary](#6-decision-boundary)
7. [The Bias-Variance Tradeoff in KNN](#7-the-bias-variance-tradeoff-in-knn)
8. [Hyperparameters](#8-hyperparameters)
9. [Curse of Dimensionality](#9-curse-of-dimensionality)
10. [Feature Scaling (Critical!)](#10-feature-scaling-critical)
11. [Python from Scratch](#11-python-from-scratch)
12. [Scikit-learn Implementation](#12-scikit-learn-implementation)
13. [Full Pipeline with Best Practices](#13-full-pipeline-with-best-practices)
14. [Finding Optimal K — The Elbow Method](#14-finding-optimal-k--the-elbow-method)
15. [Evaluation Metrics](#15-evaluation-metrics)
16. [Pros and Cons](#16-pros-and-cons)
17. [Interview Questions](#17-interview-questions)
18. [Resources](#18-resources)

---

## 1. Intuition

**Simple statement:** To classify a new point, look at the K closest points in training data and take a majority vote (classification) or average (regression).

**Real-world analogy:**
- You move to a new city. To guess the political leaning of your neighborhood, you look at your K nearest neighbors' political views and take the majority.
- Recommender systems: "Users similar to you also liked..."

**Key properties:**
- **Non-parametric** → No assumption about data distribution
- **Lazy learner** → No training phase; all computation happens at prediction time
- **Instance-based** → Stores entire training dataset

---

## 2. How KNN Works — Step by Step

```
Step 1: Choose the number of neighbors K
Step 2: Calculate distance from query point to ALL training points
Step 3: Sort distances in ascending order
Step 4: Pick the top K nearest neighbors
Step 5:
    - Classification → majority vote among K neighbors
    - Regression    → mean (or weighted mean) of K neighbors' values
Step 6: Return the prediction
```

**Visual example (K=3):**
```
Training data:          Query point: ★
  🔴 (2, 3)
  🔵 (1, 5)              ★ (3, 4)
  🔴 (4, 2)
  🔵 (5, 6)
  🔵 (3, 7)

Distances from ★:
  🔴 (2,3) → √2 ≈ 1.41
  🔴 (4,2) → √5 ≈ 2.24
  🔵 (1,5) → √5 ≈ 2.24
  ...

K=3 nearest: 🔴, 🔴, 🔵 → Majority = 🔴 → Prediction: RED
```

---

## 3. Mathematics

### 3.1 Euclidean Distance (Most Common)

$$d(x, x') = \sqrt{\sum_{i=1}^{n} (x_i - x'_i)^2}$$

For 2D: $d = \sqrt{(x_1 - x_1')^2 + (x_2 - x_2')^2}$

### 3.2 Classification Prediction

Given query point $x_q$, and K nearest neighbors $\{x^{(1)}, x^{(2)}, ..., x^{(K)}\}$ with labels $\{y^{(1)}, ..., y^{(K)}\}$:

$$\hat{y} = \text{mode}\{y^{(1)}, y^{(2)}, ..., y^{(K)}\}$$

Probability of class $c$:
$$P(y=c | x_q) = \frac{1}{K} \sum_{i=1}^{K} \mathbf{1}[y^{(i)} = c]$$

### 3.3 Regression Prediction

$$\hat{y} = \frac{1}{K} \sum_{i=1}^{K} y^{(i)}$$

### 3.4 Weighted KNN (Distance-Weighted)

Closer neighbors get more weight:

$$\hat{y} = \frac{\sum_{i=1}^{K} w_i \cdot y^{(i)}}{\sum_{i=1}^{K} w_i}, \quad w_i = \frac{1}{d(x_q, x^{(i)})^2}$$

### 3.5 Optimal K via Leave-One-Out Cross Validation

$$K^* = \arg\min_K \frac{1}{N} \sum_{i=1}^{N} \mathcal{L}(y^{(i)}, \hat{y}^{(i)}_{-i})$$

where $\hat{y}^{(i)}_{-i}$ = prediction for $x^{(i)}$ using all other training points.

---

## 4. Distance Metrics

| Metric | Formula | Use When |
|--------|---------|----------|
| **Euclidean** | $\sqrt{\sum(x_i - x_i')^2}$ | Continuous features, normalized data |
| **Manhattan** | $\sum \|x_i - x_i'\|$ | High-dim, grid-like data, robust to outliers |
| **Minkowski** | $(\sum \|x_i - x_i'\|^p)^{1/p}$ | Generalization (p=1: Manhattan, p=2: Euclidean) |
| **Chebyshev** | $\max_i \|x_i - x_i'\|$ | When max difference matters |
| **Hamming** | $\frac{1}{n}\sum \mathbf{1}[x_i \neq x_i']$ | Categorical/binary features |
| **Cosine** | $1 - \frac{x \cdot x'}{\|x\| \|x'\|}$ | Text, NLP, high-dim sparse data |

```python
from sklearn.neighbors import KNeighborsClassifier

# Change metric
knn = KNeighborsClassifier(n_neighbors=5, metric='manhattan')
knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)  # same as euclidean
knn = KNeighborsClassifier(n_neighbors=5, metric='cosine')
```

---

## 5. KNN for Classification vs Regression

### Classification
```python
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train, y_train)
clf.predict(X_test)           # returns class labels
clf.predict_proba(X_test)     # returns class probabilities
```

### Regression
```python
from sklearn.neighbors import KNeighborsRegressor

reg = KNeighborsRegressor(n_neighbors=5)
reg.fit(X_train, y_train)
reg.predict(X_test)           # returns continuous values
```

**Key difference:**
- Classification → **majority vote**
- Regression → **mean value**

---

## 6. Decision Boundary

- **K=1** → Very jagged, complex boundary (overfitting)
- **K=N (all data)** → Straight line / single class (underfitting)
- **K moderate (e.g., 5–15)** → Smooth, generalizable boundary

```
K=1 Decision Boundary:    K=15 Decision Boundary:
  Very jagged/complex         Smooth/simple
  High variance               Low variance
  Low bias                    High bias
```

**Rule of thumb:** Start with $K = \sqrt{N}$ where N = number of training samples.

---

## 7. The Bias-Variance Tradeoff in KNN

| K value | Bias | Variance | Model | Problem |
|---------|------|----------|-------|---------|
| K = 1 | Low | High | Complex | Overfitting |
| K = N | High | Low | Simple | Underfitting |
| K = optimal | Balanced | Balanced | Good | Generalization |

```
Error
  |
  |  Total Error
  |    \      /
  |     \    /  ← Optimal K
  |  Bias \ /
  |        X
  |       / \  Variance
  |______/___\________
  1                  N
           K →
```

**Note:** Unlike gradient descent-based models, KNN has no gradient descent. It has no "training" — all learning is deferred to prediction time.

---

## 8. Hyperparameters

### Complete Scikit-learn Hyperparameters

```python
KNeighborsClassifier(
    n_neighbors=5,        # K — number of neighbors (most important!)
    weights='uniform',    # 'uniform' or 'distance'
    algorithm='auto',     # 'ball_tree', 'kd_tree', 'brute', 'auto'
    leaf_size=30,         # For ball_tree/kd_tree — affects speed/memory
    p=2,                  # Power for Minkowski (1=Manhattan, 2=Euclidean)
    metric='minkowski',   # Distance metric
    metric_params=None,   # Extra params for custom metrics
    n_jobs=-1             # Parallel jobs (-1 = use all CPUs)
)
```

### Hyperparameter Details

#### `n_neighbors` (K) — THE most important

```python
# Rule of thumb: K = sqrt(n_samples)
import numpy as np
K_start = int(np.sqrt(len(X_train)))

# Try odd K to avoid ties in binary classification
K_values = [3, 5, 7, 9, 11, 13, 15, 21]
```

#### `weights`

| Value | Behavior | When to Use |
|-------|----------|-------------|
| `'uniform'` | All K neighbors equal vote | Default, balanced data |
| `'distance'` | Closer = more weight | Noisy data, unbalanced classes |

```python
# distance weighting: w_i = 1/d_i
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
```

#### `algorithm`

| Algorithm | Best For | Time Complexity |
|-----------|----------|-----------------|
| `'brute'` | Small datasets, high-dim | O(N·d) per query |
| `'kd_tree'` | Low-dim (<20 features) | O(log N) per query |
| `'ball_tree'` | High-dim, curved metrics | O(log N) per query |
| `'auto'` | Let sklearn decide | — |

```python
# For large datasets with many features → brute or ball_tree
knn = KNeighborsClassifier(algorithm='ball_tree', leaf_size=40)
```

#### `leaf_size`
- Used by `kd_tree` and `ball_tree`
- Smaller → slower build, faster query
- Larger → faster build, slower query
- Default 30 is usually fine

---

## 9. Curse of Dimensionality

**Simple statement:** In high dimensions, all points become equidistant, making "nearest neighbor" meaningless.

**Math intuition:**
- In 1D, to cover 10% of data → need 10% of range
- In 2D → need $\sqrt{0.1} \approx 31.6\%$ of range per dimension
- In D dimensions → need $0.1^{1/D}$ fraction per dimension → approaches 100% as D → ∞

```
Dimensions | % of range needed to capture 10% of data
     1     |   10%
    10     |   80%
   100     |   99.4%
  1000     |   99.94%
```

**Solutions:**
- Dimensionality reduction: PCA, t-SNE, UMAP
- Feature selection: Remove irrelevant features
- Use cosine similarity for NLP/high-dim data

```python
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('pca', PCA(n_components=50)),
    ('knn', KNeighborsClassifier(n_neighbors=5))
])
```

---

## 10. Feature Scaling (Critical!)

**Simple statement:** KNN is distance-based → features with larger scale dominate distance calculation → ALWAYS scale!

**Without scaling:**
```
Feature 1: salary    → [30000, 90000]  → dominates distance
Feature 2: age       → [25, 65]        → almost ignored
```

**With scaling:**
```
Both features → [0, 1] or mean=0, std=1 → equal contribution
```

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# StandardScaler: mean=0, std=1 (recommended for KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)   # ⚠️ Never fit on test data!

# MinMaxScaler: range [0,1]
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

## 11. Python from Scratch

### KNN Classifier from Scratch

```python
import numpy as np
from collections import Counter

class KNNClassifier:
    """
    K-Nearest Neighbors Classifier from scratch.
    """
    
    def __init__(self, k=5, metric='euclidean'):
        self.k = k
        self.metric = metric
    
    def fit(self, X, y):
        """
        Lazy learning — just store training data.
        Time complexity: O(1)
        Space complexity: O(N * d)
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self
    
    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def _manhattan_distance(self, x1, x2):
        return np.sum(np.abs(x1 - x2))
    
    def _compute_distance(self, x1, x2):
        if self.metric == 'euclidean':
            return self._euclidean_distance(x1, x2)
        elif self.metric == 'manhattan':
            return self._manhattan_distance(x1, x2)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def _predict_single(self, x):
        """Predict class for a single query point."""
        # Step 1: Compute distances to ALL training points
        distances = [
            self._compute_distance(x, x_train)
            for x_train in self.X_train
        ]
        
        # Step 2: Get indices of K nearest neighbors (sorted)
        k_nearest_indices = np.argsort(distances)[:self.k]
        
        # Step 3: Get labels of K nearest neighbors
        k_nearest_labels = self.y_train[k_nearest_indices]
        
        # Step 4: Majority vote
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    
    def predict(self, X):
        """Predict classes for all query points."""
        X = np.array(X)
        return np.array([self._predict_single(x) for x in X])
    
    def predict_proba(self, X):
        """Return class probabilities."""
        X = np.array(X)
        classes = np.unique(self.y_train)
        probas = []
        
        for x in X:
            distances = [self._compute_distance(x, x_t) for x_t in self.X_train]
            k_idx = np.argsort(distances)[:self.k]
            k_labels = self.y_train[k_idx]
            counts = Counter(k_labels)
            proba = [counts.get(c, 0) / self.k for c in classes]
            probas.append(proba)
        
        return np.array(probas)
    
    def score(self, X, y):
        """Accuracy score."""
        predictions = self.predict(X)
        return np.mean(predictions == np.array(y))


# ---- KNN Regressor from Scratch ----

class KNNRegressor:
    
    def __init__(self, k=5, weights='uniform'):
        self.k = k
        self.weights = weights
    
    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self
    
    def _predict_single(self, x):
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        k_idx = np.argsort(distances)[:self.k]
        k_distances = distances[k_idx]
        k_values = self.y_train[k_idx]
        
        if self.weights == 'uniform':
            return np.mean(k_values)
        elif self.weights == 'distance':
            # Avoid division by zero
            k_distances = np.where(k_distances == 0, 1e-10, k_distances)
            weights = 1.0 / k_distances
            return np.average(k_values, weights=weights)
    
    def predict(self, X):
        return np.array([self._predict_single(x) for x in np.array(X)])


# ---- Test it ----
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# From scratch
knn_scratch = KNNClassifier(k=5)
knn_scratch.fit(X_train_s, y_train)
acc = knn_scratch.score(X_test_s, y_test)
print(f"From Scratch Accuracy: {acc:.4f}")
```

---

## 12. Scikit-learn Implementation

### Classification

```python
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import load_iris, load_breast_cancer
import matplotlib.pyplot as plt
import seaborn as sns

# ── Load Data ──────────────────────────────────────────────────
data = load_breast_cancer()
X, y = data.data, data.target

# ── Split ──────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Scale (CRITICAL for KNN) ───────────────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ── Model ──────────────────────────────────────────────────────
knn = KNeighborsClassifier(
    n_neighbors=5,
    weights='uniform',
    metric='minkowski',
    p=2,
    n_jobs=-1
)
knn.fit(X_train, y_train)

# ── Evaluate ───────────────────────────────────────────────────
y_pred = knn.predict(X_test)
y_proba = knn.predict_proba(X_test)

print(classification_report(y_test, y_pred, target_names=data.target_names))
print(f"Test Accuracy: {knn.score(X_test, y_test):.4f}")

# Cross-validation
cv_scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=data.target_names,
            yticklabels=data.target_names)
plt.title("Confusion Matrix")
plt.show()
```

### Regression

```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score

housing = fetch_california_housing()
X, y = housing.data, housing.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

reg = KNeighborsRegressor(
    n_neighbors=5,
    weights='distance',   # distance-weighted is often better for regression
    metric='euclidean',
    n_jobs=-1
)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

print(f"R² Score:  {r2_score(y_test, y_pred):.4f}")
print(f"RMSE:      {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
```

### Finding Nearest Neighbors Directly

```python
from sklearn.neighbors import NearestNeighbors

nn = NearestNeighbors(n_neighbors=5, metric='euclidean')
nn.fit(X_train)

# Get distances and indices of nearest neighbors
distances, indices = nn.kneighbors(X_test[:5])
print("Distances:\n", distances)
print("Indices:\n", indices)
```

---

## 13. Full Pipeline with Best Practices

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.decomposition import PCA
import numpy as np

# ── Full Pipeline ─────────────────────────────────────────────
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),     # Keep 95% variance (optional)
    ('knn', KNeighborsClassifier())
])

# ── Hyperparameter Grid ───────────────────────────────────────
param_grid = {
    'knn__n_neighbors': [3, 5, 7, 9, 11, 15, 21],
    'knn__weights': ['uniform', 'distance'],
    'knn__metric': ['euclidean', 'manhattan'],
    'knn__p': [1, 2],
}

# ── Grid Search with Cross Validation ────────────────────────
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
    return_train_score=True
)

grid_search.fit(X_train, y_train)

print(f"Best Params:    {grid_search.best_params_}")
print(f"Best CV Score:  {grid_search.best_score_:.4f}")
print(f"Test Score:     {grid_search.score(X_test, y_test):.4f}")

best_model = grid_search.best_estimator_

# ── Randomized Search (faster for large grids) ───────────────
from scipy.stats import randint

param_dist = {
    'knn__n_neighbors': randint(1, 30),
    'knn__weights': ['uniform', 'distance'],
    'knn__metric': ['euclidean', 'manhattan', 'chebyshev'],
}

random_search = RandomizedSearchCV(
    pipeline, param_dist,
    n_iter=50, cv=5, scoring='accuracy',
    n_jobs=-1, random_state=42
)
random_search.fit(X_train, y_train)
print(f"Best Random Search Score: {random_search.best_score_:.4f}")
```

---

## 14. Finding Optimal K — The Elbow Method

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

train_errors = []
val_errors = []
k_range = range(1, 31)

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # Training error
    knn.fit(X_train, y_train)
    train_errors.append(1 - knn.score(X_train, y_train))
    
    # Validation error (cross-val)
    cv_score = cross_val_score(knn, X_train, y_train, cv=5).mean()
    val_errors.append(1 - cv_score)

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(k_range, train_errors, 'b-o', label='Training Error', linewidth=2)
ax.plot(k_range, val_errors, 'r-o', label='Validation Error', linewidth=2)
ax.axvline(x=np.argmin(val_errors) + 1, color='green', linestyle='--',
           label=f'Optimal K = {np.argmin(val_errors)+1}')
ax.set_xlabel('K (Number of Neighbors)', fontsize=13)
ax.set_ylabel('Error Rate', fontsize=13)
ax.set_title('KNN: Training vs Validation Error', fontsize=15)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

optimal_k = np.argmin(val_errors) + 1
print(f"Optimal K: {optimal_k}")
```

---

## 15. Evaluation Metrics

### Classification Metrics

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)

y_pred = knn.predict(X_test)
y_proba = knn.predict_proba(X_test)[:, 1]  # For binary

print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
print(f"ROC AUC:   {roc_auc_score(y_test, y_proba):.4f}")
print("\n", classification_report(y_test, y_pred))

# ROC Curve
from sklearn.metrics import RocCurveDisplay
RocCurveDisplay.from_estimator(knn, X_test, y_test)
plt.title("ROC Curve - KNN")
plt.show()
```

### Regression Metrics

```python
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    root_mean_squared_error, r2_score
)

y_pred = reg.predict(X_test)

print(f"MAE:   {mean_absolute_error(y_test, y_pred):.4f}")
print(f"MSE:   {mean_squared_error(y_test, y_pred):.4f}")
print(f"RMSE:  {root_mean_squared_error(y_test, y_pred):.4f}")
print(f"R²:    {r2_score(y_test, y_pred):.4f}")
```

---

## 16. Pros and Cons

### ✅ Advantages

| Advantage | Explanation |
|-----------|-------------|
| **Simple & Intuitive** | Easy to understand and explain |
| **No Training Phase** | Fast to "train" (just store data) |
| **Non-parametric** | Works for any decision boundary shape |
| **Multi-class Natural** | Handles multi-class without modification |
| **No Assumptions** | Doesn't assume linear/Gaussian distribution |
| **Naturally Adapts** | Adding new training data is trivial |

### ❌ Disadvantages

| Disadvantage | Explanation |
|--------------|-------------|
| **Slow Prediction** | O(N·d) per query — expensive on large datasets |
| **High Memory** | Must store entire training set |
| **Sensitive to Scale** | MUST normalize features |
| **Curse of Dimensionality** | Breaks down in high dimensions |
| **Sensitive to Irrelevant Features** | Noise features degrade distances |
| **Imbalanced Classes** | Majority class dominates |
| **Optimal K Selection** | Requires cross-validation tuning |

### When to Use KNN

✅ **Use KNN when:**
- Small to medium dataset
- Low-dimensional data
- Non-linear decision boundary needed
- Baseline/quick prototype
- Recommendation systems
- Anomaly detection

❌ **Avoid KNN when:**
- Large dataset (millions of samples)
- Very high-dimensional data (100+ features)
- Real-time predictions needed
- Memory is constrained

---

## 17. Interview Questions

### Conceptual

**Q1: Why must we scale features for KNN?**
> KNN uses distance metrics. Features with larger ranges (e.g., salary: 0–100,000) will dominate the distance over features with smaller ranges (e.g., age: 0–100). Scaling ensures equal contribution.

**Q2: What happens when K=1 vs K=N?**
> K=1 → Overfitting: the model memorizes training data, high variance, low bias. K=N → Underfitting: predicts the same class for all points (majority class), high bias, low variance.

**Q3: Why use odd K for binary classification?**
> To avoid ties. With odd K, you always get a clear majority between 2 classes.

**Q4: Is KNN a lazy or eager learner? Why?**
> Lazy learner. It doesn't build an explicit model during training — it just stores data. All computation is deferred to prediction time.

**Q5: How does KNN handle multi-class classification?**
> Naturally — it just takes majority vote among K neighbors, which can be any of the multiple classes.

**Q6: What is the time complexity of KNN prediction?**
> Brute force: O(N × d) per query, where N = training samples and d = features. With KD-tree: O(log N × d) for low dimensions.

**Q7: How would you handle imbalanced classes in KNN?**
> Options: (1) Use `weights='distance'`, (2) Oversample minority class (SMOTE), (3) Use distance-weighted voting, (4) Evaluate with F1/AUC not just accuracy.

**Q8: What is the curse of dimensionality and how does it affect KNN?**
> In high dimensions, all points become equidistant. The concept of "nearest neighbor" breaks down because no point is meaningfully closer than others. Fix: PCA, feature selection, or switch to cosine similarity.

**Q9: KNN vs K-Means — what's the difference?**
> KNN is a supervised algorithm for classification/regression. K-Means is an unsupervised clustering algorithm. Despite similar names, they are completely different.

**Q10: How do you choose the optimal K?**
> Use cross-validation (e.g., 5-fold CV) over a range of K values and pick the K with lowest validation error. Also try K = √N as starting point.

---

## 18. Resources

### 📘 Andrew Ng — ML Specialization (Coursera)
- **Course 1, Week 3**: Classification algorithms overview
- **Course 2**: Bias-Variance tradeoff, model selection
- **Course 3**: Decision boundaries, evaluation metrics
- 🔗 https://www.coursera.org/specializations/machine-learning-introduction

### 📗 Hands-On Machine Learning (Aurélien Géron)
- **Chapter 3**: Classification — KNN used as first classifier
- **Chapter 2**: End-to-end ML Pipeline with scaling
- **Chapter 5**: Covers SVM but bias-variance concepts apply directly
- Key exercise: MNIST with KNN for 97%+ accuracy
- 🔗 https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/

### 🎬 StatQuest with Josh Starmer (YouTube)
- **"K-nearest neighbors, Clearly Explained"** → Core intuition
- **"Machine Learning Fundamentals: Bias and Variance"** → K tradeoff
- **"Cross-validation"** → How to choose K properly
- 🔗 https://www.youtube.com/@statquest

### 📙 Introduction to Statistical Learning (ISLP — Python Edition)
- **Chapter 2.2.3**: K-Nearest Neighbors
- **Chapter 5**: Resampling Methods (Cross-Validation for K selection)
- **Chapter 2.2**: Bias-Variance tradeoff with KNN examples
- Free PDF: 🔗 https://www.statlearning.com/

### 📜 Scikit-learn Documentation
- KNeighborsClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
- KNeighborsRegressor: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
- NearestNeighbors: https://scikit-learn.org/stable/modules/neighbors.html
- User Guide: https://scikit-learn.org/stable/modules/neighbors.html

### 📺 Additional YouTube Resources
- **Sentdex** — Practical KNN in Python
- **codebasics** — KNN Tutorial with real dataset
- **3Blue1Brown** — Distance in higher dimensions (for Curse of Dimensionality)

---

## 🎯 Quick Cheat Sheet

```
Algorithm:     Non-parametric, lazy, instance-based
Training:      O(1) — just store data
Prediction:    O(N × d) brute / O(log N × d) kd-tree
Memory:        O(N × d)

MUST DO:       Scale features (StandardScaler)
TUNE:          K via cross-validation
WATCH OUT:     High dimensions, large datasets, imbalanced classes

For K:         Start with √N, prefer odd K, tune via CV
For weights:   'distance' if noisy or imbalanced
For metric:    euclidean default, cosine for text/NLP
For algorithm: auto (sklearn picks best for your data)

Bias-Variance: K↑ → bias↑, variance↓ (simpler model)
               K↓ → bias↓, variance↑ (complex model)
```

---

*Notes compiled from: Andrew Ng ML Specialization · Hands-On ML (Géron) · StatQuest · ISLP (James et al.) · Scikit-learn Docs*
