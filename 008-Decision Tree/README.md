# 🌳 Decision Trees — Complete ML/DL Job-Ready Notes

> **"Ask the right questions in the right order, and the answer reveals itself."**
> A Decision Tree is a **non-parametric, supervised, eager learning** algorithm that learns a hierarchy of if-else rules from data to make predictions.

---

## 📌 Table of Contents

1. [Intuition](#1-intuition)
2. [Anatomy of a Decision Tree](#2-anatomy-of-a-decision-tree)
3. [How a Decision Tree Learns — The Algorithm](#3-how-a-decision-tree-learns--the-algorithm)
4. [Splitting Criteria — The Mathematics](#4-splitting-criteria--the-mathematics)
5. [Gini Impurity — Deep Dive](#5-gini-impurity--deep-dive)
6. [Entropy & Information Gain — Deep Dive](#6-entropy--information-gain--deep-dive)
7. [Variance Reduction (Regression Trees)](#7-variance-reduction-regression-trees)
8. [CART Algorithm](#8-cart-algorithm)
9. [Tree Pruning](#9-tree-pruning)
10. [Bias-Variance Tradeoff in Decision Trees](#10-bias-variance-tradeoff-in-decision-trees)
11. [Hyperparameters — Complete Guide](#11-hyperparameters--complete-guide)
12. [Feature Importance](#12-feature-importance)
13. [Python from Scratch](#13-python-from-scratch)
14. [Scikit-learn Implementation](#14-scikit-learn-implementation)
15. [Full Pipeline with Best Practices](#15-full-pipeline-with-best-practices)
16. [Visualizing the Tree](#16-visualizing-the-tree)
17. [Evaluation Metrics](#17-evaluation-metrics)
18. [Pros and Cons](#18-pros-and-cons)
19. [Interview Questions](#19-interview-questions)
20. [Resources](#20-resources)

---

## 1. Intuition

**Simple statement:** Split the data recursively by asking yes/no questions about features. At each node, choose the question that best separates the classes (or reduces prediction error). Stop when nodes are pure or stopping criteria are met.

**Real-world analogy — 20 Questions game:**
```
Is it an animal?
  YES → Does it have 4 legs?
          YES → Is it bigger than a dog?
                  YES → Lion 🦁
                  NO  → Cat 🐱
          NO  → Does it have feathers?
                  YES → Bird 🐦
  NO  → Is it man-made?
          YES → Is it electronic?
                  YES → Computer 💻
```

**Medical diagnosis analogy:**
```
Is age > 50?
  YES → Is blood_pressure > 140?
          YES → High Risk ❗
          NO  → Medium Risk ⚠️
  NO  → Is cholesterol > 200?
          YES → Medium Risk ⚠️
          NO  → Low Risk ✅
```

**Key properties:**
- **Non-parametric** → No assumption about data distribution
- **White-box model** → Fully interpretable, rules visible
- **Handles mixed data** → Numerical and categorical features
- **No scaling needed** → Distance not used, only comparisons
- **Greedy algorithm** → Each split is locally optimal (not globally)

---

## 2. Anatomy of a Decision Tree

```
                    ┌─────────────────────┐
                    │   ROOT NODE         │  ← First split (best feature overall)
                    │  petal_length ≤ 2.5 │
                    └──────────┬──────────┘
                               │
               ┌───────────────┴───────────────┐
             YES                               NO
               │                               │
    ┌──────────▼──────────┐       ┌────────────▼────────────┐
    │   LEAF NODE         │       │   INTERNAL NODE         │
    │   class: setosa     │       │   petal_width ≤ 1.8     │
    │   (pure — gini=0)   │       └────────────┬────────────┘
    └─────────────────────┘                    │
                                ┌──────────────┴──────────────┐
                              YES                             NO
                                │                             │
                   ┌────────────▼────────┐      ┌────────────▼────────┐
                   │   LEAF NODE         │      │   LEAF NODE         │
                   │ class: versicolor   │      │  class: virginica   │
                   └─────────────────────┘      └─────────────────────┘
```

### Node Terminology

| Term | Definition |
|------|-----------|
| **Root Node** | Top node — first split, represents entire dataset |
| **Internal Node** | A split — tests one feature with a threshold |
| **Leaf Node** | Terminal node — no further splits, holds prediction |
| **Branch / Edge** | Connection between nodes (YES/NO path) |
| **Depth** | Length of longest path from root to leaf |
| **Subtree** | Any node and all its descendants |
| **Pure Node** | Leaf where all samples belong to one class (impurity = 0) |

### What Each Node Stores

```python
# Each node in sklearn's tree stores:
node.feature          # which feature is used for splitting
node.threshold        # split threshold value
node.impurity         # Gini or entropy at this node
node.n_node_samples   # number of training samples at this node
node.value            # class distribution / regression value
```

---

## 3. How a Decision Tree Learns — The Algorithm

### CART (Classification and Regression Trees) — Used by Sklearn

```
Algorithm: Recursive Binary Splitting

BUILD_TREE(dataset D, depth d):
  1. If stopping condition met:
       → Return LEAF NODE with prediction = majority class (or mean)

  2. For each feature j and each threshold t:
       → Compute split quality: Gini / Entropy / Variance
       → Keep track of best (j*, t*) = lowest impurity

  3. Split D into D_left (X_j ≤ t*) and D_right (X_j > t*)

  4. Recursively call:
       left_child  = BUILD_TREE(D_left,  d+1)
       right_child = BUILD_TREE(D_right, d+1)

  5. Return NODE with feature j*, threshold t*, left/right children

Stopping Conditions:
  - max_depth reached
  - min_samples_split not met
  - min_samples_leaf not met
  - No improvement in impurity
  - All samples in node belong to one class (pure)
```

### Prediction

```
PREDICT(x, node):
  If node is LEAF:
    return node.prediction

  If x[node.feature] ≤ node.threshold:
    return PREDICT(x, node.left_child)
  Else:
    return PREDICT(x, node.right_child)
```

**Time Complexity:**
- Training: O(N · d · log N) — N samples, d features, log N splits
- Prediction: O(depth) — just follow one path from root to leaf

---


## 4. Splitting Criteria — The Mathematics

### Core Idea

At each node, we choose the split $(j, t)$ that **minimizes the weighted impurity of the two child nodes**:

$$
\text{Cost}(j, t) =
\frac{N_{\text{left}}}{N} , \text{Impurity}(\text{left}) +
\frac{N_{\text{right}}}{N} , \text{Impurity}(\text{right})
$$

We search over all features $j$ and all possible thresholds $t$ to find:


---



### Three Main Impurity Measures

| Criterion | Formula | Used For |
|-----------|---------|---------|
| **Gini Impurity** | $1 - \sum_k p_k^2$ | Classification (CART default) |
| **Entropy** | $-\sum_k p_k \log_2 p_k$ | Classification (ID3, C4.5) |
| **Variance** | $\frac{1}{N}\sum(y_i - \bar{y})^2$ | Regression |

---

## 5. Gini Impurity — Deep Dive

### Formula

$$\text{Gini}(t) = 1 - \sum_{k=1}^{K} p_k^2$$

where $p_k = \frac{\text{count of class } k \text{ at node } t}{\text{total samples at node } t}$

### Interpretation

- **Gini = 0** → Pure node (all samples same class) — best possible
- **Gini = 0.5** → Maximum impurity for binary classification (50/50 split)
- **Gini = 1 - 1/K** → Maximum for K classes (uniform distribution)

### Worked Example (Binary Classification)

```
Node has 100 samples: 60 class A, 40 class B

p_A = 60/100 = 0.6
p_B = 40/100 = 0.4

Gini = 1 - (0.6² + 0.4²)
     = 1 - (0.36 + 0.16)
     = 1 - 0.52
     = 0.48
```

### Gini for a Split

```
Before split:  100 samples → [60 A, 40 B] → Gini = 0.48

Split on feature "age ≤ 30":
  Left (age ≤ 30):  40 samples → [35 A, 5 B]  → Gini_L = 1-(35/40)²-(5/40)² = 0.219
  Right (age > 30): 60 samples → [25 A, 35 B] → Gini_R = 1-(25/60)²-(35/60)² = 0.486

Weighted Gini = (40/100)×0.219 + (60/100)×0.486
              = 0.0875 + 0.292
              = 0.379

Gini Gain = 0.48 - 0.379 = 0.101  ← improvement!
```

### Gini Impurity in Python

```python
import numpy as np

def gini_impurity(y):
    """Compute Gini impurity for a node."""
    n = len(y)
    if n == 0:
        return 0
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / n
    return 1 - np.sum(probs ** 2)

def gini_split(y_left, y_right):
    """Compute weighted Gini after a split."""
    n = len(y_left) + len(y_right)
    return (len(y_left)/n) * gini_impurity(y_left) + \
           (len(y_right)/n) * gini_impurity(y_right)

# Example
y_left  = [0, 0, 0, 0, 1]           # 4 class 0, 1 class 1
y_right = [0, 1, 1, 1, 1, 1]        # 1 class 0, 5 class 1
print(f"Gini split: {gini_split(y_left, y_right):.4f}")
```

---

## 6. Entropy & Information Gain — Deep Dive

### Entropy Formula

$$H(t) = -\sum_{k=1}^{K} p_k \log_2(p_k)$$

Convention: $0 \cdot \log_2(0) = 0$

### Interpretation

- **H = 0** → Pure node — no uncertainty
- **H = 1** → Maximum uncertainty for binary (50/50 split)
- **H = log₂(K)** → Maximum for K classes (uniform)

### Information Gain

$$\text{IG}(t, j, t_h) = H(\text{parent}) - \left[ \frac{N_L}{N} H(L) + \frac{N_R}{N} H(R) \right]$$

We **maximize** Information Gain (equivalently, minimize weighted child entropy).

### Worked Example

```
Parent: 100 samples → [50 spam, 50 ham] → H = -0.5·log₂(0.5) - 0.5·log₂(0.5) = 1.0 bit

Split on "contains FREE":
  Left (FREE=yes): 30 samples → [28 spam, 2 ham]
    H_L = -(28/30)log₂(28/30) - (2/30)log₂(2/30) = 0.317 bits
  Right (FREE=no): 70 samples → [22 spam, 48 ham]
    H_R = -(22/70)log₂(22/70) - (48/70)log₂(48/70) = 0.902 bits

Weighted H = (30/100)×0.317 + (70/100)×0.902 = 0.095 + 0.631 = 0.726

Information Gain = 1.0 - 0.726 = 0.274 bits  ← high gain → good split!
```

### Entropy vs Gini

```python
import numpy as np
import matplotlib.pyplot as plt

p = np.linspace(0.001, 0.999, 1000)

gini    = 2 * p * (1 - p)          # Gini for binary case (simplified)
entropy = -p*np.log2(p) - (1-p)*np.log2(1-p)

plt.figure(figsize=(8, 4))
plt.plot(p, gini,    label='Gini Impurity × 2', linewidth=2)
plt.plot(p, entropy, label='Entropy (bits)',     linewidth=2)
plt.xlabel('p (probability of class 1)')
plt.ylabel('Impurity')
plt.title('Gini vs Entropy')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
# Both are very similar in practice — Gini is faster to compute (no log)
```

| Property | Gini | Entropy |
|----------|------|---------|
| Range | [0, 0.5] binary | [0, 1] binary |
| Computation | Faster (no log) | Slower |
| Behavior | Tends to isolate most frequent class | More balanced splits |
| Default in sklearn | ✅ Yes | Optional (`criterion='entropy'`) |
| Practical difference | Very small | Very small |

---

## 7. Variance Reduction (Regression Trees)

### For Regression: minimize variance in child nodes

$$\text{Variance}(t) = \frac{1}{N_t} \sum_{i \in t} (y_i - \bar{y}_t)^2$$

$$\text{Variance Reduction} = \text{Var}(\text{parent}) - \left[\frac{N_L}{N}\text{Var}(L) + \frac{N_R}{N}\text{Var}(R)\right]$$

**Prediction at leaf** = mean of all training samples that fall into that leaf:

$$\hat{y} = \frac{1}{|L|} \sum_{i \in L} y_i$$

```python
def variance_reduction(y_parent, y_left, y_right):
    n = len(y_parent)
    var_parent = np.var(y_parent)
    var_left   = np.var(y_left)
    var_right  = np.var(y_right)
    weighted   = (len(y_left)/n)*var_left + (len(y_right)/n)*var_right
    return var_parent - weighted

# sklearn uses criterion='squared_error' for regression trees
from sklearn.tree import DecisionTreeRegressor
reg = DecisionTreeRegressor(criterion='squared_error')
```

**Other regression criteria:**
- `'squared_error'` → minimizes MSE (default)
- `'friedman_mse'` → MSE with Friedman improvement score
- `'absolute_error'` → minimizes MAE (more robust to outliers)
- `'poisson'` → for count data

---

## 8. CART Algorithm

**CART (Classification and Regression Trees)** — Breiman et al. (1984) — is the algorithm scikit-learn implements.

### Key properties of CART:
1. **Binary splits only** — always splits into exactly 2 nodes
2. **Works for both** classification and regression
3. **Uses Gini** (classification) or **MSE** (regression) by default
4. **Greedy** — locally optimal split at each node
5. **Cost-complexity pruning** — post-pruning via `ccp_alpha`

### CART vs Other Tree Algorithms

| Algorithm | Splits | Criteria | Handles Missing | Notes |
|-----------|--------|----------|-----------------|-------|
| **CART** (sklearn) | Binary | Gini / MSE | No | Most common |
| **ID3** | Multi-way | Entropy / IG | No | Categorical only |
| **C4.5** | Multi-way | Gain Ratio | Yes | Handles continuous |
| **C5.0** | Multi-way | Gain Ratio | Yes | Commercial, faster |
| **CHAID** | Multi-way | Chi-square | Yes | Statistical splits |

---

## 9. Tree Pruning

### Why Prune?

A fully grown tree memorizes training data → **overfitting**. Pruning reduces complexity.

### Pre-pruning (Early Stopping)

Stop tree growth early using stopping criteria:

```python
DecisionTreeClassifier(
    max_depth=5,           # Stop at depth 5
    min_samples_split=20,  # Don't split if < 20 samples
    min_samples_leaf=10,   # Each leaf must have ≥ 10 samples
    max_features=None,     # Consider all features per split
    max_leaf_nodes=50,     # Limit total number of leaves
)
```

### Post-pruning: Cost-Complexity Pruning (ccp_alpha)

Scikit-learn implements **Minimal Cost-Complexity Pruning** via `ccp_alpha` (α):

$$R_\alpha(T) = R(T) + \alpha \cdot |T|$$

where $R(T)$ = misclassification rate, $|T|$ = number of leaves

- **Higher α** → more pruning → simpler tree
- **Lower α** → less pruning → complex tree

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

# Step 1: Get effective alphas
clf = DecisionTreeClassifier(random_state=42)
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

# Step 2: Train a tree for each alpha
clfs = []
for alpha in ccp_alphas:
    clf = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
    clf.fit(X_train, y_train)
    clfs.append(clf)

# Step 3: Find optimal alpha via cross-validation
cv_scores = [cross_val_score(clf, X_train, y_train, cv=5).mean()
             for clf in clfs]
optimal_alpha = ccp_alphas[np.argmax(cv_scores)]
print(f"Optimal ccp_alpha: {optimal_alpha:.6f}")

# Step 4: Final model
final_clf = DecisionTreeClassifier(ccp_alpha=optimal_alpha, random_state=42)
final_clf.fit(X_train, y_train)
```

```python
# Visualize pruning path
import matplotlib.pyplot as plt

train_scores = [clf.score(X_train, y_train) for clf in clfs]
test_scores  = [clf.score(X_test,  y_test)  for clf in clfs]
n_leaves     = [clf.get_n_leaves() for clf in clfs]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(ccp_alphas, train_scores, 'b-o', label='Train', linewidth=2)
ax1.plot(ccp_alphas, test_scores,  'r-o', label='Test',  linewidth=2)
ax1.set_xlabel('ccp_alpha')
ax1.set_ylabel('Accuracy')
ax1.set_title('Accuracy vs Alpha (Pruning)')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(ccp_alphas, n_leaves, 'g-o', linewidth=2)
ax2.set_xlabel('ccp_alpha')
ax2.set_ylabel('Number of Leaves')
ax2.set_title('Tree Size vs Alpha')
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## 10. Bias-Variance Tradeoff in Decision Trees

| Tree Depth | Bias | Variance | Issue |
|------------|------|----------|-------|
| Very deep (max) | Low | High | Overfitting — memorizes training data |
| Very shallow (depth=1) | High | Low | Underfitting — too simple ("stump") |
| Optimal depth | Balanced | Balanced | Generalizes well |

```
Error
  │
  │  ╲ Bias²
  │   ╲         Total Error
  │    ╲       ╱╲
  │     ╲     ╱  ╲
  │      ╲   ╱    ╲ Variance
  │       ╲ ╱      ╲
  │________╳_________
  Shallow        Deep
  (Simple)   (Complex)
      Depth →
```

**Key insight:** An unpruned Decision Tree has **zero training error** (it can perfectly memorize). But test error is high. Regularization via hyperparameters or pruning is critical.

**Note on Gradient Descent:** Decision Trees do NOT use gradient descent. Splits are found by **exhaustive greedy search** over all features and thresholds. The optimization is combinatorial, not gradient-based. (Gradient Boosted Trees, however, DO use gradients — but that's an ensemble method.)

---

## 11. Hyperparameters — Complete Guide

```python
from sklearn.tree import DecisionTreeClassifier

DecisionTreeClassifier(
    criterion='gini',          # Split quality: 'gini', 'entropy', 'log_loss'
    splitter='best',           # 'best' (optimal split) or 'random' (random split)
    max_depth=None,            # Maximum depth of tree (None = unlimited)
    min_samples_split=2,       # Min samples required to split an internal node
    min_samples_leaf=1,        # Min samples required at each leaf node
    min_weight_fraction_leaf=0.0, # Min weighted fraction at leaf
    max_features=None,         # Features to consider per split: None, int, float, 'sqrt', 'log2'
    random_state=None,         # Seed for reproducibility
    max_leaf_nodes=None,       # Limit total number of leaf nodes
    min_impurity_decrease=0.0, # Split only if impurity decreases by this amount
    class_weight=None,         # 'balanced', dict, or None
    ccp_alpha=0.0,             # Post-pruning complexity parameter
)
```

### Hyperparameter Details

#### `criterion` — Split Quality Measure

| Value | Formula | Notes |
|-------|---------|-------|
| `'gini'` | $1 - \sum p_k^2$ | Default, faster |
| `'entropy'` | $-\sum p_k \log_2 p_k$ | Slightly slower, similar results |
| `'log_loss'` | Same as entropy | Alias |

```python
# In practice, gini and entropy give very similar trees
clf_gini    = DecisionTreeClassifier(criterion='gini')
clf_entropy = DecisionTreeClassifier(criterion='entropy')
```

#### `max_depth` — Most Important Regularization Param

```python
# Controls tree depth → controls model complexity
# None → fully grown → overfitting
# Small (3-10) → regularized → better generalization

# Tune via GridSearchCV
from sklearn.model_selection import GridSearchCV
param_grid = {'max_depth': [None, 3, 5, 7, 10, 15, 20]}
```

#### `min_samples_split`

```python
# Min samples needed to attempt a split
# Default 2 → split on even 2 samples → overfitting
# Higher → fewer splits → simpler tree → regularization

# As int: absolute count
min_samples_split=20    # need ≥ 20 samples to split

# As float: fraction of total training samples
min_samples_split=0.05  # need ≥ 5% of total samples
```

#### `min_samples_leaf`

```python
# Min samples in every LEAF node — stronger regularization than min_samples_split
# Guarantees leaves are not tiny → smooths the predictions
# Especially important for REGRESSION trees

min_samples_leaf=5     # absolute: ≥ 5 samples per leaf
min_samples_leaf=0.02  # relative: ≥ 2% of total samples
```

#### `max_features`

```python
# How many features to consider at each split
# Useful for adding randomness (used heavily in Random Forests)

max_features=None      # All features (deterministic)
max_features='sqrt'    # √(n_features) — good for classification
max_features='log2'    # log₂(n_features)
max_features=0.5       # 50% of features
max_features=10        # Exactly 10 features
```

#### `max_leaf_nodes`

```python
# Limit total number of leaf nodes → controls complexity
# Grows the tree "best-first" (best split globally, not depth-first)
# Alternative to max_depth for regularization

max_leaf_nodes=20      # At most 20 leaves in the whole tree
```

#### `min_impurity_decrease`

```python
# Only split if impurity decreases by at least this amount
# Prevents splits that barely improve purity
# Good alternative to depth-based stopping

min_impurity_decrease=0.01  # Minimum 1% improvement required
```

#### `class_weight`

```python
# Handle class imbalance
class_weight=None           # All classes equal weight
class_weight='balanced'     # Weight ∝ 1/class_frequency
class_weight={0: 1, 1: 10}  # Manual: minority class gets 10× weight
```

#### `ccp_alpha` — Post-Pruning

```python
# 0.0 → no pruning (default)
# Higher → more pruning → smaller tree
# Tune via cost_complexity_pruning_path()

ccp_alpha=0.01    # Light pruning
ccp_alpha=0.1     # Heavy pruning
```

### Regression Tree Hyperparameters

```python
from sklearn.tree import DecisionTreeRegressor

DecisionTreeRegressor(
    criterion='squared_error',  # 'squared_error', 'friedman_mse', 'absolute_error', 'poisson'
    splitter='best',
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features=None,
    random_state=None,
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    ccp_alpha=0.0,
)
```

---

## 12. Feature Importance

### How Feature Importance is Computed

Feature importance in CART = total **impurity reduction** contributed by a feature, weighted by the number of samples:

$$\text{FI}(j) = \frac{\sum_{t: \text{split on } j} N_t \cdot \Delta\text{Impurity}(t)}{\sum_{\text{all splits}} N_t \cdot \Delta\text{Impurity}(t)}$$

Values are normalized to sum to 1.

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_train, y_train)

# Feature importances
importances = clf.feature_importances_
feature_names = iris.feature_names
indices = np.argsort(importances)[::-1]

print("Feature ranking:")
for i, idx in enumerate(indices):
    print(f"  {i+1}. {feature_names[idx]:25s}: {importances[idx]:.4f}")

# Plot
plt.figure(figsize=(8, 5))
plt.barh(range(len(importances)),
         importances[indices[::-1]],
         color='steelblue')
plt.yticks(range(len(importances)),
           [feature_names[i] for i in indices[::-1]])
plt.xlabel('Feature Importance (Gini Decrease)')
plt.title('Decision Tree — Feature Importances')
plt.tight_layout()
plt.show()
```

### ⚠️ Caveat: Impurity-based importance is biased

Impurity-based feature importance **overestimates** high-cardinality and numerical features. Use `permutation_importance` for an unbiased estimate:

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(
    clf, X_test, y_test,
    n_repeats=10, random_state=42, n_jobs=-1
)
sorted_idx = result.importances_mean.argsort()

plt.boxplot(
    result.importances[sorted_idx].T,
    vert=False,
    labels=[feature_names[i] for i in sorted_idx]
)
plt.title("Permutation Importances (Test Set)")
plt.tight_layout()
plt.show()
```

---

## 13. Python from Scratch

```python
import numpy as np
from collections import Counter

class DecisionNode:
    """Represents a node in the decision tree."""
    def __init__(self, feature=None, threshold=None,
                 left=None, right=None, value=None):
        self.feature   = feature    # Feature index for splitting
        self.threshold = threshold  # Threshold value for split
        self.left      = left       # Left child (feature ≤ threshold)
        self.right     = right      # Right child (feature > threshold)
        self.value     = value      # Leaf prediction (not None if leaf)
    
    def is_leaf(self):
        return self.value is not None


class DecisionTreeClassifierScratch:
    """
    CART Decision Tree Classifier from scratch.
    Uses Gini impurity for splitting.
    """
    
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf  = min_samples_leaf
        self.root              = None
    
    # ── Gini Impurity ──────────────────────────────────────────
    def _gini(self, y):
        n = len(y)
        if n == 0:
            return 0
        counts = Counter(y)
        probs  = [count / n for count in counts.values()]
        return 1 - sum(p**2 for p in probs)
    
    # ── Weighted Gini after split ──────────────────────────────
    def _weighted_gini(self, y_left, y_right):
        n = len(y_left) + len(y_right)
        if n == 0:
            return 0
        return (len(y_left)/n)  * self._gini(y_left) + \
               (len(y_right)/n) * self._gini(y_right)
    
    # ── Find best split ────────────────────────────────────────
    def _best_split(self, X, y):
        best_gini     = float('inf')
        best_feature  = None
        best_threshold = None
        n_features    = X.shape[1]
        
        for feature_idx in range(n_features):
            # Try all unique values as thresholds
            thresholds = np.unique(X[:, feature_idx])
            
            for threshold in thresholds:
                mask      = X[:, feature_idx] <= threshold
                y_left    = y[mask]
                y_right   = y[~mask]
                
                # Skip if leaves don't meet min_samples_leaf
                if len(y_left)  < self.min_samples_leaf or \
                   len(y_right) < self.min_samples_leaf:
                    continue
                
                gini = self._weighted_gini(y_left, y_right)
                
                if gini < best_gini:
                    best_gini      = gini
                    best_feature   = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    # ── Majority class prediction at leaf ─────────────────────
    def _leaf_value(self, y):
        return Counter(y).most_common(1)[0][0]
    
    # ── Recursive tree building ────────────────────────────────
    def _build_tree(self, X, y, depth=0):
        n_samples  = len(y)
        n_classes  = len(np.unique(y))
        
        # Stopping conditions
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_classes == 1 or \
           n_samples < self.min_samples_split:
            return DecisionNode(value=self._leaf_value(y))
        
        # Find best split
        feature, threshold = self._best_split(X, y)
        
        if feature is None:  # No valid split found
            return DecisionNode(value=self._leaf_value(y))
        
        # Split data
        mask    = X[:, feature] <= threshold
        X_left, y_left  = X[mask],  y[mask]
        X_right, y_right = X[~mask], y[~mask]
        
        # Recursively build children
        left_child  = self._build_tree(X_left,  y_left,  depth + 1)
        right_child = self._build_tree(X_right, y_right, depth + 1)
        
        return DecisionNode(feature=feature, threshold=threshold,
                            left=left_child, right=right_child)
    
    # ── Fit ────────────────────────────────────────────────────
    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        self.root = self._build_tree(X, y)
        return self
    
    # ── Traverse for single sample ─────────────────────────────
    def _traverse(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        else:
            return self._traverse(x, node.right)
    
    # ── Predict ────────────────────────────────────────────────
    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in np.array(X)])
    
    def score(self, X, y):
        return np.mean(self.predict(X) == np.array(y))


# ── Test ───────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

tree_scratch = DecisionTreeClassifierScratch(max_depth=5, min_samples_split=10)
tree_scratch.fit(X_train, y_train)
print(f"Decision Tree (scratch) Accuracy: {tree_scratch.score(X_test, y_test):.4f}")

# Compare with sklearn
from sklearn.tree import DecisionTreeClassifier
clf_sk = DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42)
clf_sk.fit(X_train, y_train)
print(f"Decision Tree (sklearn) Accuracy: {clf_sk.score(X_test, y_test):.4f}")
```

---

## 14. Scikit-learn Implementation

### Classification Tree

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ── Data ──────────────────────────────────────────────────────
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names
target_names  = data.target_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── NOTE: Decision Trees do NOT require feature scaling ───────

# ── Model ──────────────────────────────────────────────────────
clf = DecisionTreeClassifier(
    criterion='gini',
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42
)
clf.fit(X_train, y_train)

# ── Evaluate ───────────────────────────────────────────────────
y_pred  = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)

print(f"Train Accuracy: {clf.score(X_train, y_train):.4f}")
print(f"Test  Accuracy: {clf.score(X_test,  y_test):.4f}")
print(f"Tree Depth:     {clf.get_depth()}")
print(f"Num Leaves:     {clf.get_n_leaves()}")
print("\n", classification_report(y_test, y_pred, target_names=target_names))

# Cross-validation
cv_scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── Confusion Matrix ───────────────────────────────────────────
ConfusionMatrixDisplay.from_estimator(
    clf, X_test, y_test,
    display_labels=target_names,
    cmap='Blues'
)
plt.title("Confusion Matrix — Decision Tree")
plt.show()

# ── Tree Text Representation ──────────────────────────────────
print(export_text(clf, feature_names=list(feature_names), max_depth=3))
```

### Regression Tree

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score

housing = fetch_california_housing()
X, y = housing.data, housing.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

reg = DecisionTreeRegressor(
    criterion='squared_error',
    max_depth=6,
    min_samples_leaf=20,
    random_state=42
)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE:     {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"Depth:    {reg.get_depth()}")
```

---

## 15. Full Pipeline with Best Practices

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint
import numpy as np

# ── Pipeline (scaling optional for trees, but included for consistency) ──
pipeline = Pipeline([
    ('clf', DecisionTreeClassifier(random_state=42))
])

# ── Grid Search ────────────────────────────────────────────────
param_grid = {
    'clf__criterion':            ['gini', 'entropy'],
    'clf__max_depth':            [None, 3, 5, 7, 10, 15],
    'clf__min_samples_split':    [2, 10, 20, 50],
    'clf__min_samples_leaf':     [1, 5, 10, 20],
    'clf__max_features':         [None, 'sqrt', 'log2'],
    'clf__min_impurity_decrease':[0.0, 0.001, 0.01],
    'clf__ccp_alpha':            [0.0, 0.001, 0.005, 0.01],
}

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

print(f"Best Params:   {grid_search.best_params_}")
print(f"Best CV Score: {grid_search.best_score_:.4f}")
print(f"Test Score:    {grid_search.score(X_test, y_test):.4f}")

# ── Randomized Search (faster for large grids) ────────────────
param_dist = {
    'clf__max_depth':         [None, *range(2, 20)],
    'clf__min_samples_split': randint(2, 50),
    'clf__min_samples_leaf':  randint(1, 30),
    'clf__ccp_alpha':         np.linspace(0, 0.05, 20),
    'clf__criterion':         ['gini', 'entropy'],
}

rand_search = RandomizedSearchCV(
    pipeline, param_dist,
    n_iter=100, cv=5, scoring='accuracy',
    n_jobs=-1, random_state=42
)
rand_search.fit(X_train, y_train)
print(f"\nBest Random CV Score: {rand_search.best_score_:.4f}")
print(f"Best Random Params:   {rand_search.best_params_}")
```

---

## 16. Visualizing the Tree

### Plot Tree (Matplotlib)

```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(
    clf,
    feature_names=list(feature_names),
    class_names=list(target_names),
    filled=True,           # Color nodes by majority class
    rounded=True,          # Round corners
    impurity=True,         # Show Gini/entropy
    proportion=False,      # Show sample counts (not fractions)
    ax=ax,
    max_depth=3,           # Show only first 3 levels
    fontsize=10
)
plt.title("Decision Tree — First 3 Levels", fontsize=15)
plt.tight_layout()
plt.show()
```

### Export to Graphviz (publication quality)

```python
from sklearn.tree import export_graphviz
import graphviz

dot_data = export_graphviz(
    clf,
    out_file=None,
    feature_names=list(feature_names),
    class_names=list(target_names),
    filled=True,
    rounded=True,
    special_characters=True,
    max_depth=3
)
graph = graphviz.Source(dot_data)
graph.render("decision_tree", format='png', cleanup=True)
graph  # displays inline in Jupyter
```

### Decision Boundary Plot (2D)

```python
from sklearn.datasets import make_classification
from sklearn.inspection import DecisionBoundaryDisplay

X_2d, y_2d = make_classification(
    n_features=2, n_redundant=0, n_informative=2,
    random_state=42, n_clusters_per_class=1
)

clf_2d = DecisionTreeClassifier(max_depth=4, random_state=42)
clf_2d.fit(X_2d, y_2d)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, depth in zip(axes, [1, 3, None]):
    clf_d = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf_d.fit(X_2d, y_2d)
    DecisionBoundaryDisplay.from_estimator(
        clf_d, X_2d, ax=ax, alpha=0.3, cmap='RdBu'
    )
    ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, cmap='RdBu', edgecolors='k')
    ax.set_title(f"max_depth={depth} | "
                 f"Train: {clf_d.score(X_2d, y_2d):.2f}")
plt.suptitle("Effect of max_depth on Decision Boundary", fontsize=13)
plt.tight_layout()
plt.show()
```

---

## 17. Evaluation Metrics

### Classification

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, log_loss,
    classification_report, ConfusionMatrixDisplay, RocCurveDisplay
)

y_pred  = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
print(f"ROC AUC:   {roc_auc_score(y_test, y_proba):.4f}")
print(f"Log Loss:  {log_loss(y_test, y_proba):.4f}")

# ROC Curve
RocCurveDisplay.from_estimator(clf, X_test, y_test)
plt.title("ROC Curve — Decision Tree")
plt.show()
```

### Regression

```python
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    root_mean_squared_error, r2_score
)

y_pred = reg.predict(X_test)
print(f"MAE:  {mean_absolute_error(y_test, y_pred):.4f}")
print(f"MSE:  {mean_squared_error(y_test, y_pred):.4f}")
print(f"RMSE: {root_mean_squared_error(y_test, y_pred):.4f}")
print(f"R²:   {r2_score(y_test, y_pred):.4f}")
```

---

## 18. Pros and Cons

### ✅ Advantages

| Advantage | Explanation |
|-----------|-------------|
| **Interpretable** | Rules visible — white-box model |
| **No scaling needed** | Distance not used |
| **Handles mixed types** | Numerical + categorical features |
| **Non-linear boundaries** | Captures any shape decision boundary |
| **Fast prediction** | O(depth) — just traverse one path |
| **No assumptions** | Non-parametric, no distribution assumptions |
| **Handles missing values** | (C4.5/C5.0) — sklearn requires imputation |
| **Feature selection built-in** | Unimportant features are simply not split on |
| **Multi-output support** | Can predict multiple targets simultaneously |

### ❌ Disadvantages

| Disadvantage | Explanation |
|--------------|-------------|
| **Overfitting** | Fully grown tree memorizes data |
| **High variance** | Small data changes → very different trees |
| **Greedy algorithm** | Locally optimal splits, not globally optimal |
| **Biased toward high-cardinality features** | Gini/entropy can favor features with many values |
| **Axis-aligned splits only** | Can't capture diagonal boundaries well |
| **Unstable** | Sensitive to data perturbations |
| **Imbalanced classes** | Majority class dominates without `class_weight` |

### When to Use Decision Trees

✅ **Use when:**
- Interpretability is critical (legal, medical, financial)
- Quick exploratory baseline
- Mixed feature types
- Non-linear relationships
- Rule extraction needed
- As base learner for ensemble methods (Random Forest, Gradient Boosting)

❌ **Avoid when:**
- High accuracy is priority (use ensembles instead)
- Data is noisy / high-dimensional
- Diagonal decision boundaries expected

---

## 19. Interview Questions

**Q1: What is Gini impurity and how is it computed?**
> Gini impurity measures the probability that a randomly chosen sample would be incorrectly classified if assigned a random label according to the class distribution. Formula: $1 - \sum p_k^2$. Ranges from 0 (pure) to 0.5 (max impurity, binary).

**Q2: What is Information Gain?**
> Information Gain = entropy of parent − weighted entropy of children after split. We maximize IG to find the best split. IG = 0 means the split provided no new information.

**Q3: Why doesn't a Decision Tree need feature scaling?**
> Because splits are based on comparisons ($x_j \leq t$), not distances. A monotonic transformation of any feature gives identical splits.

**Q4: What is the difference between Gini and Entropy? Which should you prefer?**
> Both measure impurity and give very similar results. Gini is faster (no log computation). Entropy tends to create more balanced splits. In practice, the difference in final accuracy is negligible — just use Gini (default).

**Q5: How does a Decision Tree handle overfitting?**
> Pre-pruning: `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_leaf_nodes`, `min_impurity_decrease`. Post-pruning: `ccp_alpha` (cost-complexity pruning). Cross-validation is used to find optimal hyperparameter values.

**Q6: Is there gradient descent in Decision Trees?**
> No. Decision Trees use greedy exhaustive search — they try all features and all thresholds and pick the one with best impurity reduction. No gradient, no backprop. (Note: Gradient Boosted Trees DO use gradients, but that's an ensemble on top of trees.)

**Q7: What is the time complexity of training a Decision Tree?**
> O(N · d · log N) for CART — at each of the O(log N) levels, we try all N × d feature-threshold combinations. Prediction is O(depth).

**Q8: How does a Decision Tree handle multi-class classification?**
> Naturally — at each leaf, it predicts the majority class among all K classes. No one-vs-rest or modification needed.

**Q9: What is cost-complexity pruning (ccp_alpha)?**
> A post-pruning technique. Each subtree is penalized by $\alpha \times |T|$ where |T| = number of leaves. Higher α → more pruning. The optimal α is found via cross-validation using `cost_complexity_pruning_path()`.

**Q10: Why is the Decision Tree called a "greedy" algorithm?**
> At each node, it picks the locally best split without considering future splits. A globally optimal tree is NP-hard to find, so greedy is used. This means the tree isn't guaranteed to be the best possible structure.

**Q11: What is a decision stump?**
> A Decision Tree with `max_depth=1` — a single split. Used as base learners in AdaBoost.

**Q12: How does a Decision Tree handle imbalanced classes?**
> By default it doesn't — the majority class dominates. Solution: `class_weight='balanced'` assigns weights inversely proportional to class frequency.

**Q13: What's the difference between max_depth and max_leaf_nodes?**
> `max_depth` limits depth level. `max_leaf_nodes` limits total leaves and grows the tree "best-first" (picking the globally best split at each step, not just within one branch). Both control complexity; `max_leaf_nodes` often gives better trees.

---

## 20. Resources

### 📘 Andrew Ng — ML Specialization (Coursera)
- **Course 2, Week 4**: Decision Trees — full dedicated week
  - Purity measures: Gini and entropy
  - Information gain derivation
  - Recursive splitting algorithm
  - One-hot encoding for categorical features
  - Continuous feature splitting
  - Regression trees
  - When to use trees vs neural networks
- 🔗 https://www.coursera.org/specializations/machine-learning-introduction

### 📗 Hands-On Machine Learning (Aurélien Géron)
- **Chapter 6**: Decision Trees — dedicated chapter
  - CART algorithm walkthrough
  - Gini vs entropy deep dive
  - Regularization hyperparameters
  - Regression trees
  - Instability analysis
- 🔗 https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/

### 🎬 StatQuest with Josh Starmer (YouTube)
- **"Decision Trees, Part 1 - How They Work"** → Core intuition
- **"Decision Trees, Part 2 - Feature Selection and Missing Data"** → Splitting details
- **"Regression Trees, Clearly Explained"** → Regression variant
- **"How to Prune Regression Trees"** → Pruning with ccp_alpha
- **"Classification Trees in Python (sklearn)"** → Coding tutorial
- 🔗 https://www.youtube.com/@statquest

### 📙 Introduction to Statistical Learning (ISLP — Python Edition)
- **Chapter 8**: Tree-Based Methods — dedicated chapter
  - 8.1: Decision Trees (classification + regression)
  - 8.1.1: Regression Trees
  - 8.1.2: Classification Trees
  - 8.1.3: Advantages and Disadvantages
  - Lab: Python implementation with sklearn
- Free PDF: 🔗 https://www.statlearning.com/

### 📜 Scikit-learn Documentation
- DecisionTreeClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
- DecisionTreeRegressor: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
- User Guide (Trees): https://scikit-learn.org/stable/modules/tree.html
- Cost-complexity pruning: https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html
- Plot tree: https://scikit-learn.org/stable/modules/generated/sklearn.tree.plot_tree.html

### 📺 Additional Resources
- **Original CART paper**: Breiman et al. (1984) — "Classification and Regression Trees"
- **C4.5 paper**: Quinlan (1993) — "C4.5: Programs for Machine Learning"
- **Visual explanation**: http://www.r2d3.us/visual-intro-to-machine-learning-part-1/ (best visual on the web)

---

## 🎯 Quick Cheat Sheet

```
Algorithm:   Non-parametric, greedy, recursive binary splitting
Training:    O(N · d · log N)   — exhaustive search, no gradient descent
Prediction:  O(depth)           — traverse one root-to-leaf path
Memory:      O(N)               — stores tree structure

Split criteria:
  Classification → Gini (default) or Entropy (Information Gain)
  Regression     → squared_error (MSE), absolute_error (MAE)

Key formula (Gini):    1 - Σ p_k²
Key formula (Entropy): -Σ p_k · log₂(p_k)
Key formula (IG):      H(parent) - [N_L/N · H(L) + N_R/N · H(R)]

Regularization (pre-pruning):
  max_depth            ← most important knob
  min_samples_split    ← min samples to attempt split
  min_samples_leaf     ← min samples at leaf
  max_leaf_nodes       ← limit total leaves (best-first growth)
  min_impurity_decrease← min gain required to split
  ccp_alpha            ← post-pruning (tune via pruning path + CV)

Scaling:     NOT required
Missing data: NOT handled (impute beforehand)
Classes:     class_weight='balanced' for imbalanced data

No gradient descent — splits found by exhaustive greedy search
Feature importance = normalized weighted impurity decrease per feature

Bias-Variance:
  Deep tree  → low bias,  high variance → overfitting
  Shallow tree → high bias, low variance → underfitting
  → Use max_depth + ccp_alpha + GridSearchCV to find sweet spot
```

---

*Notes compiled from: Andrew Ng ML Specialization (Course 2, Week 4) · Hands-On ML Ch.6 (Géron) · StatQuest Decision Tree series · ISLP Ch.8 (James et al.) · Scikit-learn Docs*
