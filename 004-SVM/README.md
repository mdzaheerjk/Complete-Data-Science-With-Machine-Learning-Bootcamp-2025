# 🧠 Support Vector Machines (SVM) — Complete ML/DL Job-Ready Notes

> **Sources:** Andrew Ng ML Specialization (Coursera) · Hands-On ML with Scikit-Learn (Aurélien Géron) · StatQuest (Josh Starmer) · Scikit-Learn Docs · ISLP (James et al.)

---

## 📚 Table of Contents

1. [What is SVM?](#1-what-is-svm)
2. [Intuition — Maximum Margin Classifier](#2-intuition--maximum-margin-classifier)
3. [The Math — Hard Margin SVM](#3-the-math--hard-margin-svm)
4. [Soft Margin SVM (C Parameter)](#4-soft-margin-svm)
5. [The Dual Problem & Support Vectors](#5-the-dual-problem--support-vectors)
6. [Kernel Trick](#6-kernel-trick)
7. [SVM Cost Function & Hinge Loss](#7-svm-cost-function--hinge-loss)
8. [Gradient Descent for SVM](#8-gradient-descent-for-svm)
9. [SVM for Regression (SVR)](#9-svm-for-regression-svr)
10. [Multiclass SVM](#10-multiclass-svm)
11. [Assumptions & When to Use SVM](#11-assumptions--when-to-use-svm)
12. [Evaluation Metrics](#12-evaluation-metrics)
13. [Python from Scratch](#13-python-from-scratch)
14. [Scikit-Learn Implementation](#14-scikit-learn-implementation)
15. [Hyperparameters & Tuning](#15-hyperparameters--tuning)
16. [Bias-Variance & Learning Curves](#16-bias-variance--learning-curves)
17. [SVM vs Logistic Regression](#17-svm-vs-logistic-regression)
18. [Common Interview Questions](#18-common-interview-questions)
19. [Resources](#19-resources)

---

## 1. What is SVM?

> **Simple Statement:** SVM finds the **widest possible street** (maximum margin hyperplane) between two classes. Points on the edge of the street are called **support vectors** — they are the only points that matter for defining the boundary.

- **Task type:** Supervised Learning → Classification & Regression
- **Output:** Class label (SVC) or continuous value (SVR)
- **Key idea:** Maximize the margin between classes, not just find any separating line
- **Powerful because:** Works well in high dimensions, effective even when n_features > n_samples, memory efficient (only support vectors matter)

**Real-world examples:**
- Text classification (spam, sentiment)
- Image recognition (face detection)
- Bioinformatics (gene expression classification)
- Handwriting recognition
- Anomaly detection (One-Class SVM)

---

## 2. Intuition — Maximum Margin Classifier

> **Simple Statement:** Imagine two groups of points on a table. Many lines can separate them. SVM finds the line that is **furthest from both groups** — the one with the largest "safety zone." This safety zone is called the **margin**.

```
        Class -1          Class +1
           ●                 ○
        ●     ●           ○     ○
           ●    ←margin→  ○
        ●     ●     |     ○     ○
                    |
              Decision Boundary (hyperplane)
              ←   w·x + b = 0   →

Support vectors: points closest to the boundary (on the margin lines)
Margin width = 2 / ||w||
```

**Why maximize margin?**
- A larger margin means more tolerance to new unseen points
- Smaller generalization error (VC theory / structural risk minimization)
- The model that just barely separates data is fragile; SVM picks the most robust one

---

## 3. The Math — Hard Margin SVM

> **Simple Statement (Hard Margin):** Assume data is perfectly linearly separable. Find the hyperplane $\mathbf{w}^T\mathbf{x} + b = 0$ such that all Class +1 points are above the upper margin line and all Class -1 points are below the lower margin line, and the gap between the lines is as wide as possible.

### Hyperplane Definition

$$\mathbf{w}^T \mathbf{x} + b = 0$$

- $\mathbf{w}$ = weight vector (normal to the hyperplane)
- $b$ = bias
- In 2D: $w_1 x_1 + w_2 x_2 + b = 0$ — a line

### Margin Lines

$$\mathbf{w}^T \mathbf{x} + b = +1 \quad \text{(upper margin — Class +1 side)}$$
$$\mathbf{w}^T \mathbf{x} + b = -1 \quad \text{(lower margin — Class -1 side)}$$

### Hard Margin Constraints

For all training examples $(x^{(i)}, y^{(i)})$ where $y^{(i)} \in \{-1, +1\}$:

$$y^{(i)}(\mathbf{w}^T \mathbf{x}^{(i)} + b) \geq 1 \quad \forall i$$

This single constraint elegantly combines both class requirements:
- If $y^{(i)} = +1$: $\mathbf{w}^T\mathbf{x}^{(i)} + b \geq +1$
- If $y^{(i)} = -1$: $\mathbf{w}^T\mathbf{x}^{(i)} + b \leq -1$

### Width of the Margin

The perpendicular distance between the two margin lines:

$$\text{Margin} = \frac{2}{\|\mathbf{w}\|}$$

**Derivation:** A point $\mathbf{x}_+$ is on the positive margin ($\mathbf{w}^T\mathbf{x}_+ + b = 1$) and $\mathbf{x}_-$ on the negative ($\mathbf{w}^T\mathbf{x}_- + b = -1$). The width is the projection of $(\mathbf{x}_+ - \mathbf{x}_-)$ onto the unit normal $\frac{\mathbf{w}}{\|\mathbf{w}\|}$:

$$\text{width} = (\mathbf{x}_+ - \mathbf{x}_-)^T \frac{\mathbf{w}}{\|\mathbf{w}\|} = \frac{(1-b) - (-1-b)}{\|\mathbf{w}\|} = \frac{2}{\|\mathbf{w}\|}$$

### Optimization Problem (Hard Margin)

$$\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2$$

$$\text{subject to: } y^{(i)}(\mathbf{w}^T \mathbf{x}^{(i)} + b) \geq 1 \quad \forall i$$

- **Minimizing** $\|\mathbf{w}\|^2$ is equivalent to **maximizing** margin $\frac{2}{\|\mathbf{w}\|}$
- The $\frac{1}{2}$ makes the derivative cleaner
- This is a **convex quadratic programming** problem → always has a unique global solution

---

## 4. Soft Margin SVM

> **Simple Statement:** Real data is rarely perfectly separable. Soft Margin SVM allows some points to be inside the margin or even on the wrong side, but penalizes them with **slack variables**. The parameter `C` controls how much we penalize violations.

### Slack Variables

Introduce $\xi^{(i)} \geq 0$ (xi, "slack") for each training example:

$$\xi^{(i)} = \begin{cases} 0 & \text{if correctly classified outside margin} \\ 1 - y^{(i)}(\mathbf{w}^T\mathbf{x}^{(i)} + b) & \text{if inside margin or misclassified} \end{cases}$$

- $\xi^{(i)} = 0$: Point correctly outside or on margin ✅
- $0 < \xi^{(i)} < 1$: Point inside margin but correct side
- $\xi^{(i)} = 1$: Point on decision boundary
- $\xi^{(i)} > 1$: Point misclassified ❌

### Soft Margin Optimization Problem

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^{m} \xi^{(i)}$$

$$\text{subject to: } y^{(i)}(\mathbf{w}^T\mathbf{x}^{(i)} + b) \geq 1 - \xi^{(i)}, \quad \xi^{(i)} \geq 0 \quad \forall i$$

### The C Parameter

| C value | Effect | Model behavior |
|---------|--------|----------------|
| **Large C** | Penalize violations heavily | Narrow margin, fewer violations, may overfit |
| **Small C** | Allow more violations | Wide margin, more violations, may underfit |
| $C \to \infty$ | No violations allowed | Hard margin SVM |
| $C \to 0$ | Ignore all violations | No boundary |

> **Key insight:** C in SVM is the **opposite** of regularization $\lambda$ from logistic regression. Large C = low regularization. Small C = high regularization (wider, softer margin).

---

## 5. The Dual Problem & Support Vectors

> **Simple Statement:** We can reformulate the SVM optimization using Lagrange multipliers. This dual form reveals that only the support vectors (points with $\alpha_i > 0$) determine the solution — all other points are irrelevant!

### Lagrangian (Primal)

$$\mathcal{L} = \frac{1}{2}\|\mathbf{w}\|^2 - \sum_{i=1}^{m} \alpha_i \left[ y^{(i)}(\mathbf{w}^T\mathbf{x}^{(i)} + b) - 1 \right]$$

Taking derivatives and setting to zero:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = 0 \implies \mathbf{w} = \sum_{i=1}^{m} \alpha_i y^{(i)} \mathbf{x}^{(i)}$$

$$\frac{\partial \mathcal{L}}{\partial b} = 0 \implies \sum_{i=1}^{m} \alpha_i y^{(i)} = 0$$

### Dual Problem

$$\max_{\boldsymbol{\alpha}} \sum_{i=1}^{m} \alpha_i - \frac{1}{2} \sum_{i=1}^{m}\sum_{j=1}^{m} \alpha_i \alpha_j y^{(i)} y^{(j)} \mathbf{x}^{(i)T}\mathbf{x}^{(j)}$$

$$\text{subject to: } \alpha_i \geq 0, \quad \sum_{i=1}^{m} \alpha_i y^{(i)} = 0$$

### Key Insight: Support Vectors

From KKT conditions: $\alpha_i [y^{(i)}(\mathbf{w}^T\mathbf{x}^{(i)}+b) - 1] = 0$

- If $\alpha_i > 0$: the constraint is active → point is **on the margin** → **support vector**
- If $\alpha_i = 0$: point is outside the margin → **does not contribute** to $\mathbf{w}$

**$\mathbf{w}$ is a linear combination of support vectors only:**

$$\mathbf{w} = \sum_{i \in \text{SVs}} \alpha_i y^{(i)} \mathbf{x}^{(i)}$$

### Prediction using dual form:

$$f(\mathbf{x}) = \text{sign}\left(\sum_{i \in \text{SVs}} \alpha_i y^{(i)} \mathbf{x}^{(i)T}\mathbf{x} + b\right)$$

Notice: only **dot products** $\mathbf{x}^{(i)T}\mathbf{x}$ appear — this is where the kernel trick enters!

---

## 6. Kernel Trick

> **Simple Statement:** For non-linearly separable data, we'd normally need to manually create polynomial or other transformed features — expensive! The kernel trick lets us **compute dot products in a high-dimensional space without actually going there**, using a kernel function $K(\mathbf{x}^{(i)}, \mathbf{x}^{(j)})$.

### The Big Idea

Replace every dot product $\mathbf{x}^{(i)T}\mathbf{x}^{(j)}$ with a kernel function $K(\mathbf{x}^{(i)}, \mathbf{x}^{(j)})$:

$$f(\mathbf{x}) = \text{sign}\left(\sum_{i \in \text{SVs}} \alpha_i y^{(i)} K(\mathbf{x}^{(i)}, \mathbf{x}) + b\right)$$

The kernel implicitly computes $\phi(\mathbf{x}^{(i)})^T \phi(\mathbf{x}^{(j)})$ where $\phi$ maps to a higher-dimensional space, but we never compute $\phi$ explicitly.

### Common Kernels

#### Linear Kernel
$$K(\mathbf{x}^{(i)}, \mathbf{x}^{(j)}) = \mathbf{x}^{(i)T}\mathbf{x}^{(j)}$$
- No transformation; standard dot product
- Use when data is linearly separable
- Fast, no kernel hyperparameters
- **sklearn:** `kernel='linear'`

#### Polynomial Kernel
$$K(\mathbf{x}^{(i)}, \mathbf{x}^{(j)}) = (\gamma \, \mathbf{x}^{(i)T}\mathbf{x}^{(j)} + r)^d$$
- $d$ = degree, $r$ = coef0, $\gamma$ = gamma
- Implicitly maps to degree-$d$ polynomial feature space
- Good for image classification, NLP
- **sklearn:** `kernel='poly'`, hyperparams: `degree`, `gamma`, `coef0`

#### RBF (Radial Basis Function) / Gaussian Kernel ⭐ Most Popular
$$K(\mathbf{x}^{(i)}, \mathbf{x}^{(j)}) = \exp\left(-\gamma \|\mathbf{x}^{(i)} - \mathbf{x}^{(j)}\|^2\right)$$

where $\gamma = \frac{1}{2\sigma^2}$

- Maps to **infinite-dimensional** feature space
- Measures similarity via distance; $K \to 1$ when points are identical, $K \to 0$ far apart
- Works great for most non-linear problems
- **sklearn:** `kernel='rbf'`, hyperparameter: `gamma`

**Effect of $\gamma$:**
| $\gamma$ | Effect |
|---------|--------|
| **Large** $\gamma$ | Narrow Gaussian, each point influences only its close neighbors → wiggly boundary, overfitting |
| **Small** $\gamma$ | Wide Gaussian, smooth decision boundary, may underfit |

#### Sigmoid Kernel
$$K(\mathbf{x}^{(i)}, \mathbf{x}^{(j)}) = \tanh(\gamma \, \mathbf{x}^{(i)T}\mathbf{x}^{(j)} + r)$$
- Similar to neural network with tanh activation
- Not always positive semi-definite
- **sklearn:** `kernel='sigmoid'`

#### Mercer's Condition
A function $K$ is a valid kernel if and only if the kernel matrix $K_{ij} = K(\mathbf{x}^{(i)}, \mathbf{x}^{(j)})$ is **symmetric positive semi-definite** for any set of inputs.

```python
import numpy as np

# Manual kernel functions
def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def rbf_kernel(x1, x2, gamma=1.0):
    return np.exp(-gamma * np.sum((x1 - x2) ** 2))

def poly_kernel(x1, x2, degree=3, gamma=1.0, coef0=1.0):
    return (gamma * np.dot(x1, x2) + coef0) ** degree

# Kernel matrix (Gram matrix)
def compute_kernel_matrix(X, kernel_fn, **kwargs):
    m = X.shape[0]
    K = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            K[i, j] = kernel_fn(X[i], X[j], **kwargs)
    return K
```

---

## 7. SVM Cost Function & Hinge Loss

> **Simple Statement:** The SVM loss function is called **Hinge Loss**. It says: if a point is correctly classified and outside the margin, loss = 0. But if it's inside or on the wrong side of the margin, it incurs a linear penalty.

### Hinge Loss

For one example with label $y \in \{-1, +1\}$:

$$\ell(y, f(\mathbf{x})) = \max(0, 1 - y \cdot f(\mathbf{x}))$$

where $f(\mathbf{x}) = \mathbf{w}^T\mathbf{x} + b$

- If $y \cdot f(\mathbf{x}) \geq 1$: correctly classified, outside margin → **loss = 0**
- If $y \cdot f(\mathbf{x}) < 1$: inside margin or wrong side → **loss = $1 - y \cdot f(\mathbf{x})$**

### SVM Objective as Regularized Hinge Loss

$$J(\mathbf{w}, b) = \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^{m} \max\left(0, 1 - y^{(i)}(\mathbf{w}^T\mathbf{x}^{(i)} + b)\right)$$

This can be rewritten as:

$$J = \underbrace{\lambda \|\mathbf{w}\|^2}_{\text{regularization}} + \underbrace{\frac{1}{m}\sum_{i=1}^m \max(0, 1 - y^{(i)} f(\mathbf{x}^{(i)}))}_{\text{hinge loss}}$$

where $\lambda = \frac{1}{2mC}$.

### Hinge Loss vs Log Loss Comparison

```
Loss
 |
 |  Log Loss (LogReg)
3|  ╲
 |   ╲
2|    ╲ ___ Hinge Loss (SVM)
 |    /     ╲
1|   /        ╲_______________
 |  /
0|___________________________ y·f(x)
  -2  -1   0   1   2   3
```

- **Log Loss:** Penalizes even correctly classified confident predictions (slowly decays to 0)
- **Hinge Loss:** Zero loss once point is correctly outside margin (flat at 0)
- Both are convex upper bounds on 0-1 loss

---

## 8. Gradient Descent for SVM

> **Simple Statement:** We can train a linear SVM using subgradient descent because hinge loss is not differentiable at $yf(x)=1$, but its subgradient exists everywhere.

### Subgradient of Hinge Loss

The hinge loss $\ell_i = \max(0, 1 - y^{(i)}(\mathbf{w}^T\mathbf{x}^{(i)} + b))$ has subgradient:

$$\frac{\partial \ell_i}{\partial \mathbf{w}} = \begin{cases} 0 & \text{if } y^{(i)}(\mathbf{w}^T\mathbf{x}^{(i)} + b) \geq 1 \\ -y^{(i)}\mathbf{x}^{(i)} & \text{if } y^{(i)}(\mathbf{w}^T\mathbf{x}^{(i)} + b) < 1 \end{cases}$$

### Full Gradient of SVM Objective

$$\frac{\partial J}{\partial \mathbf{w}} = \mathbf{w} - C \sum_{i: y^{(i)}f(\mathbf{x}^{(i)}) < 1} y^{(i)}\mathbf{x}^{(i)}$$

$$\frac{\partial J}{\partial b} = -C \sum_{i: y^{(i)}f(\mathbf{x}^{(i)}) < 1} y^{(i)}$$

### Update Rules

$$\mathbf{w} := \mathbf{w} - \alpha \frac{\partial J}{\partial \mathbf{w}}$$

Expanding:

For each training example $i$:
- If $y^{(i)}(\mathbf{w}^T\mathbf{x}^{(i)} + b) \geq 1$ (correctly outside margin):
  $$\mathbf{w} := \mathbf{w} - \alpha \mathbf{w} = (1 - \alpha)\mathbf{w}$$
- If $y^{(i)}(\mathbf{w}^T\mathbf{x}^{(i)} + b) < 1$ (violation):
  $$\mathbf{w} := \mathbf{w} - \alpha(\mathbf{w} - C \cdot y^{(i)}\mathbf{x}^{(i)}) = (1-\alpha)\mathbf{w} + \alpha C y^{(i)}\mathbf{x}^{(i)}$$
  $$b := b + \alpha C y^{(i)}$$

---

## 9. SVM for Regression (SVR)

> **Simple Statement:** SVR flips the SVM idea for regression. Instead of finding a margin that separates classes, SVR fits a **tube of width $2\varepsilon$** around the predictions. Points inside the tube have zero loss; points outside are penalized.

### SVR Objective

$$\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^{m} (\xi^{(i)} + \xi^{(i)*})$$

subject to:
$$y^{(i)} - (\mathbf{w}^T\mathbf{x}^{(i)} + b) \leq \varepsilon + \xi^{(i)}$$
$$(\mathbf{w}^T\mathbf{x}^{(i)} + b) - y^{(i)} \leq \varepsilon + \xi^{(i)*}$$
$$\xi^{(i)}, \xi^{(i)*} \geq 0$$

### ε-insensitive Loss

$$\ell_\varepsilon(y, \hat{y}) = \max(0, |y - \hat{y}| - \varepsilon)$$

- Points within $\varepsilon$ of prediction → zero loss
- Points outside → linear penalty on excess distance

```python
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

# Generate regression data
np.random.seed(42)
X = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.sin(X).ravel() + np.random.randn(100) * 0.1

# SVR with RBF kernel
svr_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svr', SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1))
])
svr_pipe.fit(X, y)
print(f"SVR R²: {svr_pipe.score(X, y):.4f}")
```

---

## 10. Multiclass SVM

### One-vs-Rest (OvR)
- Train $K$ classifiers: class $k$ vs all others
- Predict class with highest raw score (decision function value)
- **Default in sklearn LinearSVC**

### One-vs-One (OvO)
- Train $K(K-1)/2$ binary classifiers (one per pair)
- Use majority voting
- **Default in sklearn SVC**
- Faster to train each model (smaller data per classifier)

```python
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

# OvO (default for SVC)
svc_ovo = SVC(kernel='rbf', C=1.0, decision_function_shape='ovo')

# OvR
svc_ovr = SVC(kernel='rbf', C=1.0, decision_function_shape='ovr')

# Explicit OvR wrapper
svc_ovr_explicit = OneVsRestClassifier(SVC(kernel='rbf', C=1.0))
```

---

## 11. Assumptions & When to Use SVM

### When SVM Works Best
- ✅ High-dimensional feature spaces (text, genomics) — works even when $n > m$
- ✅ Clear margin of separation in the data
- ✅ Small to medium datasets (< ~100k samples; kernel SVM scales as $O(m^2)$ to $O(m^3)$)
- ✅ Non-linear boundaries (with RBF kernel)
- ✅ When you don't need probability estimates

### When SVM Struggles
- ❌ Very large datasets — slow training ($O(m^2 n)$ for kernel SVM)
- ❌ Heavy class overlap
- ❌ Massive feature engineering needed
- ❌ Need calibrated probabilities (requires extra Platt scaling)
- ❌ Missing values — SVM doesn't handle them natively

### Key Assumptions
1. **Feature scaling required** — SVM is not scale-invariant; StandardScaler is essential
2. **No direct probability output** — uses Platt scaling (`probability=True`) as post-hoc approximation
3. **No native multiclass** — uses OvO or OvR internally

---

## 12. Evaluation Metrics

Same classification metrics as Logistic Regression, plus:

```python
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

# Decision function (signed distance from hyperplane)
decision_scores = svc_model.decision_function(X_test)

# For binary, use decision scores for ROC-AUC
# (more informative than predict_proba when probability=False)
roc_auc = roc_auc_score(y_test, decision_scores)
print(f"ROC-AUC (decision function): {roc_auc:.4f}")

# With probability=True (slower, uses Platt scaling)
svc_prob = SVC(kernel='rbf', C=1.0, probability=True)
svc_prob.fit(X_train, y_train)
y_proba = svc_prob.predict_proba(X_test)[:, 1]
roc_auc_prob = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC (probability):        {roc_auc_prob:.4f}")
```

---

## 13. Python from Scratch

### 13.1 Linear SVM with Subgradient Descent

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class LinearSVM:
    """
    Linear SVM trained with subgradient descent.
    Labels must be in {-1, +1}.
    """
    def __init__(self, C=1.0, learning_rate=0.001,
                 n_iterations=1000, verbose=False):
        self.C       = C
        self.lr      = learning_rate
        self.n_iter  = n_iterations
        self.verbose = verbose
        self.w       = None
        self.b       = 0.0
        self.losses  = []

    def _hinge_loss(self, X, y):
        margins = y * (X @ self.w + self.b)
        loss = 0.5 * np.dot(self.w, self.w) + \
               self.C * np.mean(np.maximum(0, 1 - margins))
        return loss

    def fit(self, X, y):
        """y must be in {-1, +1}"""
        m, n = X.shape
        self.w = np.zeros(n)
        self.b = 0.0

        for epoch in range(self.n_iter):
            # Shuffle data each epoch (SGD)
            idx = np.random.permutation(m)
            X_s, y_s = X[idx], y[idx]

            for i in range(m):
                xi, yi = X_s[i], y_s[i]
                margin = yi * (np.dot(self.w, xi) + self.b)

                if margin >= 1:
                    # Correctly outside margin — only regularization gradient
                    self.w -= self.lr * self.w
                else:
                    # Violation — hinge loss gradient + regularization
                    self.w -= self.lr * (self.w - self.C * yi * xi)
                    self.b += self.lr * self.C * yi

            loss = self._hinge_loss(X, y)
            self.losses.append(loss)

            if self.verbose and epoch % 100 == 0:
                print(f"Epoch {epoch:5d} | Loss: {loss:.6f}")

        return self

    def decision_function(self, X):
        return X @ self.w + self.b

    def predict(self, X):
        return np.sign(self.decision_function(X)).astype(int)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


# ── Example usage ─────────────────────────────────────────────────
X, y = make_classification(n_samples=500, n_features=2, n_redundant=0,
                             n_informative=2, random_state=42)
y = 2 * y - 1  # Convert {0,1} → {-1,+1}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

svm = LinearSVM(C=1.0, learning_rate=0.001, n_iterations=500, verbose=True)
svm.fit(X_train_s, y_train)

print(f"\nTrain Accuracy: {svm.score(X_train_s, y_train):.4f}")
print(f"Test  Accuracy: {svm.score(X_test_s,  y_test):.4f}")
```

### 13.2 Decision Boundary Visualization

```python
def plot_svm_boundary(model, X, y, scaler=None, title="SVM Decision Boundary"):
    """Works with sklearn SVC or our LinearSVM."""
    h = 0.02
    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(np.arange(x1_min, x1_max, h),
                          np.arange(x2_min, x2_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]

    Z = model.decision_function(grid).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, levels=[-3, -1, 0, 1, 3],
                 colors=['#FFAAAA', '#FFDDDD', '#DDFFDD', '#AAFFAA'], alpha=0.7)
    plt.contour(xx, yy, Z, levels=[-1, 0, 1],
                colors=['red', 'black', 'blue'],
                linestyles=['--', '-', '--'], linewidths=[1.5, 2, 1.5])

    # Plot points — highlight support vectors if available
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k', s=30, zorder=3)

    plt.title(title)
    plt.xlabel("Feature 1"); plt.ylabel("Feature 2")
    plt.annotate("Margin lines (dashed)", xy=(0.05, 0.95), xycoords='axes fraction',
                 fontsize=9, color='darkblue')
    plt.grid(True, alpha=0.3)
    plt.show()

plot_svm_boundary(svm, X_train_s, y_train, title="Linear SVM (from scratch)")
```

### 13.3 RBF Kernel SVM from Scratch

```python
class KernelSVM:
    """
    Kernel SVM using scipy's quadratic programming solver.
    Educational — sklearn's SVC is production-ready.
    """
    def __init__(self, C=1.0, kernel='rbf', gamma=1.0):
        self.C      = C
        self.kernel = kernel
        self.gamma  = gamma

    def _kernel(self, x1, x2):
        if self.kernel == 'rbf':
            return np.exp(-self.gamma * np.sum((x1 - x2) ** 2))
        elif self.kernel == 'linear':
            return np.dot(x1, x2)
        elif self.kernel == 'poly':
            return (np.dot(x1, x2) + 1) ** 2

    def _compute_gram_matrix(self, X1, X2):
        m1, m2 = len(X1), len(X2)
        K = np.zeros((m1, m2))
        for i in range(m1):
            for j in range(m2):
                K[i, j] = self._kernel(X1[i], X2[j])
        return K

    def fit(self, X, y):
        """y in {-1, +1}. Uses scipy QP solver."""
        from scipy.optimize import minimize

        m = len(y)
        K = self._compute_gram_matrix(X, X)

        # Objective: maximize Σαᵢ - ½ Σᵢⱼ αᵢαⱼyᵢyⱼK(xᵢ,xⱼ)
        # Equivalently: minimize ½αᵀQα - 1ᵀα
        Q = np.outer(y, y) * K

        def objective(alpha):
            return 0.5 * alpha @ Q @ alpha - np.sum(alpha)

        def grad_objective(alpha):
            return Q @ alpha - np.ones(m)

        constraints = {'type': 'eq', 'fun': lambda a: np.dot(a, y), 'jac': lambda a: y}
        bounds = [(0, self.C)] * m
        alpha0 = np.zeros(m)

        result = minimize(objective, alpha0, jac=grad_objective,
                          method='SLSQP', bounds=bounds,
                          constraints=constraints,
                          options={'maxiter': 500, 'ftol': 1e-8})

        self.alphas = result.x
        self.X_train = X
        self.y_train = y

        # Support vectors: αᵢ > threshold
        sv_mask = self.alphas > 1e-5
        self.sv_alphas = self.alphas[sv_mask]
        self.sv_X      = X[sv_mask]
        self.sv_y      = y[sv_mask]

        # Compute bias from support vectors on margin (0 < α < C)
        margin_mask = (self.alphas > 1e-5) & (self.alphas < self.C - 1e-5)
        if margin_mask.sum() > 0:
            K_sv = self._compute_gram_matrix(self.sv_X, X[margin_mask])
            self.b = np.mean(y[margin_mask] -
                             (self.sv_alphas * self.sv_y) @ K_sv)
        else:
            self.b = 0.0

        print(f"Number of support vectors: {sv_mask.sum()} / {m}")
        return self

    def decision_function(self, X):
        K = self._compute_gram_matrix(self.sv_X, X)
        return (self.sv_alphas * self.sv_y) @ K + self.b

    def predict(self, X):
        return np.sign(self.decision_function(X)).astype(int)
```

---

## 14. Scikit-Learn Implementation

### 14.1 SVC — Full Pipeline

```python
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (train_test_split, GridSearchCV,
                                      StratifiedKFold, cross_val_score)
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report
import numpy as np

# ── Load data ─────────────────────────────────────────────────────
data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ── Pipeline ──────────────────────────────────────────────────────
# ALWAYS scale before SVM — it's not scale-invariant!
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42))
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print(classification_report(y_test, y_pred, target_names=data.target_names))
print(f"Test Accuracy: {pipe.score(X_test, y_test):.4f}")

# ── Cross-validation ──────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy')
print(f"\nCV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
```

### 14.2 GridSearchCV — C and gamma

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import loguniform

# Grid search over C and gamma
param_grid = {
    'svc__C':     [0.01, 0.1, 1, 10, 100, 1000],
    'svc__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'svc__kernel': ['rbf'],
}

gs = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy',
                  n_jobs=-1, verbose=1, refit=True)
gs.fit(X_train, y_train)

print(f"Best params : {gs.best_params_}")
print(f"Best CV Acc : {gs.best_score_:.4f}")
print(f"Test Acc    : {gs.score(X_test, y_test):.4f}")

# ── Randomized Search (larger spaces) ────────────────────────────
param_dist = {
    'svc__C':      loguniform(1e-3, 1e4),
    'svc__gamma':  loguniform(1e-5, 1e1),
    'svc__kernel': ['rbf', 'poly', 'linear'],
}

rs = RandomizedSearchCV(pipe, param_dist, n_iter=100, cv=5,
                        scoring='accuracy', n_jobs=-1, random_state=42)
rs.fit(X_train, y_train)
print(f"\nRandom Best params: {rs.best_params_}")
```

### 14.3 LinearSVC (Large Datasets)

```python
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

# LinearSVC: faster for large datasets (uses liblinear, not libsvm)
# Does NOT support kernel tricks — only linear boundary
pipe_linear = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', LinearSVC(C=1.0, max_iter=5000, random_state=42,
                      dual=False))  # dual=False when n_samples > n_features
])

pipe_linear.fit(X_train, y_train)
print(f"LinearSVC Accuracy: {pipe_linear.score(X_test, y_test):.4f}")

# Get probabilities from LinearSVC (CalibratedClassifierCV)
calibrated = CalibratedClassifierCV(pipe_linear, cv=5)
calibrated.fit(X_train, y_train)
y_proba = calibrated.predict_proba(X_test)[:, 1]
```

### 14.4 Non-linear SVM with Polynomial Kernel

```python
from sklearn.svm import SVC
from sklearn.datasets import make_moons

X_moons, y_moons = make_moons(n_samples=500, noise=0.15, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X_moons, y_moons, test_size=0.2)

# Polynomial kernel
pipe_poly = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(kernel='poly', degree=3, C=5, coef0=1, gamma='scale'))
])

# RBF kernel
pipe_rbf = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(kernel='rbf', C=5, gamma=2.0))
])

for name, p in [('Polynomial', pipe_poly), ('RBF', pipe_rbf)]:
    p.fit(X_tr, y_tr)
    print(f"{name}: Test Acc = {p.score(X_te, y_te):.4f}")
```

### 14.5 SVR (Regression)

```python
from sklearn.svm import SVR
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
import numpy as np

housing = fetch_california_housing()
X_h, y_h = housing.data[:5000], housing.target[:5000]  # subset for speed

X_tr, X_te, y_tr, y_te = train_test_split(X_h, y_h, test_size=0.2)

svr_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svr', SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1))
])

svr_pipe.fit(X_tr, y_tr)
y_pred_svr = svr_pipe.predict(X_te)
rmse = np.sqrt(mean_squared_error(y_te, y_pred_svr))
r2   = svr_pipe.score(X_te, y_te)
print(f"SVR RMSE: {rmse:.4f} | R²: {r2:.4f}")

# SVR hyperparameter search
param_grid_svr = {
    'svr__C':       [1, 10, 100, 1000],
    'svr__gamma':   ['scale', 0.01, 0.1, 1],
    'svr__epsilon': [0.01, 0.1, 0.5],
}

gs_svr = GridSearchCV(svr_pipe, param_grid_svr, cv=5,
                      scoring='neg_root_mean_squared_error', n_jobs=-1)
gs_svr.fit(X_tr, y_tr)
print(f"Best SVR params: {gs_svr.best_params_}")
```

### 14.6 One-Class SVM (Anomaly Detection)

```python
from sklearn.svm import OneClassSVM

pipe_oc = Pipeline([
    ('scaler', StandardScaler()),
    ('oc_svm', OneClassSVM(kernel='rbf', gamma='auto', nu=0.05))
])
# nu: upper bound on fraction of outliers (0 < nu < 1)

pipe_oc.fit(X_train)  # Only normal data during training
predictions = pipe_oc.predict(X_test)
# Returns +1 (inlier) or -1 (outlier/anomaly)
```

---

## 15. Hyperparameters & Tuning

### SVC — Complete Reference

| Parameter | Default | Description | Range / Options |
|-----------|---------|-------------|----------------|
| `C` | `1.0` | Regularization — penalty for margin violations. Large C = harder margin, may overfit | `1e-3` to `1e4` (log scale) |
| `kernel` | `'rbf'` | Kernel function | `'linear'`, `'poly'`, `'rbf'`, `'sigmoid'`, `'precomputed'` |
| `gamma` | `'scale'` | RBF/poly/sigmoid kernel coefficient. Controls influence radius of each training example | `'scale'`=$\frac{1}{n \cdot \text{Var}(X)}$, `'auto'`=$\frac{1}{n}$, or float |
| `degree` | `3` | Degree for polynomial kernel only | `2`, `3`, `4`, `5` |
| `coef0` | `0.0` | Independent term in poly and sigmoid kernels | `0.0` to `10.0` |
| `probability` | `False` | Enable probability estimates (uses Platt scaling, slower) | `True`/`False` |
| `shrinking` | `True` | Heuristic to speed up training | `True`/`False` |
| `tol` | `1e-3` | Stopping criterion tolerance | `1e-5` to `1e-2` |
| `cache_size` | `200` | Size of kernel cache in MB | `200` to `2000` (increase for large data) |
| `class_weight` | `None` | Class weights for imbalanced data | `'balanced'` or dict |
| `max_iter` | `-1` | Max iterations (-1 = no limit) | Set if solver hangs |
| `decision_function_shape` | `'ovr'` | Multiclass strategy | `'ovr'`, `'ovo'` |
| `break_ties` | `False` | Break ties by decision function in OvR | `True` for deterministic |
| `random_state` | `None` | For probability estimates and shuffling | Any integer |

### LinearSVC — Additional Parameters

| Parameter | Default | Description | Options |
|-----------|---------|-------------|---------|
| `penalty` | `'l2'` | Norm for regularization | `'l1'`, `'l2'` |
| `loss` | `'squared_hinge'` | Loss function | `'hinge'`, `'squared_hinge'` |
| `dual` | `True` | Dual or primal formulation | `True` if n_samples < n_features, else `False` |
| `multi_class` | `'ovr'` | Multiclass strategy | `'ovr'`, `'crammer_singer'` |
| `intercept_scaling` | `1` | Scaling of bias feature | Float |
| `max_iter` | `1000` | Max iterations | Increase if ConvergenceWarning |

### SVR — Parameters

| Parameter | Default | Description | Range |
|-----------|---------|-------------|-------|
| `kernel` | `'rbf'` | Kernel function | Same as SVC |
| `C` | `1.0` | Regularization | `1e-3` to `1e4` |
| `epsilon` | `0.1` | Width of ε-insensitive tube | `0.01` to `1.0` |
| `gamma` | `'scale'` | Kernel coefficient | Same as SVC |

### NuSVC / NuSVR — Parameters

| Parameter | Default | Description | Range |
|-----------|---------|-------------|-------|
| `nu` | `0.5` | Upper bound on fraction of margin errors AND lower bound on fraction of support vectors | `(0, 1]` |

### Tuning Strategy

```python
# ── Step 1: Start with RBF kernel ────────────────────────────────
# ── Step 2: Grid search C and gamma on log scale ─────────────────
C_range     = np.logspace(-3, 4, 15)
gamma_range = np.logspace(-5, 2, 15)

param_grid = {
    'svc__C':     C_range,
    'svc__gamma': gamma_range,
}

# ── Step 3: Visualize C-gamma heatmap ────────────────────────────
gs = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
gs.fit(X_train, y_train)

scores = gs.cv_results_['mean_test_score'].reshape(len(C_range), len(gamma_range))

plt.figure(figsize=(8, 6))
plt.imshow(scores, interpolation='nearest', cmap='viridis',
           aspect='auto', origin='lower')
plt.colorbar(label='CV Accuracy')
plt.xticks(range(len(gamma_range)), [f'{g:.0e}' for g in gamma_range], rotation=45)
plt.yticks(range(len(C_range)), [f'{c:.0e}' for c in C_range])
plt.xlabel('gamma'); plt.ylabel('C')
plt.title('SVM C-gamma Heatmap')
plt.tight_layout()
plt.show()

print(f"Best C: {gs.best_params_['svc__C']}")
print(f"Best gamma: {gs.best_params_['svc__gamma']}")
```

---

## 16. Bias-Variance & Learning Curves

```python
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

def plot_svm_learning_curve(C_values, X, y):
    fig, axes = plt.subplots(1, len(C_values), figsize=(5 * len(C_values), 4))

    for ax, C in zip(axes, C_values):
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(kernel='rbf', C=C, gamma='scale'))
        ])

        train_sizes, train_scores, val_scores = learning_curve(
            pipe, X, y,
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=StratifiedKFold(5, shuffle=True, random_state=42),
            scoring='accuracy', n_jobs=-1
        )

        ax.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='Train')
        ax.plot(train_sizes, val_scores.mean(axis=1),   'o-', label='Val')
        ax.fill_between(train_sizes,
                        train_scores.mean(1) - train_scores.std(1),
                        train_scores.mean(1) + train_scores.std(1), alpha=0.1)
        ax.fill_between(train_sizes,
                        val_scores.mean(1) - val_scores.std(1),
                        val_scores.mean(1) + val_scores.std(1), alpha=0.1)
        ax.set_title(f"C={C}"); ax.legend(); ax.grid(True)
        ax.set_ylim(0.5, 1.05)

    plt.suptitle("SVM Learning Curves for Different C Values")
    plt.tight_layout()
    plt.show()

plot_svm_learning_curve([0.01, 1, 100], X, y)
```

### Diagnosis Table

| Symptom | Problem | Fix |
|---------|---------|-----|
| Low train acc, low test acc | High Bias | Decrease C, try RBF kernel, add features |
| High train acc, low test acc | High Variance | Increase C, decrease gamma, more data |
| Slow convergence | Training issue | Scale features, increase cache_size |
| Few support vectors | Very wide margin | May be underfitting, try higher C |
| Many support vectors (≈ m) | Overfitting or hard problem | Reduce C, try different kernel |

---

## 17. SVM vs Logistic Regression

| Aspect | SVM | Logistic Regression |
|--------|-----|-------------------|
| **Decision Boundary** | Maximum margin hyperplane | Log-odds hyperplane |
| **Loss Function** | Hinge loss | Log loss (cross-entropy) |
| **Probabilities** | No (needs Platt scaling) | Yes, native |
| **Outlier Sensitivity** | Robust (only SVs matter) | Sensitive (all points) |
| **High Dimensions** | Excellent ($n \gg m$ ok) | Good |
| **Large Datasets** | Slow ($O(m^2)$–$O(m^3)$) | Fast |
| **Non-linear** | Kernel trick | Manual feature engineering |
| **Regularization** | Large C = weak reg | Large C = strong reg (⚠ opposite!) |
| **Interpretability** | Support vectors | Coefficients |
| **Multiclass** | OvO or OvR | Native multinomial |

> **Rule of thumb:** For large datasets → Logistic Regression or LinearSVC. For complex non-linear boundaries + small-medium data → kernel SVM.

---

## 18. Common Interview Questions

**Q: What is a Support Vector Machine? What are support vectors?**
> SVM finds the hyperplane that maximizes the margin between two classes. Support vectors are the training points that lie on or inside the margin boundaries — the points closest to the decision boundary. They are the only points that define the solution; removing any other point wouldn't change the model.

**Q: What is the kernel trick and why is it powerful?**
> The kernel trick computes dot products in a high-dimensional feature space without explicitly constructing the transformation. Since the SVM dual problem only involves dot products, we substitute $K(\mathbf{x}^{(i)}, \mathbf{x}^{(j)})$ for $\phi(\mathbf{x}^{(i)})^T\phi(\mathbf{x}^{(j)})$. This allows SVMs to learn non-linear boundaries at the cost of the original feature space computation.

**Q: What does the C parameter do in SVM?**
> C controls the trade-off between maximizing the margin and minimizing training errors. Large C → narrow margin, fewer violations, harder boundary (may overfit). Small C → wide margin, more violations allowed (may underfit). Note: unlike logistic regression, larger C means LESS regularization.

**Q: What is the difference between hard margin and soft margin SVM?**
> Hard margin SVM requires all points to be correctly classified outside the margin — only works for perfectly linearly separable data. Soft margin SVM introduces slack variables that allow some points to be inside or on the wrong side of the margin, penalized by the C parameter. Soft margin is used in practice.

**Q: Why do we need to scale features for SVM?**
> SVM is based on distances (via the dot product or RBF kernel). If one feature has range [0, 10000] and another [0, 1], the first will dominate the distance computation, making the kernel useless for the second feature. StandardScaler ensures all features contribute equally.

**Q: What does gamma control in the RBF kernel?**
> Gamma defines how far the influence of a single training example reaches. High gamma: each point only influences very nearby predictions → jagged, complex boundary (overfitting). Low gamma: each point influences a large area → smooth, simple boundary (may underfit).

**Q: How does SVM handle multiclass problems?**
> SVM natively handles binary classification only. sklearn's SVC uses One-vs-One (OvO) by default — trains $K(K-1)/2$ binary classifiers and uses majority voting. LinearSVC uses One-vs-Rest (OvR). You can also explicitly use `OneVsRestClassifier` or `OneVsOneClassifier` wrappers.

**Q: Can SVM output probabilities?**
> Not directly. Setting `probability=True` in sklearn uses **Platt scaling** — a sigmoid function is fitted on top of the SVM decision scores using cross-validation. This is slower and probabilities may not be well-calibrated.

**Q: What is the complexity of SVM training?**
> Training a kernel SVM is $O(m^2 n)$ to $O(m^3)$ where $m$ = samples, $n$ = features. This makes it slow for $m > 100{,}000$. LinearSVC (using liblinear) is much faster at $O(m \cdot n)$.

**Q: What is the hinge loss and how does it compare to log loss?**
> Hinge loss: $\max(0, 1 - y f(x))$ — zero for correctly classified points outside the margin, linear penalty otherwise. Log loss decreases toward zero even for correctly classified points (never exactly zero for logistic regression). Hinge loss is sparse (most examples don't contribute to gradient), while log loss uses all examples.

**Q: What is a kernel matrix (Gram matrix)?**
> A kernel matrix $K$ of size $m \times m$ where $K_{ij} = K(\mathbf{x}^{(i)}, \mathbf{x}^{(j)})$ for all pairs of training examples. It must be symmetric positive semi-definite (Mercer's condition) to be a valid kernel. SVM training in the dual form operates on this matrix.

**Q: What are the pros and cons of SVMs?**
> Pros: Effective in high dimensions, works when n > m, memory efficient (only SVs stored), powerful with kernels, robust to outliers. Cons: Slow on large datasets, no native probability output, sensitive to feature scaling, hard to interpret, kernel/hyperparameter selection is crucial.

---

## 19. Resources

### 📺 Andrew Ng — ML Specialization (Coursera)
- **Course 3 (Advanced Learning Algorithms):** Not covered in depth in the new specialization
- **CS229 (Stanford, free):** Lecture 6–7: SVMs, kernels, optimization
- [CS229 SVM Notes (PDF)](https://cs229.stanford.edu/lectures-spring2022/main_notes.pdf) — Chapter on SVMs is exceptional

### 📺 StatQuest with Josh Starmer (YouTube)
- [Support Vector Machines, Clearly Explained!!!](https://www.youtube.com/watch?v=efR1C6CvhmE)
- [The Polynomial Kernel (SVM)](https://www.youtube.com/watch?v=Toet3EiSFcM)
- [The Radial Basis Function (RBF) Kernel](https://www.youtube.com/watch?v=Qc5IyLW_hns)
- [Support Vector Regression](https://www.youtube.com/watch?v=Yz17HCRrW8w)

### 📖 Hands-On Machine Learning (Géron) — O'Reilly
- **Chapter 5:** Support Vector Machines — full chapter covering linear SVM, soft margin, non-linear SVM, SVR, online learning
- GitHub notebooks: [https://github.com/ageron/handson-ml3](https://github.com/ageron/handson-ml3)

### 📖 Introduction to Statistical Learning with Python (ISLP)
- **Chapter 9:** Support Vector Machines — SVM theory, kernels, extensions
- Free PDF + Python labs: [https://www.statlearning.com/](https://www.statlearning.com/)

### 📚 Scikit-Learn Documentation
- [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- [LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)
- [SVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)
- [SVM User Guide](https://scikit-learn.org/stable/modules/svm.html)
- [Kernel Functions](https://scikit-learn.org/stable/modules/svm.html#kernel-functions)

### 🔗 Additional
- [CS229 SVM Notes — Andrew Ng (Stanford)](https://cs229.stanford.edu/lectures-spring2022/main_notes.pdf)
- [libsvm — original SVM library behind sklearn](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)
- [A Practical Guide to SVM Classification (Hsu et al.)](https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf) — essential practical guide

---

## 🗺️ Quick Reference Cheatsheet

```
SVM WORKFLOW
══════════════════════════════════════════════════════════════

1. EXPLORE DATA
   ├── Check n_samples (< 100k for kernel SVM)
   ├── Check n_features
   ├── Class balance
   └── Feature scales

2. PREPROCESS (MANDATORY for SVM)
   ├── StandardScaler().fit_transform() — REQUIRED
   ├── Handle missing values (SVM has none natively)
   ├── Encode categoricals
   └── Stratified split

3. CHOOSE KERNEL
   ├── Linear data / large n → LinearSVC or SVC(kernel='linear')
   ├── Non-linear / unknown → SVC(kernel='rbf') ← start here
   ├── Polynomial features → SVC(kernel='poly')
   └── Very large data → LinearSVC + SGDClassifier(loss='hinge')

4. TUNE HYPERPARAMETERS
   ├── GridSearchCV on C and gamma (log scale!)
   ├── C: [0.01, 0.1, 1, 10, 100, 1000]
   └── gamma: ['scale', 0.001, 0.01, 0.1, 1]

5. EVALUATE
   ├── Accuracy, F1, ROC-AUC
   ├── decision_function for ROC (not predict_proba)
   └── Check n_support_ (too many SVs → overfitting)

6. DIAGNOSE
   ├── High bias: decrease C, try RBF, add features
   ├── High variance: increase C, decrease gamma, more data
   └── Very slow: reduce dataset, use LinearSVC

KEY FORMULAS
══════════════════════════════════════════════════════════════
Hard margin primal:  min ½||w||²  s.t. yᵢ(wᵀxᵢ+b) ≥ 1
Soft margin primal:  min ½||w||² + C·Σξᵢ  s.t. yᵢ(wᵀxᵢ+b) ≥ 1-ξᵢ
Margin width:        2 / ||w||
Hinge loss:          max(0, 1 - y·f(x))
RBF kernel:          K(xᵢ,xⱼ) = exp(-γ||xᵢ-xⱼ||²)
Poly kernel:         K(xᵢ,xⱼ) = (γxᵢᵀxⱼ + r)^d
Prediction (dual):   f(x) = sign(Σ αᵢyᵢK(xᵢ,x) + b)
SVR ε-loss:          max(0, |y-ŷ| - ε)

C vs REGULARIZATION (CRITICAL!)
══════════════════════════════════════════════════════════════
SVM:  Large C → LESS regularization (harder margin, overfit risk)
LR:   Large C → LESS regularization (same intuition)
Ridge/Lasso: Large λ → MORE regularization (opposite!)

KERNEL SELECTION GUIDE
══════════════════════════════════════════════════════════════
Linear  → Linearly separable OR n_features >> n_samples (text)
RBF     → General purpose, unknown structure ← DEFAULT CHOICE
Poly    → Image processing, structured data
Sigmoid → Neural-net-like behaviour (rarely used)
```

---

*Notes compiled for ML/DL job readiness. Sources: Andrew Ng CS229 (Stanford), Hands-On ML Ch. 5 (Géron), StatQuest (Starmer), ISLP Ch. 9 (James et al.), Scikit-Learn documentation, libsvm practical guide (Hsu et al.).*
