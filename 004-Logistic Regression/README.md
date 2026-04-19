# 🔵 Logistic Regression — Complete ML/DL Notes

> **Topic:** Supervised Learning | **Type:** Classification | **Level:** Beginner → Advanced

---

## 📚 Table of Contents

1. [What is Logistic Regression?](#1-what-is-logistic-regression)
2. [Why Not Linear Regression for Classification?](#2-why-not-linear-regression-for-classification)
3. [The Sigmoid Function](#3-the-sigmoid-function)
4. [Types of Logistic Regression](#4-types-of-logistic-regression)
5. [The Math Behind It](#5-the-math-behind-it)
6. [Cost Function — Binary Cross-Entropy](#6-cost-function--binary-cross-entropy)
7. [Gradient Descent](#7-gradient-descent)
8. [Decision Boundary](#8-decision-boundary)
9. [Assumptions](#9-assumptions)
10. [Evaluation Metrics](#10-evaluation-metrics)
11. [Hyperparameters](#11-hyperparameters)
12. [Regularization](#12-regularization)
13. [Multiclass Classification](#13-multiclass-classification)
14. [Complete Python Code (From Scratch)](#14-complete-python-code-from-scratch)
15. [Sklearn Implementation](#15-sklearn-implementation)
16. [Common Mistakes & Tips](#16-common-mistakes--tips)
17. [Interview Questions](#17-interview-questions)
18. [Resources](#18-resources)

---

## 1. What is Logistic Regression?

Despite the name, **Logistic Regression is a Classification algorithm**, not regression.

It predicts the **probability** that an input belongs to a class:

```
P(y = 1 | X) = ?    → probability of being in class 1
```

### Simple Analogy
> Spam filter: given email features (words, sender, links), what is the probability this email is spam? If P > 0.5 → spam, else → not spam.

### Real World Use Cases
- Email spam detection (spam / not spam)
- Disease diagnosis (has disease / doesn't have)
- Credit card fraud detection (fraud / not fraud)
- Customer churn prediction (will churn / won't churn)
- Sentiment analysis (positive / negative)

### Key Point
> Logistic Regression outputs a **probability** between 0 and 1, then converts it to a class label using a **threshold** (default = 0.5).

---

## 2. Why Not Linear Regression for Classification?

Linear Regression predicts continuous values like `3.7` or `-1.2`.

For classification we need values **between 0 and 1** (probabilities).

### Problems with Linear Regression for Classification

```
Linear Regression output: ŷ = wX + b
```

- Output can be > 1 or < 0 → not a valid probability
- Sensitive to outliers (one far point shifts the whole line)
- Assumes linear relationship between features and output (not probability)

```
Example:
  y = 0 (no disease)  or  y = 1 (has disease)

  Linear regression might predict: ŷ = 1.8 or ŷ = -0.3
  → Meaningless as probability!

  Logistic regression always predicts: 0 ≤ ŷ ≤ 1  ✅
```

---

## 3. The Sigmoid Function

The **sigmoid (logistic) function** squashes any real number into the range **(0, 1)**.

### Formula

```
σ(z) = 1 / (1 + e⁻ᶻ)
```

Where:
- `z` = linear combination of features = `wᵀx + b`
- `e` = Euler's number ≈ 2.718
- Output is always between 0 and 1

### Behavior

```
z → -∞   →   σ(z) → 0
z =  0   →   σ(z) = 0.5
z → +∞   →   σ(z) → 1
```

### Derivative of Sigmoid (needed for backprop)

```
σ'(z) = σ(z) · (1 - σ(z))
```

This is elegant — you compute sigmoid once and reuse it for the derivative!

### Python

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Visualize
z = np.linspace(-10, 10, 200)
plt.figure(figsize=(8, 4))
plt.plot(z, sigmoid(z), color='blue', linewidth=2)
plt.axhline(0.5, color='red', linestyle='--', label='threshold = 0.5')
plt.axvline(0, color='gray', linestyle='--')
plt.title("Sigmoid Function σ(z) = 1 / (1 + e⁻ᶻ)")
plt.xlabel("z"); plt.ylabel("σ(z)")
plt.legend(); plt.grid(True)
plt.show()
```

---

## 4. Types of Logistic Regression

| Type | Output Classes | Method |
|---|---|---|
| **Binary** | 2 classes (0 or 1) | Sigmoid activation |
| **Multinomial** | 3+ classes (no order) | Softmax activation |
| **Ordinal** | 3+ classes (ordered: low/med/high) | Multiple thresholds |

---

## 5. The Math Behind It

### Step 1 — Linear Combination

```
z = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ
z = wᵀx + b       (vector form)
z = Xw + b        (matrix form)
```

### Step 2 — Apply Sigmoid

```
ŷ = σ(z) = 1 / (1 + e⁻ᶻ)
```

This gives us **P(y = 1 | x; w)** — probability of positive class given features x.

### Step 3 — Decision

```
if ŷ ≥ 0.5  →  predict y = 1   (positive class)
if ŷ < 0.5  →  predict y = 0   (negative class)
```

Note: `ŷ ≥ 0.5` is equivalent to `z ≥ 0` since σ(0) = 0.5

### Full Forward Pass (Matrix Form)

```
Z = Xw + b          shape: (m, 1)
A = σ(Z)            shape: (m, 1)   ← predicted probabilities
```

Where:
- `X` = feature matrix, shape `(m, n)` — m samples, n features
- `w` = weight vector, shape `(n, 1)`
- `b` = bias scalar
- `A` = predicted probabilities

---

## 6. Cost Function — Binary Cross-Entropy

### Why Not MSE?

If we use MSE with sigmoid, the cost surface becomes **non-convex** (many local minima) → gradient descent can get stuck.

Cross-Entropy gives a **convex** cost surface → one global minimum → gradient descent works!

### Log Loss (Binary Cross-Entropy)

For a single sample:
```
L(ŷ, y) = -[y · log(ŷ) + (1 - y) · log(1 - ŷ)]
```

For the full dataset (m samples):
```
J(w, b) = -(1/m) · Σᵢ [yᵢ · log(ŷᵢ) + (1 - yᵢ) · log(1 - ŷᵢ)]
```

### Why This Formula Works

When **y = 1** (positive):
```
L = -log(ŷ)
  → if ŷ → 1 (correct):   L → 0  ✅  (no penalty)
  → if ŷ → 0 (wrong):     L → ∞  ❌  (heavy penalty)
```

When **y = 0** (negative):
```
L = -log(1 - ŷ)
  → if ŷ → 0 (correct):   L → 0  ✅
  → if ŷ → 1 (wrong):     L → ∞  ❌
```

### Matrix Form

```
J(w, b) = -(1/m) · [yᵀ log(A) + (1 - y)ᵀ log(1 - A)]
```

Where `log` is applied element-wise.

### Probabilistic Interpretation (MLE)

Cross-entropy loss comes from **Maximum Likelihood Estimation (MLE)**.

The likelihood for Bernoulli distribution:
```
P(y | x; w) = ŷʸ · (1 - ŷ)¹⁻ʸ
```

Taking the negative log-likelihood of all m samples:
```
-log L = -(1/m) Σ [y log(ŷ) + (1-y) log(1-ŷ)]
```

This is exactly our cost function! We **minimize** the negative log-likelihood.

---

## 7. Gradient Descent

### Compute Gradients

Taking the partial derivatives of `J` with respect to `w` and `b`:

**Gradient for weights:**
```
∂J/∂w = (1/m) · Xᵀ · (A - y)
```

**Gradient for bias:**
```
∂J/∂b = (1/m) · Σ (A - y)
       = (1/m) · np.sum(A - y)
```

> Notice: the gradient formula looks identical to linear regression! The difference is that `A = σ(Xw + b)` instead of `A = Xw + b`.

### Update Rule

```
w := w - α · ∂J/∂w
b := b - α · ∂J/∂b
```

Where `α` = learning rate.

### Full Gradient Descent Step-by-Step

```
For each iteration:
  1. Z = Xw + b                        → linear combination
  2. A = σ(Z)                          → probabilities (forward pass)
  3. J = -(1/m)[yᵀlog(A)+(1-y)ᵀlog(1-A)]  → compute cost
  4. dw = (1/m) · Xᵀ(A - y)           → gradient for weights
  5. db = (1/m) · Σ(A - y)            → gradient for bias
  6. w  := w - α · dw                  → update weights
  7. b  := b - α · db                  → update bias
```

### Types of Gradient Descent

| Type | Samples/Update | Pros | Cons |
|---|---|---|---|
| **Batch GD** | All m | Stable, exact gradient | Slow on large data |
| **Stochastic GD (SGD)** | 1 sample | Fast, escapes local minima | Very noisy |
| **Mini-Batch GD** | k samples (32, 64, 128) | Best of both | Need to choose batch size |

---

## 8. Decision Boundary

The decision boundary is where `ŷ = 0.5`, which means `z = 0`.

```
z = wᵀx + b = 0
```

### Linear Boundary (standard)

```
w₁x₁ + w₂x₂ + b = 0   → a straight line in 2D
```

### Non-Linear Boundary (with polynomial features)

```
w₁x₁ + w₂x₂ + w₃x₁² + w₄x₂² + w₅x₁x₂ + b = 0
```

With higher-degree polynomial features, the decision boundary can be circles, ellipses, or complex curves.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=200, n_features=2,
                            n_redundant=0, random_state=42)

model = LogisticRegression()
model.fit(X, y)

# Plot decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(8, 5))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
plt.title("Logistic Regression Decision Boundary")
plt.xlabel("Feature 1"); plt.ylabel("Feature 2")
plt.show()
```

---

## 9. Assumptions

| # | Assumption | Details |
|---|---|---|
| 1 | **Binary outcome** | Target y ∈ {0, 1} for binary LR |
| 2 | **Linearity of log-odds** | Log(p/1-p) is linear in features |
| 3 | **No multicollinearity** | Features not highly correlated |
| 4 | **Independence** | Observations are independent |
| 5 | **Large sample size** | At least 10 events per feature (EPV rule) |
| 6 | **No extreme outliers** | Outliers distort decision boundary |

### Log-Odds (Logit)

The **logit** is the log of the odds ratio:
```
logit(p) = log(p / (1-p)) = wᵀx + b
```

This is why "linearity" is assumed in **log-odds space**, not in probability space.

---

## 10. Evaluation Metrics

### 10.1 Confusion Matrix

```
                Predicted 0    Predicted 1
Actual 0    |     TN        |     FP       |
Actual 1    |     FN        |     TP       |
```

- **TP** = True Positive  (said 1, was 1) ✅
- **TN** = True Negative  (said 0, was 0) ✅
- **FP** = False Positive (said 1, was 0) ❌ — Type I Error
- **FN** = False Negative (said 0, was 1) ❌ — Type II Error

### 10.2 Key Metrics

**Accuracy**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
> ⚠️ Misleading on imbalanced datasets! (99% negatives → 99% accuracy by always predicting 0)

**Precision** (of all predicted positives, how many are actually positive)
```
Precision = TP / (TP + FP)
```

**Recall / Sensitivity** (of all actual positives, how many did we catch)
```
Recall = TP / (TP + FN)
```

**F1 Score** (harmonic mean of precision and recall)
```
F1 = 2 · (Precision · Recall) / (Precision + Recall)
```

**Specificity** (of all actual negatives, how many did we correctly identify)
```
Specificity = TN / (TN + FP)
```

### Precision-Recall Trade-off

- **High threshold (e.g., 0.9)** → high precision, low recall → conservative (rarely says positive)
- **Low threshold (e.g., 0.3)** → low precision, high recall → aggressive (says positive often)

```
Medical diagnosis: prefer HIGH recall (catch all diseases, even false alarms)
Spam filter: prefer HIGH precision (don't want real emails in spam)
```

### 10.3 ROC Curve & AUC

**ROC Curve** plots:
- X-axis: False Positive Rate = FP / (FP + TN) = 1 - Specificity
- Y-axis: True Positive Rate = Recall = TP / (TP + FN)

**AUC (Area Under Curve)**:
- AUC = 1.0 → perfect model
- AUC = 0.5 → random guessing (diagonal line)
- AUC = 0.0 → perfectly wrong

```python
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {auc_score:.3f}')
plt.plot([0, 1], [0, 1], 'r--', label='Random')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve")
plt.legend(); plt.grid(True)
plt.show()
```

### 10.4 Log Loss (Cross-Entropy)

```
Log Loss = -(1/m) Σ [yᵢ log(ŷᵢ) + (1 - yᵢ) log(1 - ŷᵢ)]
```

- Lower is better
- Directly measures quality of probability predictions
- Used in competitions (Kaggle)

```python
from sklearn.metrics import log_loss
print(f"Log Loss: {log_loss(y_test, y_prob):.4f}")
```

---

## 11. Hyperparameters

These are settings you configure **before training**.

### 11.1 `C` — Inverse Regularization Strength

```
C = 1 / λ
```

- `C` is the **inverse** of regularization strength
- **Large C** → weak regularization → model fits training data closely → may overfit
- **Small C** → strong regularization → model is simpler → may underfit
- Default: `C = 1.0`

```python
from sklearn.linear_model import LogisticRegression

# Weak regularization (complex model)
model_high_c = LogisticRegression(C=100)

# Strong regularization (simple model)
model_low_c = LogisticRegression(C=0.01)
```

### 11.2 `penalty` — Type of Regularization

```python
LogisticRegression(penalty='l2')                          # default — Ridge
LogisticRegression(penalty='l1')                          # Lasso — feature selection
LogisticRegression(penalty='elasticnet', l1_ratio=0.5)    # ElasticNet
LogisticRegression(penalty=None)                          # no regularization
```

| Penalty | Formula Added | Effect |
|---|---|---|
| `'l2'` (default) | `λ Σ wⱼ²` | Shrinks weights, no zeros |
| `'l1'` | `λ Σ\|wⱼ\|` | Can zero out weights → feature selection |
| `'elasticnet'` | L1 + L2 | Combined |
| `None` | — | No regularization |

### 11.3 `solver` — Optimization Algorithm

| Solver | Supports L1? | Supports L2? | Best For |
|---|---|---|---|
| `'lbfgs'` (default) | ❌ | ✅ | Small/medium datasets |
| `'liblinear'` | ✅ | ✅ | Small datasets, binary |
| `'saga'` | ✅ | ✅ | Large datasets |
| `'sag'` | ❌ | ✅ | Large datasets |
| `'newton-cg'` | ❌ | ✅ | Multiclass |

```python
LogisticRegression(solver='lbfgs')        # default
LogisticRegression(solver='liblinear')    # good for small data + L1
LogisticRegression(solver='saga')         # large data + L1
```

### 11.4 `max_iter` — Maximum Iterations

Number of iterations for the solver to converge.

```python
LogisticRegression(max_iter=100)    # default (often not enough!)
LogisticRegression(max_iter=1000)   # increase if ConvergenceWarning appears
```

> ⚠️ Always increase `max_iter` if you see **ConvergenceWarning** in sklearn.

### 11.5 `tol` — Tolerance for Stopping

Stop training when improvement is less than this value.

```python
LogisticRegression(tol=1e-4)    # default
LogisticRegression(tol=1e-6)    # more precise convergence
```

### 11.6 `multi_class` — Multiclass Strategy

```python
LogisticRegression(multi_class='auto')         # default
LogisticRegression(multi_class='ovr')          # One-vs-Rest
LogisticRegression(multi_class='multinomial')  # Softmax
```

### 11.7 `class_weight` — Handle Imbalanced Data

```python
LogisticRegression(class_weight='balanced')    # auto-weights by class frequency
LogisticRegression(class_weight={0: 1, 1: 5}) # manual: class 1 gets 5x weight
```

### 11.8 `threshold` — Decision Threshold

Not a model parameter but set during prediction:

```python
y_prob = model.predict_proba(X_test)[:, 1]
threshold = 0.3   # lower = more sensitive = higher recall
y_pred_custom = (y_prob >= threshold).astype(int)
```

### 11.9 `l1_ratio` — ElasticNet Mix

Only used when `penalty='elasticnet'`:

```python
LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5)
# 0 = pure L2 (Ridge), 1 = pure L1 (Lasso), 0.5 = 50/50
```

### 11.10 `random_state`

For reproducibility when solver uses randomness (e.g., saga):

```python
LogisticRegression(random_state=42)
```

### Complete Hyperparameter Summary Table

| Hyperparameter | Default | Range / Options | Controls |
|---|---|---|---|
| `C` | 1.0 | 0.001 – 1000 | Regularization strength (inverse) |
| `penalty` | `'l2'` | `'l1','l2','elasticnet',None` | Regularization type |
| `solver` | `'lbfgs'` | see table above | Optimization algorithm |
| `max_iter` | 100 | 100 – 10,000 | Max iterations to converge |
| `tol` | 1e-4 | 1e-6 – 1e-3 | Stopping tolerance |
| `multi_class` | `'auto'` | `'ovr','multinomial'` | Multiclass strategy |
| `class_weight` | None | `'balanced'`, dict | Handle class imbalance |
| `l1_ratio` | None | 0.0 – 1.0 | ElasticNet L1/L2 mix |
| `fit_intercept` | True | True / False | Include bias term |
| `random_state` | None | integer | Reproducibility |

---

## 12. Regularization

### Without Regularization

```
J(w, b) = -(1/m) Σ [y log(ŷ) + (1-y) log(1-ŷ)]
```

### L2 Regularization (Ridge)

```
J(w, b) = -(1/m) Σ [y log(ŷ) + (1-y) log(1-ŷ)] + (λ/2m) Σ wⱼ²
```

Gradient update with L2:
```
∂J/∂w = (1/m) Xᵀ(A - y) + (λ/m) w
w := w - α · [(1/m) Xᵀ(A - y) + (λ/m) w]
w := w(1 - α·λ/m) - α · (1/m) Xᵀ(A - y)
```

The term `(1 - α·λ/m)` shrinks weights each iteration — this is **weight decay**.

### L1 Regularization (Lasso)

```
J(w, b) = -(1/m) Σ [y log(ŷ) + (1-y) log(1-ŷ)] + (λ/m) Σ |wⱼ|
```

Gradient update with L1 (uses sign):
```
∂J/∂w = (1/m) Xᵀ(A - y) + (λ/m) · sign(w)
```

---

## 13. Multiclass Classification

### One-vs-Rest (OvR) / One-vs-All (OvA)

Train **K separate binary classifiers**, one per class.

```
Classifier 1: class 0 vs rest (1, 2, 3)
Classifier 2: class 1 vs rest (0, 2, 3)
Classifier 3: class 2 vs rest (0, 1, 3)
```

Pick the class with the **highest probability**.

### Softmax Regression (Multinomial)

Uses the **Softmax function** instead of sigmoid:

```
P(y = k | x) = exp(wₖᵀx) / Σⱼ exp(wⱼᵀx)
```

- Outputs a probability distribution over all K classes
- All probabilities sum to 1
- Directly models multiclass (more principled than OvR)

**Cost function — Categorical Cross-Entropy:**
```
J = -(1/m) Σᵢ Σₖ yᵢₖ · log(ŷᵢₖ)
```

Where `yᵢₖ = 1` if sample i belongs to class k (one-hot encoding).

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

# OvR
ovr_model = LogisticRegression(multi_class='ovr', solver='lbfgs', max_iter=1000)
ovr_model.fit(X, y)

# Multinomial (Softmax)
softmax_model = LogisticRegression(multi_class='multinomial',
                                   solver='lbfgs', max_iter=1000)
softmax_model.fit(X, y)

print("OvR Accuracy:        ", ovr_model.score(X, y))
print("Multinomial Accuracy:", softmax_model.score(X, y))
```

---

## 14. Complete Python Code (From Scratch)

### Binary Logistic Regression with Gradient Descent

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
# 1. Generate Data
# ─────────────────────────────────────────────
np.random.seed(42)
X, y = make_classification(n_samples=1000, n_features=2,
                            n_redundant=0, n_informative=2,
                            random_state=42)
y = y.reshape(-1, 1)   # shape (m, 1)

# Train-test split and scaling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

m_train, n = X_train.shape

# ─────────────────────────────────────────────
# 2. Helper Functions
# ─────────────────────────────────────────────
def sigmoid(z):
    """σ(z) = 1 / (1 + e^{-z})"""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))   # clip to avoid overflow

def compute_cost(A, y, w, lambda_reg=0.0):
    """Binary cross-entropy + optional L2 regularization"""
    m = len(y)
    eps = 1e-15   # prevent log(0)
    ce  = -(1/m) * np.sum(y * np.log(A + eps) + (1 - y) * np.log(1 - A + eps))
    l2  = (lambda_reg / (2 * m)) * np.sum(w**2)
    return ce + l2

def compute_gradients(X, A, y, w, lambda_reg=0.0):
    """Compute dw and db with optional L2"""
    m  = len(y)
    err = A - y                                     # shape (m, 1)
    dw  = (1/m) * X.T.dot(err) + (lambda_reg/m)*w  # shape (n, 1)
    db  = (1/m) * np.sum(err)                       # scalar
    return dw, db

def predict(X, w, b, threshold=0.5):
    A = sigmoid(X.dot(w) + b)
    return (A >= threshold).astype(int)

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

# ─────────────────────────────────────────────
# 3. Initialize Parameters
# ─────────────────────────────────────────────
w = np.zeros((n, 1))   # shape (n, 1)
b = 0.0
cost_history = []

# ─────────────────────────────────────────────
# 4. Hyperparameters
# ─────────────────────────────────────────────
learning_rate = 0.1
n_iterations  = 1000
lambda_reg    = 0.01    # L2 regularization

# ─────────────────────────────────────────────
# 5. Training Loop — Batch Gradient Descent
# ─────────────────────────────────────────────
for i in range(n_iterations):
    Z  = X_train.dot(w) + b       # linear: (m,1)
    A  = sigmoid(Z)                # probabilities: (m,1)
    
    cost = compute_cost(A, y_train, w, lambda_reg)
    cost_history.append(cost)
    
    dw, db = compute_gradients(X_train, A, y_train, w, lambda_reg)
    w -= learning_rate * dw
    b -= learning_rate * db
    
    if i % 100 == 0:
        acc = accuracy(y_train, predict(X_train, w, b))
        print(f"Iter {i:4d} | Cost: {cost:.4f} | Train Acc: {acc:.4f}")

# ─────────────────────────────────────────────
# 6. Evaluation
# ─────────────────────────────────────────────
y_pred_test = predict(X_test, w, b)
print(f"\nTest Accuracy: {accuracy(y_test, y_pred_test):.4f}")
print(f"Weights: {w.flatten()}")
print(f"Bias:    {b:.4f}")

# ─────────────────────────────────────────────
# 7. Metrics from Scratch
# ─────────────────────────────────────────────
def metrics_scratch(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    TP = np.sum((y_pred == 1) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    precision   = TP / (TP + FP + 1e-9)
    recall      = TP / (TP + FN + 1e-9)
    f1          = 2 * precision * recall / (precision + recall + 1e-9)
    specificity = TN / (TN + FP + 1e-9)
    return dict(TP=TP, TN=TN, FP=FP, FN=FN,
                precision=precision, recall=recall,
                f1=f1, specificity=specificity)

m_dict = metrics_scratch(y_test, y_pred_test)
for k, v in m_dict.items():
    print(f"{k:12s}: {v:.4f}" if isinstance(v, float) else f"{k:12s}: {v}")

# ─────────────────────────────────────────────
# 8. Plots
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(cost_history, color='blue')
axes[0].set_title("Cost vs Iterations")
axes[0].set_xlabel("Iteration"); axes[0].set_ylabel("Cost")
axes[0].grid(True)

x1_vals = np.linspace(X_test[:, 0].min()-1, X_test[:, 0].max()+1, 200)
x2_vals = -(w[0] * x1_vals + b) / (w[1] + 1e-9)
axes[1].scatter(X_test[:, 0], X_test[:, 1],
                c=y_test.flatten(), cmap='coolwarm', alpha=0.6, edgecolors='k')
axes[1].plot(x1_vals, x2_vals, 'g-', lw=2, label='Decision Boundary')
axes[1].set_title("Decision Boundary"); axes[1].legend(); axes[1].grid(True)

plt.tight_layout(); plt.show()
```

### Mini-Batch Gradient Descent

```python
def mini_batch_gd(X, y, learning_rate=0.1, n_epochs=100, batch_size=32, lambda_reg=0.0):
    m, n = X.shape
    w = np.zeros((n, 1))
    b = 0.0
    cost_history = []

    for epoch in range(n_epochs):
        idx = np.random.permutation(m)
        X_s, y_s = X[idx], y[idx]
        epoch_costs = []

        for i in range(0, m, batch_size):
            Xb = X_s[i:i+batch_size]
            yb = y_s[i:i+batch_size]
            A  = sigmoid(Xb.dot(w) + b)
            cost = compute_cost(A, yb, w, lambda_reg)
            epoch_costs.append(cost)
            dw, db = compute_gradients(Xb, A, yb, w, lambda_reg)
            w -= learning_rate * dw
            b -= learning_rate * db

        cost_history.append(np.mean(epoch_costs))

    return w, b, cost_history

w_mb, b_mb, hist_mb = mini_batch_gd(X_train, y_train,
                                     learning_rate=0.1, n_epochs=200,
                                     batch_size=32, lambda_reg=0.01)
print(f"Mini-Batch Test Acc: {accuracy(y_test, predict(X_test, w_mb, b_mb)):.4f}")
```

---

## 15. Sklearn Implementation

### Full Pipeline with Tuning

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     GridSearchCV, StratifiedKFold,
                                     learning_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_curve, roc_auc_score, precision_recall_curve,
                              log_loss, ConfusionMatrixDisplay, f1_score)
from sklearn.datasets import load_breast_cancer

# ─────────────────────────────────────────────
# 1. Load Dataset
# ─────────────────────────────────────────────
data = load_breast_cancer()
X    = pd.DataFrame(data.data, columns=data.feature_names)
y    = data.target   # 0 = malignant, 1 = benign

print(f"Shape: {X.shape}")
print(f"Class distribution: {np.bincount(y)}")

# ─────────────────────────────────────────────
# 2. Train-Test Split (stratified!)
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─────────────────────────────────────────────
# 3. Build Pipeline
# ─────────────────────────────────────────────
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(
        penalty='l2',
        C=1.0,
        solver='lbfgs',
        max_iter=1000,
        random_state=42
    ))
])

pipeline.fit(X_train, y_train)

# ─────────────────────────────────────────────
# 4. Predictions
# ─────────────────────────────────────────────
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print(f"\nAccuracy: {(y_pred == y_test).mean():.4f}")

# ─────────────────────────────────────────────
# 5. Classification Report
# ─────────────────────────────────────────────
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# ─────────────────────────────────────────────
# 6. Confusion Matrix
# ─────────────────────────────────────────────
cm   = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=data.target_names)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix"); plt.show()

# ─────────────────────────────────────────────
# 7. ROC Curve
# ─────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc_score   = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, 'b-', lw=2, label=f'AUC = {auc_score:.4f}')
plt.plot([0,1],[0,1],'r--')
plt.fill_between(fpr, tpr, alpha=0.1, color='blue')
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curve"); plt.legend(); plt.grid(True); plt.show()

# ─────────────────────────────────────────────
# 8. Precision-Recall Curve
# ─────────────────────────────────────────────
prec_vals, rec_vals, _ = precision_recall_curve(y_test, y_prob)
plt.figure(figsize=(7, 5))
plt.plot(rec_vals, prec_vals, 'g-', lw=2)
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Precision-Recall Curve"); plt.grid(True); plt.show()

# ─────────────────────────────────────────────
# 9. Log Loss
# ─────────────────────────────────────────────
print(f"\nLog Loss: {log_loss(y_test, y_prob):.4f}")
print(f"AUC:      {auc_score:.4f}")
```

### Hyperparameter Tuning — GridSearchCV

```python
# ─────────────────────────────────────────────
# GridSearchCV for C and penalty
# ─────────────────────────────────────────────
param_grid = [
    {
        'model__penalty': ['l2'],
        'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'model__solver': ['lbfgs']
    },
    {
        'model__penalty': ['l1'],
        'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'model__solver': ['liblinear']
    }
]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    pipeline, param_grid,
    cv=cv, scoring='roc_auc',
    n_jobs=-1, verbose=1
)
grid.fit(X_train, y_train)

print(f"\nBest Parameters: {grid.best_params_}")
print(f"Best CV AUC:     {grid.best_score_:.4f}")
best = grid.best_estimator_
print(f"Test AUC:        {roc_auc_score(y_test, best.predict_proba(X_test)[:,1]):.4f}")
```

### C vs AUC Regularization Path

```python
# ─────────────────────────────────────────────
# Regularization path
# ─────────────────────────────────────────────
C_values = np.logspace(-4, 4, 50)
train_aucs, test_aucs = [], []

for C in C_values:
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(C=C, max_iter=1000, random_state=42))
    ])
    pipe.fit(X_train, y_train)
    train_aucs.append(roc_auc_score(y_train, pipe.predict_proba(X_train)[:,1]))
    test_aucs.append(roc_auc_score(y_test,   pipe.predict_proba(X_test)[:,1]))

plt.figure(figsize=(9, 5))
plt.semilogx(C_values, train_aucs, 'b-', label='Train AUC')
plt.semilogx(C_values, test_aucs,  'r-', label='Test AUC')
plt.xlabel("C (Inverse Regularization)"); plt.ylabel("AUC")
plt.title("Regularization Strength vs AUC")
plt.legend(); plt.grid(True); plt.show()
```

### Feature Coefficients (Log-Odds → Odds Ratio)

```python
# ─────────────────────────────────────────────
# Coefficients and Odds Ratios
# ─────────────────────────────────────────────
model_fitted = pipeline.named_steps['model']
coef         = model_fitted.coef_[0]
odds_ratios  = np.exp(coef)

coef_df = pd.DataFrame({
    'Feature':     data.feature_names,
    'Coefficient': coef,
    'Odds_Ratio':  odds_ratios
}).sort_values('Coefficient')

print("\nTop Features by |Coefficient|:")
print(coef_df.reindex(coef_df['Coefficient'].abs().sort_values(ascending=False).index)
             .head(10).to_string(index=False))

# Plot
colors = ['red' if c < 0 else 'green' for c in coef_df['Coefficient']]
plt.figure(figsize=(10, 7))
plt.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, alpha=0.7)
plt.axvline(0, color='black', linewidth=0.8)
plt.xlabel("Coefficient (log-odds)")
plt.title("Logistic Regression Feature Coefficients")
plt.tight_layout(); plt.show()
```

### Threshold Tuning

```python
# ─────────────────────────────────────────────
# Find Best Threshold for F1
# ─────────────────────────────────────────────
thresholds = np.linspace(0.01, 0.99, 100)
f1_scores  = [f1_score(y_test, (y_prob >= t).astype(int)) for t in thresholds]

best_thresh = thresholds[np.argmax(f1_scores)]
print(f"Best Threshold: {best_thresh:.2f} | Best F1: {max(f1_scores):.4f}")

plt.figure(figsize=(8, 4))
plt.plot(thresholds, f1_scores, 'b-')
plt.axvline(best_thresh, color='red', linestyle='--', label=f'Best={best_thresh:.2f}')
plt.xlabel("Threshold"); plt.ylabel("F1 Score")
plt.title("F1 Score vs Threshold"); plt.legend(); plt.grid(True); plt.show()
```

### Imbalanced Dataset Handling

```python
# ─────────────────────────────────────────────
# Handle Class Imbalance
# ─────────────────────────────────────────────

# Option 1: class_weight='balanced' (fastest, try first)
model_balanced = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(class_weight='balanced', max_iter=1000))
])
model_balanced.fit(X_train, y_train)
print(f"Balanced Model F1: {f1_score(y_test, model_balanced.predict(X_test)):.4f}")

# Option 2: SMOTE oversampling (requires: pip install imbalanced-learn)
try:
    from imblearn.over_sampling import SMOTE
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"After SMOTE: {np.bincount(y_res)}")
except ImportError:
    print("pip install imbalanced-learn  for SMOTE")

# Option 3: Manual class weights
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
weight_dict   = dict(zip(np.unique(y_train), class_weights))
print(f"Computed class weights: {weight_dict}")
```

### Learning Curves

```python
# ─────────────────────────────────────────────
# Learning Curves — Bias-Variance Diagnosis
# ─────────────────────────────────────────────
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    pipeline, X, y,
    cv=5, scoring='roc_auc',
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1
)

t_mean, t_std = train_scores.mean(1), train_scores.std(1)
v_mean, v_std = val_scores.mean(1),   val_scores.std(1)

plt.figure(figsize=(9, 5))
plt.plot(train_sizes, t_mean, 'b-o', label='Train AUC')
plt.plot(train_sizes, v_mean, 'r-o', label='Val AUC')
plt.fill_between(train_sizes, t_mean-t_std, t_mean+t_std, alpha=0.1, color='blue')
plt.fill_between(train_sizes, v_mean-v_std, v_mean+v_std, alpha=0.1, color='red')
plt.xlabel("Training Size"); plt.ylabel("AUC")
plt.title("Learning Curves"); plt.legend(); plt.grid(True); plt.show()

# Interpretation:
# High train AUC, low val AUC → OVERFITTING → increase C or add more data
# Both AUC low → UNDERFITTING → decrease C, add features, use polynomial
```

---

## 16. Common Mistakes & Tips

### ❌ Mistake 1: Not Scaling Features
```python
# Wrong — LR with regularization is scale-sensitive
model.fit(X_train, y_train)

# Correct — always use a pipeline with scaler
pipeline = Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression())])
```

### ❌ Mistake 2: Using Accuracy on Imbalanced Data
```python
# Wrong — 99% accuracy but predicts majority class only
print(model.score(X_test, y_test))

# Correct
print(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
print(f1_score(y_test, model.predict(X_test)))
```

### ❌ Mistake 3: Ignoring ConvergenceWarning
```python
# Wrong
LogisticRegression()                  # default max_iter=100 often fails

# Correct
LogisticRegression(max_iter=1000)
```

### ❌ Mistake 4: Forgetting to Stratify Split
```python
# Wrong
train_test_split(X, y, test_size=0.2)

# Correct
train_test_split(X, y, test_size=0.2, stratify=y)
```

### ❌ Mistake 5: Using predict() Instead of predict_proba() for AUC
```python
# Wrong — loses probability information
roc_auc_score(y_test, model.predict(X_test))

# Correct
roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
```

### ✅ Tips

- Always wrap in a **Pipeline** — avoids data leakage, cleaner code
- Use **stratified K-Fold** for cross-validation
- Try `class_weight='balanced'` first for imbalanced data — it's fast
- Tune **threshold** in addition to hyperparameters for imbalanced problems
- Check **odds ratios** = `np.exp(coef_)` for interpretability
- Use `solver='saga'` for large datasets or L1 regularization

---

## 17. Interview Questions

**Q1: Why is logistic regression called "regression" if it's a classifier?**
> It regresses a linear function into probabilities using sigmoid. The "regression" refers to estimating the probability, not the classification step itself.

**Q2: Why use cross-entropy loss instead of MSE for logistic regression?**
> MSE with sigmoid produces a non-convex loss surface with many local minima, making gradient descent unreliable. Cross-entropy is convex — one global minimum guaranteed.

**Q3: What is the relationship between C and λ in sklearn?**
> `C = 1/λ`. Larger C means smaller penalty (less regularization). Smaller C means stronger regularization and a simpler model.

**Q4: What is the decision boundary of logistic regression?**
> The hyperplane where `wᵀx + b = 0`, i.e., predicted probability = 0.5. Linear in original feature space; non-linear with polynomial features.

**Q5: Can logistic regression handle multiclass?**
> Yes, via One-vs-Rest (K binary classifiers) or Softmax/Multinomial regression (directly models all K classes). Sklearn handles both via `multi_class` parameter.

**Q6: What is the log-odds interpretation of coefficients?**
> Each coefficient `wⱼ` is the change in log-odds per unit increase in feature j. `exp(wⱼ)` gives the odds ratio — if it's 2.0, the feature doubles the odds of the positive class.

**Q7: How do you handle class imbalance?**
> Options: `class_weight='balanced'`, SMOTE oversampling, undersampling, adjusting decision threshold, or optimizing F1/AUC instead of accuracy.

**Q8: Precision vs Recall — when to prioritize each?**
> Recall: when missing positives is costly (cancer detection, fraud). Precision: when false alarms are costly (spam filter, legal decisions).

**Q9: What are logistic regression's assumptions?**
> Binary outcome, linearity of log-odds, independence of observations, no multicollinearity, sufficient sample size (≥10 events per predictor).

**Q10: What is the gradient of log loss w.r.t. weights?**
```
∂J/∂w = (1/m) · Xᵀ(A - y)
```
> Identical in form to linear regression — the difference is `A = σ(Xw+b)` instead of `A = Xw+b`.

---

## 18. Resources

### 📖 Books & Courses

**Andrew Ng — Machine Learning Specialization (Coursera)**
- Week 3: Binary Classification, Sigmoid, Log Loss, Gradient Descent
- Week 4: Regularization for Logistic Regression
- 🔗 [Coursera Course](https://www.coursera.org/specializations/machine-learning-introduction)
- 🔗 [CS229 Lecture Notes (Free PDF)](https://cs229.stanford.edu/notes2022fall/main_notes.pdf)

**Hands-On Machine Learning (Aurélien Géron, 3rd Ed.)**
- Chapter 3: Classification (metrics, confusion matrix, ROC, PR curves)
- Chapter 4: Training Models (logistic regression math, softmax)
- 🔗 [GitHub Notebooks](https://github.com/ageron/handson-ml3)

**Introduction to Statistical Learning with Python (ISLP)**
- Chapter 4: Classification — Logistic Regression, MLE, Multiclass, LDA comparison
- 🔗 [Free PDF](https://www.statlearning.com/)
- 🔗 [Python Code Labs](https://www.statlearning.com/resources-python)

### 📺 YouTube — StatQuest with Josh Starmer

| Topic | Link |
|---|---|
| Logistic Regression Main Ideas | [youtube.com/watch?v=yIYKR4sgzI8](https://www.youtube.com/watch?v=yIYKR4sgzI8) |
| Coefficients (log-odds) Pt 1 | [youtube.com/watch?v=vN5cNN2-HWE](https://www.youtube.com/watch?v=vN5cNN2-HWE) |
| MLE for Logistic Regression Pt 2 | [youtube.com/watch?v=BfKanl1aSG0](https://www.youtube.com/watch?v=BfKanl1aSG0) |
| ROC and AUC | [youtube.com/watch?v=4jRBRDbJemM](https://www.youtube.com/watch?v=4jRBRDbJemM) |
| Precision and Recall | [youtube.com/watch?v=7J_qcttfnJI](https://www.youtube.com/watch?v=7J_qcttfnJI) |
| Ridge + Lasso Regularization | [youtube.com/watch?v=Q81RR3yKn30](https://www.youtube.com/watch?v=Q81RR3yKn30) |

### 📄 Scikit-learn Documentation

| Page | Link |
|---|---|
| LogisticRegression API | [scikit-learn.org — LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) |
| Linear Models User Guide | [scikit-learn.org — linear_model](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression) |
| Classification Metrics | [scikit-learn.org — model_evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics) |
| Imbalanced Classes | [scikit-learn.org — class_weight](https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html) |

---

## 📎 Quick Reference Cheat Sheet

```
Model:            ŷ = σ(wᵀx + b)
Sigmoid:          σ(z) = 1 / (1 + e⁻ᶻ)
Sigmoid deriv:    σ'(z) = σ(z)(1 - σ(z))
Log-Odds:         log(p/(1-p)) = wᵀx + b
Cost (log loss):  J = -(1/m) Σ[y·log(ŷ) + (1-y)·log(1-ŷ)]
Gradient dw:      (1/m) · Xᵀ(A - y)
Gradient db:      (1/m) · Σ(A - y)
Update:           w := w - α·dw,  b := b - α·db
Ridge cost:       J + (λ/2m)·Σwⱼ²
Lasso cost:       J + (λ/m)·Σ|wⱼ|
Decision:         predict 1 if σ(z) ≥ 0.5  (i.e., z ≥ 0)

Precision:        TP / (TP + FP)
Recall:           TP / (TP + FN)
F1:               2·P·R / (P + R)
Specificity:      TN / (TN + FP)
AUC = 1.0:        perfect  |  AUC = 0.5: random

C (sklearn):      C = 1/λ  →  large C = less regularization
Odds Ratio:       exp(wⱼ)  →  multiplicative effect on odds

Softmax:          P(y=k|x) = exp(wₖᵀx) / Σⱼ exp(wⱼᵀx)
Cat. Cross-Ent.:  J = -(1/m) Σᵢ Σₖ yᵢₖ · log(ŷᵢₖ)
```

---

*Made with 💙 for ML/DL learners — covers sigmoid math, MLE derivation, gradient descent, every hyperparameter, sklearn pipelines, evaluation metrics, imbalanced data, multiclass, and all the best learning resources.*
