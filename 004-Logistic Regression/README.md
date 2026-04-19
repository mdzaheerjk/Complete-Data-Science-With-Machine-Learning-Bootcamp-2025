# 📊 Logistic Regression — Complete ML/DL Job-Ready Notes

> **Sources:** Andrew Ng ML Specialization (Coursera) · Hands-On ML with Scikit-Learn (Aurélien Géron) · StatQuest (Josh Starmer) · Scikit-Learn Docs · ISLP (James et al.)

---

## 📚 Table of Contents

1. [What is Logistic Regression?](#1-what-is-logistic-regression)
2. [Types of Classification](#2-types-of-classification)
3. [The Sigmoid Function](#3-the-sigmoid-function)
4. [The Math — Decision Boundary](#4-the-math--decision-boundary)
5. [Cost Function — Log Loss (Binary Cross-Entropy)](#5-cost-function--log-loss)
6. [Gradient Descent for Logistic Regression](#6-gradient-descent)
7. [Maximum Likelihood Estimation (MLE)](#7-maximum-likelihood-estimation)
8. [Multiclass — Softmax & OvR & OvO](#8-multiclass)
9. [Assumptions](#9-assumptions)
10. [Evaluation Metrics](#10-evaluation-metrics)
11. [Python from Scratch](#11-python-from-scratch)
12. [Scikit-Learn Implementation](#12-scikit-learn-implementation)
13. [Hyperparameters & Tuning](#13-hyperparameters--tuning)
14. [Regularization in Logistic Regression](#14-regularization)
15. [Class Imbalance](#15-class-imbalance)
16. [Bias-Variance & Learning Curves](#16-bias-variance--learning-curves)
17. [Common Interview Questions](#17-common-interview-questions)
18. [Resources](#18-resources)

---

## 1. What is Logistic Regression?

> **Simple Statement:** Despite the name, Logistic Regression is a **classification** algorithm. It predicts the **probability** that an input belongs to a class (e.g., spam or not spam), then uses a threshold to make a decision.

- **Task type:** Supervised Learning → Classification
- **Output:** Probability $\in (0, 1)$, then thresholded to class label
- **Key idea:** Apply the **sigmoid function** to a linear combination of features so the output is always between 0 and 1
- **Why not Linear Regression for classification?** LR can predict values outside [0, 1], is sensitive to outliers, and doesn't model class probabilities correctly

**Real-world examples:**
- Email spam detection (spam / not spam)
- Disease diagnosis (positive / negative)
- Credit default prediction (default / no default)
- Customer churn prediction

---

## 2. Types of Classification

| Type | Classes | Example | Sklearn Strategy |
|------|---------|---------|-----------------|
| **Binary** | 2 | Spam / Not Spam | Direct sigmoid |
| **Multiclass** | > 2 | Digit recognition (0–9) | OvR, OvO, Softmax |
| **Multilabel** | Multiple binary labels | Movie genres | One LR per label |

---

## 3. The Sigmoid Function

> **Simple Statement:** The sigmoid function is an S-shaped curve that squashes any real number into the range (0, 1). This lets us interpret the output as a probability.

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**Properties:**
- $\sigma(0) = 0.5$
- $\sigma(z) \to 1$ as $z \to +\infty$
- $\sigma(z) \to 0$ as $z \to -\infty$
- Symmetric: $\sigma(-z) = 1 - \sigma(z)$
- Derivative: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ ← elegant!

**Where $z$ comes from:**

$$z = \mathbf{w}^T \mathbf{x} + b = w_1x_1 + w_2x_2 + \cdots + w_nx_n + b$$

**Full model:**

$$\hat{y} = P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = np.linspace(-10, 10, 300)
plt.figure(figsize=(8, 4))
plt.plot(z, sigmoid(z), 'b-', linewidth=2)
plt.axhline(0.5, color='red', linestyle='--', label='Threshold = 0.5')
plt.axvline(0,   color='gray', linestyle='--')
plt.xlabel('z'); plt.ylabel('σ(z)')
plt.title('Sigmoid Function'); plt.legend(); plt.grid(True)
plt.show()
```

---

## 4. The Math — Decision Boundary

> **Simple Statement:** The decision boundary is the line (or curve) where the predicted probability = 0.5. Everything on one side → Class 1, other side → Class 0.

**Prediction rule:**

$$\hat{y} = \begin{cases} 1 & \text{if } \hat{p} \geq 0.5 \\ 0 & \text{if } \hat{p} < 0.5 \end{cases}$$

**Since $\sigma(z) = 0.5$ when $z = 0$:**

$$\hat{y} = 1 \iff \mathbf{w}^T \mathbf{x} + b \geq 0$$

So the **decision boundary** is:

$$\mathbf{w}^T \mathbf{x} + b = 0$$

This is a **hyperplane** in feature space. For 2D: a line $w_1x_1 + w_2x_2 + b = 0$.

**Non-linear decision boundaries:** Use polynomial features to get circular or complex boundaries — the model is still linear in parameters.

```
Example: w₁ = 1, w₂ = 1, b = -1.5
Decision boundary: x₁ + x₂ = 1.5
Points where x₁ + x₂ ≥ 1.5 → Class 1
Points where x₁ + x₂ < 1.5  → Class 0
```

---

## 5. Cost Function — Log Loss

> **Simple Statement:** We can't use MSE for logistic regression because the sigmoid makes the cost function non-convex (many local minima). Instead we use Log Loss, which is convex and penalizes confident wrong predictions very heavily.

### Why Not MSE?

If we use MSE with sigmoid: $J = \frac{1}{2m}\sum(\sigma(z) - y)^2$
- The cost surface is **non-convex** — gradient descent may get stuck in local minima
- Log Loss gives a **convex** surface with a single global minimum

### Binary Cross-Entropy (Log Loss)

For a single example:

$$\mathcal{L}(\hat{y}, y) = -\left[ y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \right]$$

**Intuition:**
- If $y = 1$: loss $= -\log(\hat{y})$. When $\hat{y} \to 1$, loss $\to 0$. When $\hat{y} \to 0$, loss $\to \infty$
- If $y = 0$: loss $= -\log(1 - \hat{y})$. When $\hat{y} \to 0$, loss $\to 0$. When $\hat{y} \to 1$, loss $\to \infty$

**Over the entire training set:**

$$J(\mathbf{w}, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log\hat{y}^{(i)} + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]$$

**Compact notation:** $\hat{y}^{(i)} = \sigma(\mathbf{w}^T \mathbf{x}^{(i)} + b)$

**Matrix form:**

$$J(\boldsymbol{\theta}) = -\frac{1}{m} \left[ \mathbf{y}^T \log(\hat{\mathbf{y}}) + (1-\mathbf{y})^T \log(1 - \hat{\mathbf{y}}) \right]$$

```python
def log_loss(y_true, y_pred):
    """Binary cross-entropy loss."""
    eps = 1e-15  # clip for numerical stability (avoid log(0))
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
```

---

## 6. Gradient Descent

> **Simple Statement:** We minimize the log loss by computing which direction (gradient) increases the loss the most, then stepping in the opposite direction. Remarkably, the update rule looks identical to linear regression!

### Deriving the Gradients

$$J(\mathbf{w}, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log \sigma(z^{(i)}) + (1-y^{(i)}) \log(1 - \sigma(z^{(i)})) \right]$$

Using chain rule and the elegant sigmoid derivative $\sigma'(z) = \sigma(z)(1-\sigma(z))$:

$$\frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} \left(\hat{y}^{(i)} - y^{(i)}\right) x_j^{(i)}$$

$$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} \left(\hat{y}^{(i)} - y^{(i)}\right)$$

**These are identical to linear regression gradients!** The only difference is that $\hat{y} = \sigma(\mathbf{w}^T\mathbf{x} + b)$ instead of $\hat{y} = \mathbf{w}^T\mathbf{x} + b$.

### Update Rules (Simultaneous Update)

$$w_j := w_j - \alpha \cdot \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)}) x_j^{(i)}$$

$$b := b - \alpha \cdot \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})$$

### Vectorized Update

$$\mathbf{w} := \mathbf{w} - \frac{\alpha}{m} \mathbf{X}^T (\hat{\mathbf{y}} - \mathbf{y})$$

$$b := b - \frac{\alpha}{m} \sum (\hat{\mathbf{y}} - \mathbf{y})$$

where $\hat{\mathbf{y}} = \sigma(\mathbf{X}\mathbf{w} + b)$

### Regularized Gradient Descent

With L2 (Ridge) regularization:

$$\frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)}) x_j^{(i)} + \frac{\lambda}{m} w_j$$

$$w_j := w_j - \alpha \left[ \frac{1}{m} \sum (\hat{y}^{(i)} - y^{(i)}) x_j^{(i)} + \frac{\lambda}{m} w_j \right]$$

Note: **bias $b$ is NOT regularized** (standard practice).

---

## 7. Maximum Likelihood Estimation

> **Simple Statement:** MLE finds the weights that make the observed training data most probable. Maximizing likelihood is mathematically equivalent to minimizing log loss.

### Likelihood for One Example

$$P(y^{(i)} \mid \mathbf{x}^{(i)}; \mathbf{w}, b) = \hat{y}^{(i)^{y^{(i)}}} \cdot (1 - \hat{y}^{(i)})^{1 - y^{(i)}}$$

- If $y^{(i)} = 1$: probability = $\hat{y}^{(i)}$
- If $y^{(i)} = 0$: probability = $1 - \hat{y}^{(i)}$

### Joint Likelihood (all examples, assuming independence)

$$\mathcal{L}(\mathbf{w}, b) = \prod_{i=1}^{m} P(y^{(i)} \mid \mathbf{x}^{(i)})$$

### Log-Likelihood (easier to optimize)

$$\log \mathcal{L} = \sum_{i=1}^{m} \left[ y^{(i)} \log \hat{y}^{(i)} + (1-y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]$$

**Minimizing log loss = Maximizing log-likelihood.** This is why log loss is the natural and principled choice for logistic regression.

---

## 8. Multiclass

### 8.1 One-vs-Rest (OvR) / One-vs-All (OvA)

> **Simple Statement:** Train $K$ binary classifiers. Classifier $k$ learns to distinguish class $k$ from all other classes combined. Predict the class with highest probability.

- Train $K$ models: class 0 vs rest, class 1 vs rest, ..., class $K-1$ vs rest
- At inference: run all $K$ models, pick class with highest score
- Fast, simple, works well in practice
- **Default in sklearn** for `LogisticRegression`

### 8.2 One-vs-One (OvO)

> **Simple Statement:** Train one classifier for each pair of classes. For $K$ classes, train $K(K-1)/2$ classifiers. Use majority voting.

- For $K=10$: $10 \times 9 / 2 = 45$ classifiers!
- Each classifier trained on a smaller subset (2 classes only)
- Scales poorly with $K$, but each model trains faster
- Preferred by SVM

### 8.3 Softmax Regression (Multinomial Logistic Regression)

> **Simple Statement:** Extend logistic regression to multiple classes directly. Compute a score for each class, then normalize with softmax to get probabilities that sum to 1.

**Softmax function:**

$$P(y = k \mid \mathbf{x}) = \frac{e^{\mathbf{w}_k^T \mathbf{x} + b_k}}{\sum_{j=1}^{K} e^{\mathbf{w}_j^T \mathbf{x} + b_j}}$$

**Cross-entropy loss (multiclass):**

$$J = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} y_k^{(i)} \log \hat{p}_k^{(i)}$$

where $y_k^{(i)}$ is 1 if example $i$ belongs to class $k$ (one-hot encoding).

```python
# Softmax from scratch
def softmax(z):
    """z shape: (m, K)"""
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # numerical stability
    return exp_z / exp_z.sum(axis=1, keepdims=True)

# sklearn: use multi_class='multinomial' + solver='lbfgs'
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
```

---

## 9. Assumptions

| # | Assumption | Description | How to check |
|---|-----------|-------------|-------------|
| 1 | **Binary/categorical outcome** | $y$ must be categorical | By definition |
| 2 | **Independence** | Observations are independent | Domain knowledge |
| 3 | **Linearity in log-odds** | $\log\frac{p}{1-p}$ is linear in features | Box-Tidwell test, partial residual plots |
| 4 | **No multicollinearity** | Features not highly correlated | VIF, correlation matrix |
| 5 | **No extreme outliers** | Outliers can distort decision boundary | Cook's distance, leverage |
| 6 | **Large enough sample** | Enough examples per class | Rule of thumb: ≥10 events per predictor |

**Key difference from Linear Regression:** Logistic Regression does NOT assume normality of residuals or homoscedasticity.

---

## 10. Evaluation Metrics

### 10.1 Confusion Matrix

```
                 Predicted
                  0      1
Actual  0  [  TN  |  FP  ]
        1  [  FN  |  TP  ]

TN = True Negative   FP = False Positive (Type I Error)
FN = False Negative  TP = True Positive
(Type II Error)
```

### 10.2 Core Metrics

| Metric | Formula | Simple Statement | Use When |
|--------|---------|-----------------|----------|
| **Accuracy** | $\frac{TP+TN}{TP+TN+FP+FN}$ | % correctly classified | Balanced classes |
| **Precision** | $\frac{TP}{TP+FP}$ | Of predicted positives, how many are correct? | FP is costly (spam) |
| **Recall (Sensitivity)** | $\frac{TP}{TP+FN}$ | Of actual positives, how many did we catch? | FN is costly (cancer) |
| **F1 Score** | $\frac{2 \cdot P \cdot R}{P + R}$ | Harmonic mean of Precision & Recall | Imbalanced classes |
| **Specificity** | $\frac{TN}{TN+FP}$ | Of actual negatives, how many correctly identified? | Medical tests |

### 10.3 ROC-AUC

> **Simple Statement:** ROC curve plots True Positive Rate vs False Positive Rate at all possible thresholds. AUC measures area under this curve. AUC = 1 is perfect, AUC = 0.5 is random guessing.

- **ROC curve:** Sweep threshold from 0 to 1, plot TPR (Recall) vs FPR at each threshold
- **AUC:** Single number summary. Higher = better at ranking positives above negatives
- Threshold-independent — great for comparing models

### 10.4 Precision-Recall Curve

> Use instead of ROC-AUC when classes are **highly imbalanced** (e.g., 1% positive rate). ROC-AUC can look great even with terrible precision on the minority class.

### 10.5 Log Loss (as evaluation metric)

$$\text{Log Loss} = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log \hat{p}^{(i)} + (1-y^{(i)}) \log(1-\hat{p}^{(i)}) \right]$$

- Penalizes confident wrong predictions
- Lower is better
- Useful when you care about calibrated probabilities

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix, classification_report,
    RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

# Comprehensive evaluation
def evaluate_classifier(model, X_test, y_test):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print("=" * 50)
    print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision : {precision_score(y_test, y_pred):.4f}")
    print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score  : {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC   : {roc_auc_score(y_test, y_proba):.4f}")
    print(f"Log Loss  : {log_loss(y_test, y_proba):.4f}")
    print("=" * 50)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=axes[0])
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=axes[1])
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=axes[2])
    plt.tight_layout()
    plt.show()
```

---

## 11. Python from Scratch

### 11.1 Binary Logistic Regression — Full Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegressionGD:
    """
    Binary Logistic Regression using Gradient Descent (vectorized).
    """
    def __init__(self, learning_rate=0.1, n_iterations=1000,
                 reg_lambda=0.0, verbose=False):
        self.lr        = learning_rate
        self.n_iter    = n_iterations
        self.lam       = reg_lambda      # L2 regularization strength
        self.verbose   = verbose
        self.weights   = None
        self.bias      = None
        self.cost_hist = []
    
    @staticmethod
    def sigmoid(z):
        # Numerically stable sigmoid
        return np.where(z >= 0,
                        1 / (1 + np.exp(-z)),
                        np.exp(z) / (1 + np.exp(z)))
    
    def _compute_cost(self, y, y_hat):
        m = len(y)
        eps = 1e-15
        y_hat = np.clip(y_hat, eps, 1 - eps)
        ce = -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        # L2 regularization term (don't regularize bias)
        reg = (self.lam / (2 * m)) * np.sum(self.weights ** 2)
        return ce + reg
    
    def fit(self, X, y):
        m, n = X.shape
        
        # Initialize parameters to zero
        self.weights = np.zeros(n)
        self.bias    = 0.0
        
        for i in range(self.n_iter):
            # ── Forward pass ──────────────────────────────────────
            z     = X @ self.weights + self.bias    # linear: (m,)
            y_hat = self.sigmoid(z)                 # probabilities: (m,)
            
            # ── Compute gradients ─────────────────────────────────
            errors = y_hat - y                      # (m,)
            dw = (1 / m) * (X.T @ errors) + (self.lam / m) * self.weights
            db = (1 / m) * np.sum(errors)
            
            # ── Update parameters (simultaneous) ──────────────────
            self.weights -= self.lr * dw
            self.bias    -= self.lr * db
            
            # ── Track cost ────────────────────────────────────────
            cost = self._compute_cost(y, y_hat)
            self.cost_hist.append(cost)
            
            if self.verbose and i % 100 == 0:
                print(f"Iter {i:5d} | Loss: {cost:.6f}")
        
        return self
    
    def predict_proba(self, X):
        z = X @ self.weights + self.bias
        return self.sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
    
    def score(self, X, y):
        return np.mean(self.predict(X) == y)


# ── Example Usage ─────────────────────────────────────────────────
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

X, y = make_classification(n_samples=1000, n_features=10,
                            n_informative=6, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

model = LogisticRegressionGD(learning_rate=0.1, n_iterations=1000,
                              reg_lambda=0.01, verbose=True)
model.fit(X_train_s, y_train)

print(f"\nTrain Accuracy: {model.score(X_train_s, y_train):.4f}")
print(f"Test  Accuracy: {model.score(X_test_s, y_test):.4f}")

# Plot learning curve
plt.figure(figsize=(7, 4))
plt.plot(model.cost_hist)
plt.xlabel("Iteration"); plt.ylabel("Log Loss")
plt.title("Learning Curve"); plt.grid(True)
plt.show()
```

### 11.2 Decision Boundary Visualization

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# 2D data for visualization
X2, y2 = make_classification(n_samples=300, n_features=2,
                              n_redundant=0, n_informative=2,
                              random_state=42)

scaler2 = StandardScaler()
X2s = scaler2.fit_transform(X2)

model2 = LogisticRegressionGD(learning_rate=0.1, n_iterations=2000)
model2.fit(X2s, y2)

# Create mesh grid
x1_min, x1_max = X2s[:, 0].min() - 0.5, X2s[:, 0].max() + 0.5
x2_min, x2_max = X2s[:, 1].min() - 0.5, X2s[:, 1].max() + 0.5
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 200),
                        np.linspace(x2_min, x2_max, 200))

Z = model2.predict_proba(np.c_[xx1.ravel(), xx2.ravel()])
Z = Z.reshape(xx1.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx1, xx2, Z, levels=20, cmap='RdBu', alpha=0.6)
plt.colorbar(label='P(y=1)')
plt.contour(xx1, xx2, Z, levels=[0.5], colors='black', linewidths=2)
plt.scatter(X2s[:, 0], X2s[:, 1], c=y2, cmap='RdBu_r', edgecolors='k', s=30)
plt.title("Logistic Regression Decision Boundary")
plt.xlabel("Feature 1"); plt.ylabel("Feature 2")
plt.show()
```

### 11.3 Threshold Tuning

```python
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

# Evaluate metrics at different thresholds
thresholds = np.linspace(0.1, 0.9, 50)
precisions, recalls, f1s = [], [], []

y_proba = model.predict_proba(X_test_s)

for t in thresholds:
    y_pred_t = (y_proba >= t).astype(int)
    precisions.append(precision_score(y_test, y_pred_t, zero_division=0))
    recalls.append(recall_score(y_test, y_pred_t, zero_division=0))
    f1s.append(f1_score(y_test, y_pred_t, zero_division=0))

best_t = thresholds[np.argmax(f1s)]
print(f"Best threshold (max F1): {best_t:.2f}")

plt.figure(figsize=(8, 4))
plt.plot(thresholds, precisions, label='Precision')
plt.plot(thresholds, recalls,   label='Recall')
plt.plot(thresholds, f1s,       label='F1 Score')
plt.axvline(best_t, color='red', linestyle='--', label=f'Best t={best_t:.2f}')
plt.xlabel("Threshold"); plt.ylabel("Score")
plt.title("Precision / Recall / F1 vs Threshold")
plt.legend(); plt.grid(True)
plt.show()
```

---

## 12. Scikit-Learn Implementation

### 12.1 Full Pipeline

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.datasets import load_breast_cancer
import numpy as np

# ── Load data ─────────────────────────────────────────────────────
data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # stratify preserves class ratio
)

# ── Build pipeline ────────────────────────────────────────────────
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  LogisticRegression(C=1.0, max_iter=1000, random_state=42))
])

# ── Train ─────────────────────────────────────────────────────────
pipe.fit(X_train, y_train)

# ── Predict ───────────────────────────────────────────────────────
y_pred  = pipe.predict(X_test)
y_proba = pipe.predict_proba(X_test)[:, 1]

# ── Stratified Cross-Validation ───────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring='roc_auc')
print(f"CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── Full evaluation ───────────────────────────────────────────────
evaluate_classifier(pipe, X_test, y_test)
```

### 12.2 Grid Search Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import loguniform

# Grid search
param_grid = {
    'model__C':       [0.001, 0.01, 0.1, 1, 10, 100],
    'model__penalty': ['l1', 'l2'],
    'model__solver':  ['liblinear'],   # supports both l1 and l2
}

gs = GridSearchCV(pipe, param_grid, cv=5, scoring='roc_auc',
                  n_jobs=-1, verbose=1)
gs.fit(X_train, y_train)
print(f"Best params : {gs.best_params_}")
print(f"Best ROC-AUC: {gs.best_score_:.4f}")

# Randomized search (faster for large spaces)
param_dist = {
    'model__C':       loguniform(1e-4, 1e3),
    'model__penalty': ['l1', 'l2', 'elasticnet'],
    'model__solver':  ['saga'],
    'model__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],  # for elasticnet
}

rs = RandomizedSearchCV(pipe, param_dist, n_iter=50, cv=5,
                        scoring='roc_auc', n_jobs=-1, random_state=42)
rs.fit(X_train, y_train)
print(f"\nBest params (random): {rs.best_params_}")
```

### 12.3 Multiclass Classification

```python
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

digits = load_digits()
X, y = digits.data, digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# OvR (default)
pipe_ovr = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(multi_class='ovr', solver='liblinear',
                                  C=1.0, max_iter=1000))
])

# Multinomial (Softmax)
pipe_softmax = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(multi_class='multinomial', solver='lbfgs',
                                  C=1.0, max_iter=1000))
])

for name, p in [('OvR', pipe_ovr), ('Softmax', pipe_softmax)]:
    p.fit(X_train, y_train)
    acc = p.score(X_test, y_test)
    print(f"{name} Accuracy: {acc:.4f}")

print(classification_report(y_test, pipe_softmax.predict(X_test)))
```

### 12.4 Calibrated Probabilities

```python
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay

# Check if probabilities are well-calibrated
fig, ax = plt.subplots(figsize=(7, 5))
CalibrationDisplay.from_estimator(pipe, X_test, y_test,
                                   n_bins=10, ax=ax, name='LogReg')
ax.set_title("Calibration Curve (Reliability Diagram)")
plt.show()

# If not calibrated, wrap with CalibratedClassifierCV
# calibrated = CalibratedClassifierCV(pipe, method='isotonic', cv=5)
# calibrated.fit(X_train, y_train)
```

---

## 13. Hyperparameters & Tuning

### LogisticRegression (sklearn) — Complete Reference

| Parameter | Default | Description | Options / Range |
|-----------|---------|-------------|----------------|
| `C` | `1.0` | **Inverse** regularization strength ($C = 1/\lambda$). Smaller C = stronger regularization | `1e-4` to `1e4` (log scale) |
| `penalty` | `'l2'` | Type of regularization | `'l1'`, `'l2'`, `'elasticnet'`, `None` |
| `solver` | `'lbfgs'` | Optimization algorithm | See table below |
| `max_iter` | `100` | Max iterations for solver convergence | `100` to `10000` |
| `tol` | `1e-4` | Convergence tolerance | `1e-6` to `1e-2` |
| `multi_class` | `'auto'` | Strategy for multiclass | `'ovr'`, `'multinomial'`, `'auto'` |
| `class_weight` | `None` | Weights for imbalanced classes | `'balanced'` or dict `{0:1, 1:5}` |
| `l1_ratio` | `None` | ElasticNet mix (0=L2, 1=L1) | `0.0` to `1.0` |
| `fit_intercept` | `True` | Compute bias $b$ | `True`/`False` |
| `intercept_scaling` | `1.0` | Scale bias feature when solver='liblinear' | `1.0` |
| `random_state` | `None` | Reproducibility | Any integer |
| `n_jobs` | `None` | Parallel jobs (OvR only) | `-1` for all CPUs |
| `warm_start` | `False` | Reuse previous solution | `True` for incremental |
| `verbose` | `0` | Solver verbosity | `0`, `1`, `2` |
| `dual` | `False` | Dual formulation (liblinear, L2 only) | `True` when n_samples < n_features |

### Solver Compatibility Matrix

| Solver | L1 | L2 | ElasticNet | None | Multiclass | Best for |
|--------|----|----|-----------|------|-----------|---------|
| `lbfgs` | ❌ | ✅ | ❌ | ✅ | multinomial | Default, small-medium data |
| `liblinear` | ✅ | ✅ | ❌ | ❌ | OvR only | Small datasets, L1 |
| `newton-cg` | ❌ | ✅ | ❌ | ✅ | multinomial | Dense data |
| `newton-cholesky` | ❌ | ✅ | ❌ | ✅ | OvR only | n_samples >> n_features |
| `sag` | ❌ | ✅ | ❌ | ✅ | multinomial | Large datasets |
| `saga` | ✅ | ✅ | ✅ | ✅ | multinomial | **Best for large datasets + L1/ElasticNet** |

> **Rule of thumb:** Start with `lbfgs`. For large data use `saga`. For L1 penalty use `liblinear` (small) or `saga` (large).

### Key Tuning Workflow

```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=2000, random_state=42))
])

# Step 1: Find best C and penalty
param_grid_1 = {
    'lr__C':       np.logspace(-4, 4, 20),
    'lr__penalty': ['l2'],
    'lr__solver':  ['lbfgs'],
}

# Step 2: Try L1 for feature selection
param_grid_2 = {
    'lr__C':       np.logspace(-4, 4, 20),
    'lr__penalty': ['l1'],
    'lr__solver':  ['liblinear'],
}

# Step 3: Try ElasticNet
param_grid_3 = {
    'lr__C':         np.logspace(-4, 4, 10),
    'lr__penalty':   ['elasticnet'],
    'lr__solver':    ['saga'],
    'lr__l1_ratio':  [0.1, 0.3, 0.5, 0.7, 0.9],
}

gs = GridSearchCV(pipe, param_grid_1, cv=5, scoring='roc_auc', n_jobs=-1)
gs.fit(X_train, y_train)
print(f"Best C: {gs.best_params_['lr__C']:.6f}")
```

---

## 14. Regularization

> **Simple Statement:** Regularization penalizes large weights to prevent overfitting. In logistic regression, `C` is the **inverse** of $\lambda$ — **smaller C = stronger regularization**. This is the opposite of linear regression where $\alpha$ (or $\lambda$) directly controls strength.

### L2 Regularization (Default)

$$J = -\frac{1}{m}\sum \text{log loss} + \frac{1}{2C} \sum_{j=1}^{n} w_j^2$$

- Shrinks all weights, never to zero
- Handles multicollinearity well

### L1 Regularization (Lasso)

$$J = -\frac{1}{m}\sum \text{log loss} + \frac{1}{C} \sum_{j=1}^{n} |w_j|$$

- Can set weights to exactly 0 → feature selection
- Good when many features are irrelevant

### Effect of C on Decision Boundary

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                             n_informative=2, random_state=42)
X = StandardScaler().fit_transform(X)

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
C_values = [0.001, 0.1, 1, 100]

for ax, C in zip(axes, C_values):
    model = LogisticRegression(C=C, solver='lbfgs', max_iter=1000)
    model.fit(X, y)
    
    xx, yy = np.meshgrid(np.linspace(X[:,0].min()-0.5, X[:,0].max()+0.5, 200),
                          np.linspace(X[:,1].min()-0.5, X[:,1].max()+0.5, 200))
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, levels=20, cmap='RdBu', alpha=0.6)
    ax.contour(xx, yy, Z, levels=[0.5], colors='black')
    ax.scatter(X[:,0], X[:,1], c=y, cmap='RdBu_r', edgecolors='k', s=20)
    ax.set_title(f"C={C}\nAcc={model.score(X,y):.2f}")

plt.tight_layout()
plt.show()
```

---

## 15. Class Imbalance

> **Simple Statement:** When one class has far fewer examples (e.g., 5% fraud, 95% normal), the model can cheat by always predicting the majority class and get high accuracy. We need special handling.

### Strategies

| Strategy | How | Sklearn |
|----------|-----|---------|
| `class_weight='balanced'` | Automatically reweights loss by inverse class frequency | `LogisticRegression(class_weight='balanced')` |
| Custom weights | Manually specify | `class_weight={0: 1, 1: 10}` |
| Oversampling (SMOTE) | Synthesize minority samples | `imbalanced-learn` library |
| Undersampling | Remove majority samples | `RandomUnderSampler` |
| Threshold tuning | Lower threshold for minority class | Post-training adjustment |

```python
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Option 1: Auto balanced
model_balanced = LogisticRegression(class_weight='balanced', solver='lbfgs')

# Option 2: Compute weights manually
classes = np.unique(y_train)
weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weights = dict(zip(classes, weights))
print("Class weights:", class_weights)
model_manual = LogisticRegression(class_weight=class_weights, solver='lbfgs')

# Option 3: SMOTE (requires imbalanced-learn)
# pip install imbalanced-learn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

pipe_smote = ImbPipeline([
    ('smote',  SMOTE(random_state=42)),
    ('scaler', StandardScaler()),
    ('model',  LogisticRegression())
])
pipe_smote.fit(X_train, y_train)
```

---

## 16. Bias-Variance & Learning Curves

```python
from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(model, X, y, title="Learning Curve"):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring='roc_auc',
        n_jobs=-1
    )
    
    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)
    
    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_mean, 'o-', color='blue',  label='Train AUC')
    plt.plot(train_sizes, val_mean,   'o-', color='orange', label='Val AUC')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.fill_between(train_sizes, val_mean - val_std,   val_mean + val_std,   alpha=0.1)
    plt.xlabel("Training Size"); plt.ylabel("ROC-AUC")
    plt.title(title); plt.legend(); plt.grid(True)
    plt.ylim(0.5, 1.05)
    plt.show()

plot_learning_curve(pipe, X, y, "Logistic Regression Learning Curve")
```

**Diagnosis:**

| Pattern | Problem | Fix |
|---------|---------|-----|
| Both train & val AUC low | High Bias (underfitting) | Add features, polynomial features, reduce C |
| Train AUC >> Val AUC | High Variance (overfitting) | Reduce C (more regularization), more data |
| Val AUC keeps improving with data | More data will help | Collect more data |
| Val AUC plateaus early | Model capacity too low | More complex features |

---

## 17. Common Interview Questions

**Q: Why is Logistic Regression called "regression" if it's a classifier?**
> Because it models a continuous quantity — the probability of class membership — using regression. The discrete classification decision is a post-processing step (thresholding).

**Q: Why can't we use MSE as the cost function for Logistic Regression?**
> With the sigmoid nonlinearity, MSE creates a non-convex cost surface with many local minima, making gradient descent unreliable. Log loss (cross-entropy) is convex with a single global minimum.

**Q: Explain the sigmoid function and why we use it.**
> Sigmoid maps any real number to (0, 1), giving a valid probability interpretation. Its S-shape represents the idea that probability changes slowly for extreme inputs and fast near the decision boundary.

**Q: What is the decision boundary in Logistic Regression?**
> The hyperplane where $\mathbf{w}^T\mathbf{x} + b = 0$, i.e., where the predicted probability equals 0.5. Points on one side are classified as 1, the other as 0.

**Q: What does the parameter C control? What happens as C → 0 and C → ∞?**
> C is inverse regularization strength ($C = 1/\lambda$). As C → 0: strong regularization, weights shrink to zero, simpler boundary (underfitting). As C → ∞: no regularization, model fits training data perfectly (may overfit).

**Q: What is the difference between Precision and Recall? When would you prioritize each?**
> Precision: of all predicted positives, how many are correct? Recall: of all actual positives, how many did we find? Prioritize precision when FP is costly (spam filter — don't want to miss real emails). Prioritize recall when FN is costly (cancer screening — don't want to miss disease).

**Q: What is ROC-AUC? Why use it over accuracy?**
> ROC-AUC measures the model's ability to rank positives higher than negatives across all thresholds. Unlike accuracy, it's threshold-independent and handles class imbalance better. AUC = 0.5 is random guessing; AUC = 1.0 is perfect.

**Q: What is Log Loss and what does it penalize?**
> Log Loss (cross-entropy) measures the quality of predicted probabilities. It penalizes confident wrong predictions exponentially — predicting 0.99 probability for the wrong class incurs a very high loss.

**Q: How does Logistic Regression handle multiclass problems?**
> Two approaches: OvR (One-vs-Rest) trains K binary classifiers, predicts the class with highest score. Multinomial (Softmax) trains a single model with K sets of weights, using softmax to get a proper probability distribution over all K classes.

**Q: Is Logistic Regression a parametric or non-parametric model?**
> Parametric — it assumes a specific functional form (linear relationship between features and log-odds), and the number of parameters is fixed regardless of training data size.

**Q: What are the log-odds and how do they relate to Logistic Regression?**
> Log-odds (logit) = $\log\frac{p}{1-p}$. Logistic Regression models log-odds as a linear function of features: $\log\frac{p}{1-p} = \mathbf{w}^T\mathbf{x} + b$. This is the fundamental linearity assumption.

**Q: How does Logistic Regression differ from a Linear SVM?**
> Both find a linear decision boundary. LogReg outputs calibrated probabilities and minimizes log loss. SVM maximizes the margin and doesn't directly output probabilities. LogReg is more interpretable; SVM is more robust to outliers via the margin.

**Q: What is gradient descent convergence and how do you detect it?**
> When the loss stops decreasing meaningfully between iterations: $|J^{(t)} - J^{(t-1)}| < \epsilon$. Monitor training loss curves — if they plateau, you've converged. If loss increases, reduce learning rate.

---

## 18. Resources

### 📺 Andrew Ng — ML Specialization (Coursera)
- **Course 1, Week 3:** Logistic Regression — classification, sigmoid, decision boundary, log loss, gradient descent
- **Key videos:** "Classification with Logistic Regression", "Cost function for Logistic Regression", "Simplified loss function", "Gradient Descent Implementation"
- [https://www.coursera.org/specializations/machine-learning-introduction](https://www.coursera.org/specializations/machine-learning-introduction)

### 📺 StatQuest with Josh Starmer (YouTube)
- [Logistic Regression, Clearly Explained!!!](https://www.youtube.com/watch?v=yIYKR4sgzI8)
- [Logistic Regression Details Pt 1: Coefficients](https://www.youtube.com/watch?v=vN5cNN2-HWE)
- [Logistic Regression Details Pt 2: Maximum Likelihood](https://www.youtube.com/watch?v=BfKanl1aSG0)
- [Logistic Regression Details Pt 3: R-squared and p-values](https://www.youtube.com/watch?v=xxFYro8QuXA)
- [ROC and AUC Clearly Explained](https://www.youtube.com/watch?v=4jRBRDbJemM)
- [The Confusion Matrix](https://www.youtube.com/watch?v=Kdsp6soqA7o)

### 📖 Hands-On Machine Learning (Géron) — O'Reilly
- **Chapter 3:** Classification — confusion matrix, precision/recall, ROC-AUC, multiclass
- **Chapter 4:** Training Linear Models — Logistic Regression, Softmax Regression
- GitHub notebooks: [https://github.com/ageron/handson-ml3](https://github.com/ageron/handson-ml3)

### 📖 Introduction to Statistical Learning with Python (ISLP)
- **Chapter 4:** Classification — Logistic Regression, LDA, multiclass, model comparison
- Free PDF + Python labs: [https://www.statlearning.com/](https://www.statlearning.com/)

### 📚 Scikit-Learn Documentation
- [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)
- [Linear Models User Guide](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
- [Classification Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)
- [Calibration](https://scikit-learn.org/stable/modules/calibration.html)

### 🔗 Additional
- [CS229 Lecture Notes — Logistic Regression (Andrew Ng, Stanford)](https://cs229.stanford.edu/lectures-spring2022/main_notes.pdf)
- [3Blue1Brown — Neural Networks Series (sigmoid intuition)](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)

---

## 🗺️ Quick Reference Cheatsheet

```
LOGISTIC REGRESSION WORKFLOW
══════════════════════════════════════════════════════════════

1. EXPLORE DATA
   ├── Check class balance (value_counts, bar plot)
   ├── Feature distributions by class
   ├── Correlation matrix
   └── Handle missing values

2. PREPROCESS
   ├── Scale features → StandardScaler (required for GD solvers)
   ├── Encode categoricals (OneHot)
   ├── Stratified split: train_test_split(..., stratify=y)
   └── Handle imbalance if needed

3. TRAIN
   ├── Start: LogisticRegression(C=1, solver='lbfgs')
   ├── Tune C via GridSearchCV (log scale)
   ├── Try L1 for feature selection
   └── Large data: solver='saga'

4. EVALUATE
   ├── Classification report (precision, recall, F1)
   ├── ROC-AUC (ranking ability)
   ├── Confusion matrix
   ├── PR curve if imbalanced
   └── Log Loss if probability quality matters

5. TUNE THRESHOLD
   ├── Default = 0.5
   ├── Lower if recall matters (disease, fraud)
   ├── Raise if precision matters (spam)
   └── Optimize F1 or business metric

6. DIAGNOSE
   ├── Learning curves → bias/variance
   ├── High bias: reduce C, add features
   └── High variance: increase C, get more data

KEY FORMULAS SNAPSHOT
══════════════════════════════════════════════════════════════
Sigmoid:        σ(z) = 1 / (1 + e⁻ᶻ)
Prediction:     ŷ = σ(Xw + b)
Log Loss:       J = -(1/m) Σ [y·log(ŷ) + (1-y)·log(1-ŷ)]
Gradient w:     ∂J/∂w = (1/m) Xᵀ(ŷ - y)
Gradient b:     ∂J/∂b = (1/m) Σ(ŷ - y)
Decision bdry:  wᵀx + b = 0
Regularization: C = 1/λ (smaller C = stronger regularization)
Log-odds:       log(p/(1-p)) = wᵀx + b
Softmax:        P(y=k|x) = exp(wₖᵀx) / Σ exp(wⱼᵀx)

SOLVER CHEATSHEET
══════════════════════════════════════════════════════════════
Small data, L2 only     → lbfgs (default)
Small data, L1          → liblinear
Large data, any penalty → saga
Multiclass + L1/EN      → saga
n_samples >> n_features → newton-cholesky
```

---

*Notes compiled for ML/DL job readiness. Sources: Andrew Ng Coursera ML Specialization (Week 3), Hands-On ML Ch. 3–4 (Géron), StatQuest (Starmer), ISLP Ch. 4 (James et al.), Scikit-Learn documentation.*
