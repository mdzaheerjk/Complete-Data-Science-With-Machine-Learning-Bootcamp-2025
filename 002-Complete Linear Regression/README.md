# 📐 Linear Regression — Complete ML/DL Job-Ready Notes

> **Sources:** Andrew Ng ML Specialization (Coursera) · Hands-On ML with Scikit-Learn (Aurélien Géron) · StatQuest (Josh Starmer) · Scikit-Learn Docs · ISLP (James et al.)

---

## 📚 Table of Contents

1. [What is Linear Regression?](#1-what-is-linear-regression)
2. [Types](#2-types)
3. [The Math](#3-the-math)
4. [Cost Function (MSE / SSE)](#4-cost-function)
5. [Gradient Descent — Full Deep Dive](#5-gradient-descent)
6. [Normal Equation (Closed-Form)](#6-normal-equation)
7. [Assumptions](#7-assumptions)
8. [Evaluation Metrics](#8-evaluation-metrics)
9. [Python from Scratch](#9-python-from-scratch)
10. [Scikit-Learn Implementation](#10-scikit-learn-implementation)
11. [Hyperparameters & Tuning](#11-hyperparameters--tuning)
12. [Regularization (Ridge, Lasso, ElasticNet)](#12-regularization)
13. [Feature Scaling & Preprocessing](#13-feature-scaling--preprocessing)
14. [Polynomial Regression](#14-polynomial-regression)
15. [Bias-Variance Tradeoff](#15-bias-variance-tradeoff)
16. [Common Interview Questions](#16-common-interview-questions)
17. [Resources](#17-resources)

---

## 1. What is Linear Regression?

> **Simple Statement:** Linear Regression tries to draw the **best-fit straight line** through data points to predict a continuous number.

- **Task type:** Supervised Learning → Regression
- **Output:** Continuous value (e.g., house price, salary, temperature)
- **Goal:** Find the line (or hyperplane) that minimizes prediction error

**Real-world examples:**
- Predicting house prices from area (sqft)
- Predicting salary from years of experience
- Predicting CO₂ emissions from engine size

---

## 2. Types

| Type | Description | Example |
|------|-------------|---------|
| **Simple Linear Regression** | 1 input feature, 1 output | Area → Price |
| **Multiple Linear Regression** | n input features, 1 output | Area + Rooms + Location → Price |
| **Polynomial Regression** | Non-linear via degree expansion | Curved relationship |
| **Ridge Regression** | L2 regularization | Prevent overfitting |
| **Lasso Regression** | L1 regularization + feature selection | Sparse models |
| **ElasticNet** | L1 + L2 combined | Best of both |

---

## 3. The Math

### Simple Linear Regression

$$\hat{y} = w \cdot x + b$$

- $\hat{y}$ = predicted value
- $x$ = input feature
- $w$ = weight (slope) — how much $y$ changes per unit of $x$
- $b$ = bias (intercept) — value of $y$ when $x = 0$

### Multiple Linear Regression

$$\hat{y} = w_1x_1 + w_2x_2 + \cdots + w_nx_n + b$$

**Vector form (compact):**

$$\hat{y} = \mathbf{w}^T \mathbf{x} + b$$

Or with bias absorbed into weight vector:

$$\hat{y} = \boldsymbol{\theta}^T \mathbf{x}$$

where $\boldsymbol{\theta} = [b, w_1, w_2, \ldots, w_n]^T$ and $\mathbf{x} = [1, x_1, x_2, \ldots, x_n]^T$

**Matrix form (entire dataset):**

$$\hat{\mathbf{y}} = \mathbf{X} \boldsymbol{\theta}$$

- $\mathbf{X}$ is shape $(m \times (n+1))$ — $m$ examples, $n$ features + bias column of 1s
- $\boldsymbol{\theta}$ is shape $((n+1) \times 1)$

---

## 4. Cost Function

### Mean Squared Error (MSE)

> **Simple Statement:** Measure how far off your predictions are from true values. We square the errors so negatives don't cancel positives, and large errors are punished more.

$$J(\mathbf{w}, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right)^2$$

- $m$ = number of training examples
- The $\frac{1}{2}$ is a convenience trick — it cancels when we differentiate
- Also written as $\text{MSE} = \frac{1}{m} \sum (\hat{y}_i - y_i)^2$

**Why MSE and not MAE?**
- MSE is differentiable everywhere → works with gradient descent
- MSE penalizes large errors more heavily (squared)
- MAE is more robust to outliers but harder to optimize

**Matrix form:**

$$J(\boldsymbol{\theta}) = \frac{1}{2m} (\mathbf{X}\boldsymbol{\theta} - \mathbf{y})^T (\mathbf{X}\boldsymbol{\theta} - \mathbf{y})$$

---

## 5. Gradient Descent

> **Simple Statement:** Imagine standing on a hilly landscape (the cost surface). You look around, find the steepest downhill direction, take a small step that way. Repeat until you reach the valley (minimum cost).

### 5.1 The Core Idea

We want to minimize $J(\mathbf{w}, b)$. We do this by iteratively updating parameters in the **direction of steepest descent** of the cost function.

$$w_j := w_j - \alpha \frac{\partial J}{\partial w_j}$$
$$b := b - \alpha \frac{\partial J}{\partial b}$$

- $\alpha$ = **learning rate** (how big each step is)
- $\frac{\partial J}{\partial w_j}$ = **gradient** (direction of steepest ascent, so we subtract)

### 5.2 Deriving the Gradients

Starting from:
$$J = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2 = \frac{1}{2m} \sum_{i=1}^{m} (w \cdot x^{(i)} + b - y^{(i)})^2$$

**Gradient w.r.t. $w_j$** (chain rule):

$$\frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} \left(\hat{y}^{(i)} - y^{(i)}\right) x_j^{(i)}$$

**Gradient w.r.t. $b$:**

$$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} \left(\hat{y}^{(i)} - y^{(i)}\right)$$

> **Key Insight:** The gradient is just the **average error** times the **feature value**. If predictions are too high, we decrease $w$. If predictions are too low, we increase $w$.

### 5.3 Update Rules (Simultaneous Update!)

```
IMPORTANT: Always compute ALL gradients FIRST, then update ALL parameters.
Never update w, then use the updated w to compute the gradient for b.
```

$$w_j := w_j - \alpha \cdot \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)}) x_j^{(i)}$$

$$b := b - \alpha \cdot \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})$$

### 5.4 Types of Gradient Descent

| Type | Batch Size | Update Frequency | Pros | Cons |
|------|-----------|-----------------|------|------|
| **Batch GD** | All $m$ examples | Once per epoch | Stable, guaranteed convergence | Slow on large data |
| **Stochastic GD (SGD)** | 1 example | Every example | Fast, online learning | Noisy, oscillates |
| **Mini-Batch GD** | $k$ examples (32–512) | Every mini-batch | Best of both | Need to tune batch size |

### 5.5 Learning Rate $\alpha$ — Critical!

| Learning Rate | Behavior |
|--------------|----------|
| Too **large** | Overshoots minimum, diverges, cost increases |
| Too **small** | Converges very slowly, wastes compute |
| **Just right** | Smooth, steady decrease in cost |

**How to pick $\alpha$:** Try values like `0.001, 0.003, 0.01, 0.03, 0.1, 0.3` (each ~3x the last). Plot learning curves.

**Signs of convergence:** Cost decreases and flattens. If cost ever **increases**, lower $\alpha$.

### 5.6 Convergence Test

$$|J^{(t)} - J^{(t-1)}| < \epsilon \quad \text{(e.g., } \epsilon = 10^{-6})$$

### 5.7 Vectorized Gradient Descent (Matrix Form)

$$\boldsymbol{\theta} := \boldsymbol{\theta} - \frac{\alpha}{m} \mathbf{X}^T (\mathbf{X}\boldsymbol{\theta} - \mathbf{y})$$

This is much faster than loops — uses NumPy BLAS operations under the hood.

---

## 6. Normal Equation (Closed-Form)

> **Simple Statement:** Instead of iterating, solve for the optimal weights directly using linear algebra. Set gradient = 0 and solve.

Setting $\nabla_\theta J = 0$:

$$\mathbf{X}^T \mathbf{X} \boldsymbol{\theta} = \mathbf{X}^T \mathbf{y}$$

$$\boxed{\boldsymbol{\theta} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}}$$

**Pros:**
- No learning rate to tune
- No iterations
- Exact solution in one step

**Cons:**
- Computing $(\mathbf{X}^T \mathbf{X})^{-1}$ is $O(n^3)$ — slow when $n$ (features) is large
- Matrix may be **singular** (non-invertible) if features are linearly dependent
- Doesn't scale to millions of features (use GD instead)

**Rule of thumb:** Use Normal Equation when $n < 10{,}000$ features. Otherwise use gradient descent.

**Pseudo-inverse (Moore-Penrose):** scikit-learn uses SVD decomposition instead of direct inverse → numerically stable.

---

## 7. Assumptions

> Linear regression has 5 key assumptions. Violating them doesn't break the model, but does affect reliability of predictions and coefficients.

| # | Assumption | What it means | How to check |
|---|-----------|---------------|-------------|
| 1 | **Linearity** | Relationship between X and y is linear | Scatter plots, residual vs fitted plot |
| 2 | **Independence** | Observations are independent of each other | Domain knowledge, Durbin-Watson test |
| 3 | **Homoscedasticity** | Residuals have constant variance | Residual plot (should be random cloud) |
| 4 | **Normality of residuals** | Residuals follow normal distribution | QQ-plot, histogram of residuals |
| 5 | **No multicollinearity** | Features are not highly correlated with each other | Correlation matrix, VIF score |

**Remember with acronym:** **LINE + M** (Linearity, Independence, Normality, Equal-variance + no Multicollinearity)

---

## 8. Evaluation Metrics

### Mean Absolute Error (MAE)
$$\text{MAE} = \frac{1}{m} \sum_{i=1}^m |y_i - \hat{y}_i|$$
- Same unit as $y$
- Robust to outliers
- Not differentiable at 0

### Mean Squared Error (MSE)
$$\text{MSE} = \frac{1}{m} \sum_{i=1}^m (y_i - \hat{y}_i)^2$$
- Penalizes large errors more
- Not in same unit as $y$

### Root Mean Squared Error (RMSE)
$$\text{RMSE} = \sqrt{\text{MSE}}$$
- Same unit as $y$ ✅
- Most commonly reported

### R² Score (Coefficient of Determination)
$$R^2 = 1 - \frac{\text{SS}_\text{res}}{\text{SS}_\text{tot}} = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$

- Range: $(-\infty, 1]$
- $R^2 = 1$: perfect fit
- $R^2 = 0$: model is no better than just predicting the mean
- $R^2 < 0$: model is worse than mean prediction (overfitting or wrong model)

> **Simple Statement:** R² tells you what **percentage of variance** in $y$ is explained by your model.

### Adjusted R²
$$\bar{R}^2 = 1 - (1 - R^2) \frac{m-1}{m-p-1}$$
- $p$ = number of features
- Penalizes for adding useless features
- Always use Adjusted R² for multiple regression

---

## 9. Python from Scratch

### 9.1 Simple Linear Regression — Step by Step

```python
import numpy as np
import matplotlib.pyplot as plt

# ── 1. Generate toy data ──────────────────────────────────────────
np.random.seed(42)
m = 100
X = 2 * np.random.rand(m, 1)          # shape (100, 1)
y = 4 + 3 * X + np.random.randn(m, 1) # true: w=3, b=4

# ── 2. Cost Function ──────────────────────────────────────────────
def compute_cost(X, y, w, b):
    m = len(y)
    predictions = X * w + b
    errors = predictions - y
    cost = (1 / (2 * m)) * np.sum(errors ** 2)
    return cost

# ── 3. Gradient Computation ───────────────────────────────────────
def compute_gradients(X, y, w, b):
    m = len(y)
    predictions = X * w + b
    errors = predictions - y           # shape (m, 1)
    
    dw = (1 / m) * np.sum(errors * X)  # scalar
    db = (1 / m) * np.sum(errors)      # scalar
    return dw, db

# ── 4. Gradient Descent ───────────────────────────────────────────
def gradient_descent(X, y, w_init=0, b_init=0, alpha=0.1, n_iter=1000):
    w, b = w_init, b_init
    cost_history = []
    
    for i in range(n_iter):
        dw, db = compute_gradients(X, y, w, b)
        
        # Simultaneous update
        w = w - alpha * dw
        b = b - alpha * db
        
        cost = compute_cost(X, y, w, b)
        cost_history.append(cost)
        
        if i % 100 == 0:
            print(f"Iter {i:4d} | Cost: {cost:.6f} | w: {w:.4f} | b: {b:.4f}")
    
    return w, b, cost_history

# ── 5. Train ──────────────────────────────────────────────────────
w, b, cost_history = gradient_descent(X, y, alpha=0.1, n_iter=1000)
print(f"\nFinal: w = {w:.4f}, b = {b:.4f}")
# Expected: w ≈ 3.0, b ≈ 4.0

# ── 6. Plot Learning Curve ────────────────────────────────────────
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(cost_history)
plt.xlabel("Iteration"); plt.ylabel("Cost (MSE/2)")
plt.title("Learning Curve"); plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(X, y, alpha=0.5, label="Data")
X_line = np.linspace(0, 2, 100).reshape(-1, 1)
plt.plot(X_line, w * X_line + b, 'r-', linewidth=2, label=f"Fit: y={w:.2f}x+{b:.2f}")
plt.xlabel("X"); plt.ylabel("y")
plt.title("Best-Fit Line"); plt.legend(); plt.grid(True)
plt.tight_layout()
plt.show()
```

### 9.2 Multiple Linear Regression — Vectorized

```python
import numpy as np

class LinearRegressionGD:
    """
    Linear Regression using Gradient Descent (vectorized, NumPy only).
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000, verbose=False):
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.verbose = verbose
        self.weights = None   # shape (n_features,)
        self.bias = None      # scalar
        self.cost_history = []
    
    def fit(self, X, y):
        m, n = X.shape   # m=samples, n=features
        
        # Initialize parameters
        self.weights = np.zeros(n)
        self.bias = 0.0
        
        for i in range(self.n_iter):
            # Forward pass: predictions
            y_pred = X @ self.weights + self.bias   # (m,)
            
            # Compute errors
            errors = y_pred - y                      # (m,)
            
            # Compute gradients (vectorized!)
            dw = (1 / m) * (X.T @ errors)           # (n,)
            db = (1 / m) * np.sum(errors)            # scalar
            
            # Update parameters (simultaneous)
            self.weights -= self.lr * dw
            self.bias    -= self.lr * db
            
            # Track cost
            cost = (1 / (2 * m)) * np.sum(errors ** 2)
            self.cost_history.append(cost)
            
            if self.verbose and i % 100 == 0:
                print(f"Iter {i:5d} | Cost: {cost:.6f}")
        
        return self
    
    def predict(self, X):
        return X @ self.weights + self.bias
    
    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot  # R²


# ── Usage ─────────────────────────────────────────────────────────
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = make_regression(n_samples=1000, n_features=5, noise=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# IMPORTANT: scale features before gradient descent
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

model = LinearRegressionGD(learning_rate=0.1, n_iterations=1000, verbose=True)
model.fit(X_train_s, y_train)
print(f"R² on test: {model.score(X_test_s, y_test):.4f}")
```

### 9.3 Normal Equation from Scratch

```python
def normal_equation(X, y):
    """
    θ = (XᵀX)⁻¹ Xᵀy
    Uses pseudo-inverse for numerical stability.
    """
    # Add bias column of 1s
    X_b = np.c_[np.ones((len(X), 1)), X]
    
    # Closed-form solution
    theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
    # OR simply: theta = np.linalg.lstsq(X_b, y, rcond=None)[0]
    
    b = theta[0]
    w = theta[1:]
    return w, b

w, b = normal_equation(X_train, y_train)
print(f"Weights: {w}")
print(f"Bias:    {b:.4f}")
```

---

## 10. Scikit-Learn Implementation

### 10.1 Basic Pipeline

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# ── Load / prepare data ───────────────────────────────────────────
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
X, y = housing.data, housing.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Build pipeline ────────────────────────────────────────────────
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  LinearRegression())  
])

# ── Fit ───────────────────────────────────────────────────────────
pipe.fit(X_train, y_train)

# ── Predict ───────────────────────────────────────────────────────
y_pred = pipe.predict(X_test)

# ── Evaluate ──────────────────────────────────────────────────────
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"R²   : {r2:.4f}")

# ── Cross-validation ──────────────────────────────────────────────
cv_scores = cross_val_score(pipe, X, y, cv=5, scoring='neg_root_mean_squared_error')
print(f"\nCV RMSE: {-cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── Model Coefficients ────────────────────────────────────────────
model = pipe.named_steps['model']
print("\nCoefficients:", model.coef_)
print("Intercept:   ", model.intercept_)
```

### 10.2 Residual Analysis

```python
import matplotlib.pyplot as plt

residuals = y_test - y_pred

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Residuals vs Fitted
axes[0].scatter(y_pred, residuals, alpha=0.3)
axes[0].axhline(0, color='red', linestyle='--')
axes[0].set_xlabel("Fitted values"); axes[0].set_ylabel("Residuals")
axes[0].set_title("Residuals vs Fitted")

# Histogram of residuals
axes[1].hist(residuals, bins=50, edgecolor='black')
axes[1].set_xlabel("Residual"); axes[1].set_title("Residual Distribution")

# QQ Plot
from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[2])
axes[2].set_title("QQ Plot (Normality Check)")

plt.tight_layout()
plt.show()
```

### 10.3 Feature Importance Visualization

```python
import pandas as pd
import matplotlib.pyplot as plt

feature_names = housing.feature_names
coefs = pd.Series(model.coef_, index=feature_names).sort_values()

plt.figure(figsize=(8, 5))
coefs.plot(kind='barh', color=['red' if c < 0 else 'steelblue' for c in coefs])
plt.axvline(0, color='black', linewidth=0.8)
plt.xlabel("Coefficient Value")
plt.title("Feature Coefficients (after StandardScaling = comparable)")
plt.tight_layout()
plt.show()
```

---

## 11. Hyperparameters & Tuning

### LinearRegression (sklearn)

| Parameter | Default | Description | When to change |
|-----------|---------|-------------|----------------|
| `fit_intercept` | `True` | Whether to compute bias $b$ | Set `False` if data is already centered |
| `copy_X` | `True` | Copy X before fitting | `False` to save memory (modifies X in-place) |
| `n_jobs` | `None` | Parallel jobs for computation | Set `-1` to use all CPUs on large datasets |
| `positive` | `False` | Force all coefficients ≥ 0 | When negative coefs are physically impossible |

### SGDRegressor (GD-based, for large datasets)

| Parameter | Default | Description | Recommended Range |
|-----------|---------|-------------|------------------|
| `loss` | `'squared_error'` | Loss function | `'huber'` for outlier robustness |
| `penalty` | `'l2'` | Regularization type | `'l1'`, `'l2'`, `'elasticnet'`, `None` |
| `alpha` | `0.0001` | Regularization strength | `1e-5` to `1e0` (log scale) |
| `l1_ratio` | `0.15` | Mix for ElasticNet | `0` to `1` |
| `max_iter` | `1000` | Max epochs | `500` to `5000` |
| `tol` | `1e-3` | Convergence tolerance | `1e-4` to `1e-2` |
| `learning_rate` | `'invscaling'` | LR schedule | `'constant'`, `'optimal'`, `'adaptive'` |
| `eta0` | `0.01` | Initial learning rate | `1e-4` to `1e-1` |
| `early_stopping` | `False` | Stop if val score doesn't improve | `True` for faster training |
| `validation_fraction` | `0.1` | Val set size for early stopping | `0.1` to `0.2` |
| `n_iter_no_change` | `5` | Patience for early stopping | `5` to `20` |
| `warm_start` | `False` | Reuse previous fit | `True` for incremental learning |
| `shuffle` | `True` | Shuffle data each epoch | Keep `True` for SGD |
| `random_state` | `None` | Reproducibility | Set any integer |

```python
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('sgd', SGDRegressor(random_state=42))
])

param_grid = {
    'sgd__alpha':         [0.0001, 0.001, 0.01, 0.1],
    'sgd__learning_rate': ['constant', 'invscaling', 'adaptive'],
    'sgd__eta0':          [0.001, 0.01, 0.1],
    'sgd__max_iter':      [1000, 2000],
}

gs = GridSearchCV(pipe, param_grid, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
gs.fit(X_train, y_train)

print("Best params:", gs.best_params_)
print("Best RMSE: ", -gs.best_score_)
```

---

## 12. Regularization

> **Simple Statement:** Regularization adds a penalty to the cost function for large weights, preventing the model from memorizing training data (overfitting).

### 12.1 Ridge Regression (L2)

$$J(\boldsymbol{\theta}) = \text{MSE} + \alpha \sum_{j=1}^{n} w_j^2$$

- Shrinks all weights toward 0 but never exactly to 0
- Works well when many features have small effects
- **Closed form:** $\boldsymbol{\theta} = (\mathbf{X}^T\mathbf{X} + \alpha\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$
- The $\alpha\mathbf{I}$ term also fixes the singularity problem!

```python
from sklearn.linear_model import Ridge, RidgeCV
import numpy as np

# Manual alpha selection
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# Or use RidgeCV (built-in cross-validation)
alphas = np.logspace(-3, 3, 100)
ridge_cv = RidgeCV(alphas=alphas, cv=5, scoring='neg_mean_squared_error')
ridge_cv.fit(X_train, y_train)
print(f"Best alpha: {ridge_cv.alpha_}")
print(f"R²: {ridge_cv.score(X_test, y_test):.4f}")
```

### 12.2 Lasso Regression (L1)

$$J(\boldsymbol{\theta}) = \text{MSE} + \alpha \sum_{j=1}^{n} |w_j|$$

- Can shrink weights to **exactly 0** → automatic feature selection
- Produces **sparse** models
- Useful when you suspect only a few features matter

```python
from sklearn.linear_model import Lasso, LassoCV

lasso_cv = LassoCV(cv=5, random_state=42, max_iter=10000)
lasso_cv.fit(X_train, y_train)
print(f"Best alpha: {lasso_cv.alpha_}")

# Check which features were zeroed out
import pandas as pd
coef_df = pd.DataFrame({
    'feature': housing.feature_names,
    'coefficient': lasso_cv.coef_
})
print(coef_df[coef_df.coefficient != 0])  # surviving features
```

### 12.3 ElasticNet (L1 + L2)

$$J(\boldsymbol{\theta}) = \text{MSE} + r \cdot \alpha \sum|w_j| + \frac{1-r}{2} \cdot \alpha \sum w_j^2$$

- `l1_ratio` $r=1$ → pure Lasso; $r=0$ → pure Ridge
- Best when you want both feature selection AND grouping of correlated features

```python
from sklearn.linear_model import ElasticNet, ElasticNetCV

en_cv = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
                     cv=5, random_state=42, max_iter=10000)
en_cv.fit(X_train, y_train)
print(f"Best alpha: {en_cv.alpha_}, l1_ratio: {en_cv.l1_ratio_}")
```

### 12.4 Comparison Table

| | Ridge | Lasso | ElasticNet |
|--|-------|-------|------------|
| Penalty | $\sum w_j^2$ | $\sum |w_j|$ | Mix of both |
| Feature selection | ❌ | ✅ | ✅ |
| Handles correlated features | ✅ | ❌ (picks one) | ✅ |
| Sparse solution | ❌ | ✅ | ✅ |
| Sklearn class | `Ridge` | `Lasso` | `ElasticNet` |

---

## 13. Feature Scaling & Preprocessing

> **Simple Statement:** Gradient descent works much better when all features are on the same scale. Otherwise, the cost function is elongated and GD bounces around.

### Why Scaling Matters

- Feature $x_1$ ∈ [0, 5000] (area in sqft)
- Feature $x_2$ ∈ [0, 5] (number of rooms)
- Without scaling: $w_1$ needs tiny updates, $w_2$ needs large updates → uneven gradients

### StandardScaler (Z-score normalization)

$$x' = \frac{x - \mu}{\sigma}$$

- Mean = 0, Std = 1
- Best for: Gaussian-ish features, Ridge regression
- Not bounded

### MinMaxScaler (Min-Max normalization)

$$x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

- Range [0, 1]
- Sensitive to outliers
- Best for: Neural networks, bounded features

### RobustScaler

$$x' = \frac{x - \text{median}}{\text{IQR}}$$

- Best for: Data with outliers

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline

# CRITICAL RULE: Fit scaler on TRAIN set only. Transform both train and test.
# Using Pipeline prevents data leakage automatically.

pipe = Pipeline([
    ('scaler', StandardScaler()),   # swap in MinMaxScaler() or RobustScaler() as needed
    ('model', LinearRegression())
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
```

---

## 14. Polynomial Regression

> **Simple Statement:** Transform features into polynomial features (e.g., $x, x^2, x^3$) to fit curved relationships, then apply linear regression on the transformed features.

$$\hat{y} = w_0 + w_1 x + w_2 x^2 + w_3 x^3 + \cdots$$

This is still **linear** in the parameters (weights)!

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import validation_curve
import numpy as np
import matplotlib.pyplot as plt

# Generate non-linear data
np.random.seed(42)
X = np.sort(np.random.rand(100, 1) * 6 - 3, axis=0)
y = 0.5 * X.ravel() ** 2 + X.ravel() + 2 + np.random.randn(100)

# Compare degrees
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, degree in zip(axes, [1, 2, 10]):
    pipe = Pipeline([
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])
    pipe.fit(X, y)
    
    X_plot = np.linspace(-3, 3, 300).reshape(-1, 1)
    y_plot = pipe.predict(X_plot)
    
    ax.scatter(X, y, alpha=0.5, s=20)
    ax.plot(X_plot, y_plot, 'r-', linewidth=2)
    ax.set_title(f"Degree {degree} | R²={pipe.score(X, y):.3f}")
    ax.grid(True)

plt.tight_layout()
plt.show()

# ── Choose best degree via cross-validation ───────────────────────
degrees = range(1, 11)
train_scores, val_scores = [], []

for d in degrees:
    pipe = Pipeline([
        ('poly', PolynomialFeatures(degree=d, include_bias=False)),
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])
    cv = cross_val_score(pipe, X, y, cv=5, scoring='neg_rmse' if False else 'r2')
    val_scores.append(cv.mean())

best_degree = degrees[np.argmax(val_scores)]
print(f"Best degree: {best_degree}")
```

---

## 15. Bias-Variance Tradeoff

> **Simple Statement:** 
> - **High Bias (Underfitting):** Model is too simple, misses patterns. Bad on train AND test.
> - **High Variance (Overfitting):** Model memorizes training data, fails to generalize. Great on train, bad on test.
> - **Goal:** Find the sweet spot.

$$\text{Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Noise}$$

| Symptom | Diagnosis | Fix |
|---------|-----------|-----|
| High train error, high test error | High Bias | More features, higher degree, less regularization |
| Low train error, high test error | High Variance | More data, regularization, fewer features, lower degree |
| Both errors high and close | Underfitting | More complex model |
| Train ≪ Test error | Overfitting | Regularize, get more data |

### Learning Curves (Diagnostic Tool)

```python
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curve(model, X, y, title="Learning Curve"):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )
    
    train_rmse = -train_scores.mean(axis=1)
    val_rmse   = -val_scores.mean(axis=1)
    
    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_rmse, 'o-', color='blue',  label='Train RMSE')
    plt.plot(train_sizes, val_rmse,   'o-', color='orange', label='Val RMSE')
    plt.fill_between(train_sizes,
                     train_rmse - (-train_scores).std(axis=1),
                     train_rmse + (-train_scores).std(axis=1), alpha=0.1)
    plt.xlabel("Training Size"); plt.ylabel("RMSE")
    plt.title(title); plt.legend(); plt.grid(True)
    plt.show()

# Underfit model (degree 1)
pipe_linear = Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())])
plot_learning_curve(pipe_linear, X, y, "Linear (degree 1) — Underfitting?")

# Overfit model (degree 10)
pipe_poly10 = Pipeline([
    ('poly', PolynomialFeatures(10, include_bias=False)),
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])
plot_learning_curve(pipe_poly10, X, y, "Polynomial (degree 10) — Overfitting?")
```

**How to read learning curves:**
- If **both curves converge high** → High Bias (underfitting) → more complex model
- If **big gap between train and val** → High Variance (overfitting) → regularize or get more data
- **Good model:** val curve decreases and approaches train curve at reasonable RMSE

---

## 16. Common Interview Questions

### Conceptual

**Q: What is the difference between correlation and regression?**
> Correlation measures the strength and direction of a linear relationship between two variables (a single number between -1 and 1). Regression gives you the equation to predict one variable from another(s).

**Q: Why do we use MSE instead of absolute error for linear regression?**
> MSE is differentiable everywhere, making it compatible with gradient descent. It also penalizes large errors more heavily via squaring. MAE is not differentiable at 0 and treats all errors equally.

**Q: Can linear regression be used for classification?**
> Technically yes (predict 0 or 1 and threshold), but it's not ideal because predictions can go outside [0, 1], it's sensitive to outliers, and doesn't model probability correctly. Use Logistic Regression instead.

**Q: What happens if features are correlated (multicollinearity)?**
> Coefficients become unstable and hard to interpret. $(\mathbf{X}^T\mathbf{X})$ becomes near-singular. Standard errors inflate. Solution: remove correlated features, use PCA, or use Ridge regression.

**Q: When would you use Normal Equation vs Gradient Descent?**
> Normal Equation: small $n$ (< 10k features), no need to tune hyperparameters, exact solution. Gradient Descent: large $n$, large $m$, works online, scalable.

**Q: What is the effect of regularization on coefficients?**
> Both Ridge and Lasso shrink coefficients toward 0. Ridge never reaches exactly 0 (all features kept). Lasso can zero out features completely (feature selection). As $\alpha$ increases, more shrinkage occurs.

**Q: Why must we scale features before Gradient Descent?**
> Without scaling, the cost function contours are elongated ellipses. Gradient descent oscillates and converges slowly. After scaling, contours are circular, and GD converges in a straighter path.

**Q: What is R² and what are its limitations?**
> R² measures proportion of variance explained. Limitations: always increases when you add features (use Adjusted R²); doesn't indicate whether assumptions are met; can be high even if the model is wrong for extrapolation.

**Q: Explain Gradient Descent intuitively.**
> Like descending a mountain blindfolded: you feel the slope under your feet (gradient), step in the steepest downhill direction, repeat until you reach the valley (minimum cost).

**Q: What is the difference between batch, stochastic, and mini-batch gradient descent?**
> Batch GD computes gradient on all data — stable but slow. SGD uses one example per update — fast but noisy. Mini-batch uses a small batch (32–512) — the industry standard, balancing speed and stability.

---

## 17. Resources

### 📺 Andrew Ng — ML Specialization (Coursera)
- Course 1, Week 1–2: Linear regression, cost function, gradient descent
- **Key videos:** "Cost function intuition", "Gradient descent in practice"
- [https://www.coursera.org/specializations/machine-learning-introduction](https://www.coursera.org/specializations/machine-learning-introduction)
- Free audit available!

### 📺 StatQuest with Josh Starmer (YouTube)
- [Linear Regression, Clearly Explained!!!](https://www.youtube.com/watch?v=nk2CQITm_eo)
- [Gradient Descent, Step-by-Step](https://www.youtube.com/watch?v=sDv4f4s2SB8)
- [R-squared, Clearly Explained](https://www.youtube.com/watch?v=2AQKmw14mHM)
- [Ridge vs Lasso Regression](https://www.youtube.com/watch?v=Xm2C_gTAl8c)
- [Multiple Regression](https://www.youtube.com/watch?v=EkAQAi3a4js)

### 📖 Hands-On Machine Learning (Géron) — O'Reilly
- **Chapter 4:** Training Linear Models — gradient descent, normal equation, regularization
- GitHub: [https://github.com/ageron/handson-ml3](https://github.com/ageron/handson-ml3)
- Notebooks are free on Colab!

### 📖 Introduction to Statistical Learning with Python (ISLP)
- **Chapter 3:** Linear Regression — comprehensive statistical treatment
- Free PDF: [https://www.statlearning.com/](https://www.statlearning.com/)
- Python labs included

### 📚 Scikit-Learn Documentation
- [LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [SGDRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html)
- [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
- [Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
- [Linear Models User Guide](https://scikit-learn.org/stable/modules/linear_model.html)

### 🔗 Additional
- [3Blue1Brown — Essence of Calculus (for gradient intuition)](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)
- [CS229 Notes (Stanford) — Andrew Ng's original course notes (free)](https://cs229.stanford.edu/lectures-spring2022/main_notes.pdf)

---

## 🗺️ Quick Reference Cheatsheet

```
LINEAR REGRESSION WORKFLOW
══════════════════════════════════════════════════════

1. EXPLORE DATA
   ├── Check shape, dtypes, missing values
   ├── Plot distributions (hist, boxplot)
   ├── Correlation matrix / heatmap
   └── Scatter plots: y vs each feature

2. PREPROCESS
   ├── Handle missing values (impute)
   ├── Encode categoricals (OneHot, Ordinal)
   ├── Scale features → StandardScaler
   └── Split: train / val / test

3. TRAIN
   ├── LinearRegression (small n)
   ├── SGDRegressor (large dataset)
   └── Start simple, then regularize

4. EVALUATE
   ├── RMSE, MAE, R², Adjusted R²
   ├── Cross-validation (CV=5 or 10)
   ├── Learning curves
   └── Residual plots

5. TUNE
   ├── If High Bias: more features, higher degree poly
   ├── If High Variance: regularization (Ridge/Lasso)
   └── Grid/RandomSearchCV for hyperparameters

6. DEPLOY
   ├── Refit on full train+val data
   ├── Evaluate on held-out test set
   └── Package with Pipeline (scaler + model)

KEY FORMULAS SNAPSHOT
══════════════════════════════════════════════════════
Prediction:     ŷ = Xθ
Cost:           J = (1/2m) ||Xθ - y||²
GD Update:      θ := θ - (α/m) Xᵀ(Xθ - y)
Normal Eq:      θ = (XᵀX)⁻¹ Xᵀy
Ridge Cost:     J + α Σwⱼ²
Lasso Cost:     J + α Σ|wⱼ|
R²:             1 - SS_res / SS_tot
```

---

*Notes compiled for ML/DL job readiness. Sources: Andrew Ng Coursera ML Specialization, Hands-On ML (Géron), StatQuest (Starmer), ISLP (James et al.), Scikit-Learn documentation.*
