# 📈 Linear Regression — Complete ML/DL Notes

> **Topic:** Supervised Learning | **Type:** Regression | **Level:** Beginner → Advanced

---

## 📚 Table of Contents

1. [What is Linear Regression?](#1-what-is-linear-regression)
2. [Types of Linear Regression](#2-types-of-linear-regression)
3. [The Math Behind It](#3-the-math-behind-it)
4. [Cost Function (Loss)](#4-cost-function-loss)
5. [Gradient Descent](#5-gradient-descent)
6. [Normal Equation (Closed-Form)](#6-normal-equation-closed-form)
7. [Assumptions](#7-assumptions)
8. [Evaluation Metrics](#8-evaluation-metrics)
9. [Hyperparameters](#9-hyperparameters)
10. [Regularization (Ridge, Lasso, ElasticNet)](#10-regularization-ridge-lasso-elasticnet)
11. [Feature Scaling](#11-feature-scaling)
12. [Complete Python Code](#12-complete-python-code)
13. [Sklearn Implementation](#13-sklearn-implementation)
14. [Common Mistakes & Tips](#14-common-mistakes--tips)
15. [Interview Questions](#15-interview-questions)

---

## 1. What is Linear Regression?

Linear Regression is a **supervised machine learning algorithm** that models the relationship between:
- **Input features** (independent variables) → `X`
- **Output/target** (dependent variable) → `y`

It assumes the relationship is **linear** (a straight line in 2D, a hyperplane in higher dimensions).

### Simple Analogy
> If you study more hours → you score more marks. Linear Regression finds that line!

### When to Use
- Predicting house prices, salaries, stock prices
- When the relationship between X and y is approximately linear
- When you need interpretability (understand *which* features matter)

---

## 2. Types of Linear Regression

| Type | Features | Formula |
|---|---|---|
| **Simple** | 1 input feature | `y = w₀ + w₁x` |
| **Multiple** | 2+ input features | `y = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ` |
| **Polynomial** | Non-linear data, uses powers of x | `y = w₀ + w₁x + w₂x²` |

---

## 3. The Math Behind It

### Simple Linear Regression

```
ŷ = w₀ + w₁ · x
```

Where:
- `ŷ` = predicted value
- `w₀` = **bias / intercept** (value of y when x = 0)
- `w₁` = **weight / slope** (how much y changes per unit of x)
- `x` = input feature

### Multiple Linear Regression

```
ŷ = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ
```

### Matrix / Vector Form (used in code)

```
ŷ = X · W
```

Where:
- `X` = feature matrix of shape `(m, n+1)` — includes a column of 1s for bias
- `W` = weight vector of shape `(n+1, 1)`
- `m` = number of samples
- `n` = number of features

**Example:**

```
X = [[1, x₁₁, x₁₂],     W = [[w₀],
     [1, x₂₁, x₂₂],          [w₁],
     [1, x₃₁, x₃₂]]          [w₂]]

ŷ = X · W
```

---

## 4. Cost Function (Loss)

The cost function measures **how wrong** our predictions are. We want to **minimize** this.

### Mean Squared Error (MSE) — Most Common

```
J(W) = (1/m) · Σ (ŷᵢ - yᵢ)²
```

Or equivalently:

```
J(W) = (1/2m) · Σ (ŷᵢ - yᵢ)²
```

> The `1/2` is added for mathematical convenience (derivative becomes clean).

Where:
- `m` = number of training samples
- `ŷᵢ` = predicted value for sample i
- `yᵢ` = actual value for sample i

### Why MSE?
- Penalizes **larger errors more** (squared)
- Always **positive**
- **Differentiable** — needed for gradient descent
- Has a nice **convex** shape (one global minimum)

### MSE in Matrix Form

```
J(W) = (1/2m) · (XW - y)ᵀ (XW - y)
```

---

## 5. Gradient Descent

Gradient Descent is the **optimization algorithm** used to find the weights `W` that minimize the cost function.

### Core Idea
> "Walk downhill on the cost surface, step by step, until you reach the bottom."

### Update Rule

```
W := W - α · ∂J/∂W
```

Where:
- `α` = **learning rate** (step size)
- `∂J/∂W` = gradient (slope of cost function w.r.t. W)

### Gradient Computation

For weights (slopes):
```
∂J/∂wⱼ = (1/m) · Σ (ŷᵢ - yᵢ) · xᵢⱼ
```

For bias:
```
∂J/∂w₀ = (1/m) · Σ (ŷᵢ - yᵢ)
```

In matrix form:
```
∂J/∂W = (1/m) · Xᵀ · (XW - y)
```

Update step:
```
W := W - α · (1/m) · Xᵀ · (XW - y)
```

### Types of Gradient Descent

| Type | Samples per Update | Speed | Noise |
|---|---|---|---|
| **Batch GD** | All m samples | Slow, stable | Low |
| **Stochastic GD (SGD)** | 1 sample | Fast, noisy | High |
| **Mini-Batch GD** | k samples (e.g., 32, 64) | Balanced | Medium |

---

## 6. Normal Equation (Closed-Form)

Instead of iterating, we can solve for `W` **directly** (no learning rate needed):

```
W = (XᵀX)⁻¹ · Xᵀ · y
```

### When to Use
- Small datasets (< 10,000 samples)
- When you want an **exact** answer without tuning

### Limitations
- Computing `(XᵀX)⁻¹` is **O(n³)** — very slow for large `n`
- Fails if `XᵀX` is **not invertible** (e.g., duplicate features)
- Not scalable for large datasets → use Gradient Descent instead

---

## 7. Assumptions

Linear Regression **requires** these assumptions to work well:

| # | Assumption | What it Means |
|---|---|---|
| 1 | **Linearity** | Relationship between X and y is linear |
| 2 | **Independence** | Each sample is independent of others |
| 3 | **Homoscedasticity** | Variance of errors is constant (not growing) |
| 4 | **Normality of errors** | Residuals follow a normal distribution |
| 5 | **No multicollinearity** | Features are not highly correlated with each other |

### How to Check
```python
import matplotlib.pyplot as plt
import numpy as np

residuals = y_test - y_pred

# 1. Residual plot (check linearity + homoscedasticity)
plt.scatter(y_pred, residuals)
plt.axhline(0, color='red')
plt.xlabel("Predicted"); plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

# 2. QQ plot (check normality)
import scipy.stats as stats
stats.probplot(residuals, dist="norm", plot=plt)
plt.show()

# 3. Correlation matrix (check multicollinearity)
import seaborn as sns
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()
```

---

## 8. Evaluation Metrics

### 1. Mean Absolute Error (MAE)
```
MAE = (1/m) · Σ |ŷᵢ - yᵢ|
```
- Average of absolute errors
- Less sensitive to outliers
- Same unit as y

### 2. Mean Squared Error (MSE)
```
MSE = (1/m) · Σ (ŷᵢ - yᵢ)²
```
- Penalizes large errors more
- Not in same unit as y (squared)

### 3. Root Mean Squared Error (RMSE)
```
RMSE = √MSE
```
- Same unit as y — easy to interpret
- Most commonly used

### 4. R² Score (Coefficient of Determination)
```
R² = 1 - (SS_res / SS_tot)

SS_res = Σ (yᵢ - ŷᵢ)²   ← residual sum of squares
SS_tot = Σ (yᵢ - ȳ)²    ← total sum of squares (ȳ = mean of y)
```

| R² Value | Meaning |
|---|---|
| `1.0` | Perfect fit |
| `0.8 - 1.0` | Good fit |
| `0.5 - 0.8` | Moderate fit |
| `< 0.5` | Poor fit |
| `< 0` | Model worse than mean baseline |

### 5. Adjusted R²
```
Adjusted R² = 1 - [(1 - R²)(m - 1) / (m - n - 1)]
```
- Penalizes adding useless features
- Always use this for **multiple regression**

---

## 9. Hyperparameters

These are the **settings you tune** before training.

### 9.1 Learning Rate (`α`)

The **most important** hyperparameter for gradient descent.

```
W := W - α · gradient
```

| α Value | Effect |
|---|---|
| Too small (e.g., 0.0001) | Converges very slowly |
| Too large (e.g., 1.0) | Overshoots, may diverge |
| **Ideal (0.001 - 0.1)** | Fast and stable convergence |

```python
# Try different learning rates
learning_rates = [0.001, 0.01, 0.1, 0.3, 1.0]
```

### 9.2 Number of Iterations / Epochs

How many times gradient descent updates the weights.

```python
n_iterations = 1000   # typical range: 100 to 10,000
```

- **Too few** → underfitting (model hasn't learned enough)
- **Too many** → wasted computation (cost already converged)

### 9.3 Batch Size (Mini-Batch GD)

Number of samples used per gradient update.

```python
batch_size = 32   # common: 16, 32, 64, 128, 256
```

| Batch Size | Pros | Cons |
|---|---|---|
| 1 (SGD) | Fast, can escape local minima | Very noisy |
| Full (Batch) | Stable gradient | Slow for large data |
| 32–256 (Mini) | Best of both | Need to tune |

### 9.4 Regularization Parameter (`λ` or `alpha`)

Controls **how much** to penalize large weights (prevents overfitting).

```python
alpha = 0.1   # Ridge/Lasso regularization strength
```

- **λ = 0** → No regularization (standard regression)
- **λ small** → Slight regularization
- **λ large** → Strong regularization (may underfit)

### 9.5 Tolerance (`tol`)

Stop training when improvement is below this threshold.

```python
tol = 1e-4   # default in sklearn
```

### 9.6 Fit Intercept (`fit_intercept`)

Whether to compute and include the bias term `w₀`.

```python
fit_intercept = True   # default; set False if data is already centered
```

### 9.7 Solver (Optimization Method)

```python
# For sklearn's Ridge, SGDRegressor etc.
solver = 'saga'   # options: 'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'
```

### Summary Table of Hyperparameters

| Hyperparameter | Common Values | What it Controls |
|---|---|---|
| `learning_rate` α | 0.0001 – 0.3 | Step size in gradient descent |
| `n_iterations` | 100 – 10,000 | Training duration |
| `batch_size` | 16, 32, 64, 128 | Samples per gradient update |
| `lambda` / `alpha` | 0.001 – 10 | Regularization strength |
| `tolerance` | 1e-4 – 1e-8 | Early stopping threshold |
| `fit_intercept` | True / False | Include bias term or not |
| `solver` | 'lbfgs', 'saga' | Optimization algorithm |

---

## 10. Regularization (Ridge, Lasso, ElasticNet)

Used to **prevent overfitting** by penalizing large weights.

### 10.1 Ridge Regression (L2)

```
J(W) = MSE + λ · Σ wⱼ²
```

- Adds **sum of squared weights** to cost
- Shrinks weights toward zero but **never to exactly zero**
- Good when **all features** are somewhat relevant

```python
from sklearn.linear_model import Ridge
model = Ridge(alpha=1.0)   # alpha = λ
model.fit(X_train, y_train)
```

### 10.2 Lasso Regression (L1)

```
J(W) = MSE + λ · Σ |wⱼ|
```

- Adds **sum of absolute weights** to cost
- Can shrink weights to **exactly zero** → automatic feature selection!
- Good when you suspect **many features are irrelevant**

```python
from sklearn.linear_model import Lasso
model = Lasso(alpha=0.1)
model.fit(X_train, y_train)
```

### 10.3 ElasticNet (L1 + L2)

```
J(W) = MSE + λ₁ · Σ |wⱼ| + λ₂ · Σ wⱼ²
```

- Combines Ridge and Lasso
- `l1_ratio` controls the mix (0 = Ridge, 1 = Lasso)

```python
from sklearn.linear_model import ElasticNet
model = ElasticNet(alpha=0.1, l1_ratio=0.5)   # 50% L1 + 50% L2
model.fit(X_train, y_train)
```

### Comparison Table

| Method | Penalty | Feature Selection | Handles Correlated Features |
|---|---|---|---|
| Linear Regression | None | No | Poorly |
| Ridge (L2) | `λΣwⱼ²` | No | Yes |
| Lasso (L1) | `λΣ\|wⱼ\|` | **Yes** | Poorly |
| ElasticNet | L1 + L2 | **Yes** | Yes |

---

## 11. Feature Scaling

Linear Regression (with gradient descent) is **sensitive to feature scale**. Always scale!

### Why?
If `x₁` is in range [0, 1] but `x₂` is in range [0, 1,000,000], gradient descent will have a very elongated cost surface — slow and unstable.

### Methods

#### Standardization (Z-score normalization)
```
x_scaled = (x - μ) / σ
```
- Mean = 0, Std = 1
- Best for most cases

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)   # use fit from train!
```

#### Min-Max Normalization
```
x_scaled = (x - x_min) / (x_max - x_min)
```
- Scales to [0, 1]

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```

> ⚠️ **Always fit scaler on training data only**, then transform both train and test.

---

## 12. Complete Python Code

### From Scratch (NumPy only)

```python
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# 1. Generate Synthetic Data
# ─────────────────────────────────────────────
np.random.seed(42)
m = 100           # number of samples
X = 2 * np.random.rand(m, 1)
y = 4 + 3 * X + np.random.randn(m, 1)   # y = 4 + 3x + noise

# ─────────────────────────────────────────────
# 2. Add Bias Column (column of ones)
# ─────────────────────────────────────────────
X_b = np.c_[np.ones((m, 1)), X]   # shape: (100, 2)

# ─────────────────────────────────────────────
# 3. Hyperparameters
# ─────────────────────────────────────────────
learning_rate = 0.1
n_iterations  = 1000
batch_size    = 32       # for mini-batch GD

# ─────────────────────────────────────────────
# 4. Initialize Weights
# ─────────────────────────────────────────────
W = np.random.randn(2, 1)   # [w0 (bias), w1 (slope)]
cost_history = []

# ─────────────────────────────────────────────
# 5. Gradient Descent
# ─────────────────────────────────────────────
for iteration in range(n_iterations):
    # Predictions
    y_pred = X_b.dot(W)
    
    # Error
    error = y_pred - y
    
    # Cost (MSE)
    cost = (1 / (2 * m)) * np.sum(error ** 2)
    cost_history.append(cost)
    
    # Gradient
    gradient = (1 / m) * X_b.T.dot(error)
    
    # Update weights
    W = W - learning_rate * gradient

print(f"Learned weights: bias={W[0][0]:.4f}, slope={W[1][0]:.4f}")
print(f"True values:     bias=4.0000, slope=3.0000")
print(f"Final cost: {cost_history[-1]:.6f}")

# ─────────────────────────────────────────────
# 6. Predict
# ─────────────────────────────────────────────
def predict(X_new, W):
    X_new_b = np.c_[np.ones((len(X_new), 1)), X_new]
    return X_new_b.dot(W)

X_new = np.array([[0], [2]])
print("\nPredictions:", predict(X_new, W).flatten())

# ─────────────────────────────────────────────
# 7. Plot Cost History
# ─────────────────────────────────────────────
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(cost_history)
plt.title("Cost vs Iterations")
plt.xlabel("Iteration")
plt.ylabel("MSE Cost")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(X, y, alpha=0.5, label="Data")
X_line = np.linspace(0, 2, 100).reshape(-1, 1)
plt.plot(X_line, predict(X_line, W), 'r-', linewidth=2, label="Prediction")
plt.title("Linear Regression Fit")
plt.xlabel("X"); plt.ylabel("y")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

### Normal Equation (Closed-Form)

```python
import numpy as np

# Normal Equation: W = (XᵀX)⁻¹ Xᵀ y
W_normal = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print(f"Normal Eq: bias={W_normal[0][0]:.4f}, slope={W_normal[1][0]:.4f}")
```

### Evaluation Metrics (from scratch)

```python
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

# Test
y_pred_all = X_b.dot(W)
print(f"MAE:  {mean_absolute_error(y, y_pred_all):.4f}")
print(f"MSE:  {mean_squared_error(y, y_pred_all):.4f}")
print(f"RMSE: {rmse(y, y_pred_all):.4f}")
print(f"R²:   {r2_score(y, y_pred_all):.4f}")
```

---

## 13. Sklearn Implementation

### Basic Usage

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────
# Dataset (House Price Example)
# ─────────────────────────────────────────────
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print("Shape:", X.shape)
print("Features:", X.columns.tolist())
print("\nFirst 5 rows:")
print(X.head())

# ─────────────────────────────────────────────
# Train-Test Split
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ─────────────────────────────────────────────
# Feature Scaling
# ─────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# Linear Regression
# ─────────────────────────────────────────────
lr = LinearRegression(fit_intercept=True)
lr.fit(X_train_scaled, y_train)

y_pred = lr.predict(X_test_scaled)

print("\n--- Linear Regression ---")
print(f"Intercept: {lr.intercept_:.4f}")
print(f"Coefficients: {lr.coef_}")
print(f"MAE:  {mean_absolute_error(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"R²:   {r2_score(y_test, y_pred):.4f}")
```

### Ridge, Lasso, ElasticNet

```python
# ─────────────────────────────────────────────
# Ridge Regression
# ─────────────────────────────────────────────
ridge = Ridge(alpha=1.0, fit_intercept=True, solver='auto')
ridge.fit(X_train_scaled, y_train)
y_pred_ridge = ridge.predict(X_test_scaled)
print(f"\nRidge R²: {r2_score(y_test, y_pred_ridge):.4f}")

# ─────────────────────────────────────────────
# Lasso Regression
# ─────────────────────────────────────────────
lasso = Lasso(alpha=0.01, max_iter=10000, tol=1e-4)
lasso.fit(X_train_scaled, y_train)
y_pred_lasso = lasso.predict(X_test_scaled)
print(f"Lasso R²: {r2_score(y_test, y_pred_lasso):.4f}")
print(f"Lasso zero coefficients: {np.sum(lasso.coef_ == 0)}")

# ─────────────────────────────────────────────
# ElasticNet
# ─────────────────────────────────────────────
en = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000)
en.fit(X_train_scaled, y_train)
y_pred_en = en.predict(X_test_scaled)
print(f"ElasticNet R²: {r2_score(y_test, y_pred_en):.4f}")
```

### Hyperparameter Tuning with GridSearchCV

```python
# ─────────────────────────────────────────────
# Grid Search for Ridge
# ─────────────────────────────────────────────
param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
}

ridge_cv = GridSearchCV(
    Ridge(),
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)
ridge_cv.fit(X_train_scaled, y_train)

print(f"\nBest alpha: {ridge_cv.best_params_}")
print(f"Best CV R²: {ridge_cv.best_score_:.4f}")
best_ridge = ridge_cv.best_estimator_
y_pred_best = best_ridge.predict(X_test_scaled)
print(f"Test R²:    {r2_score(y_test, y_pred_best):.4f}")
```

### Pipeline (Best Practice)

```python
# ─────────────────────────────────────────────
# Full Pipeline: Scale → Model
# ─────────────────────────────────────────────
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Ridge(alpha=1.0))
])

pipeline.fit(X_train, y_train)
y_pred_pipe = pipeline.predict(X_test)
print(f"\nPipeline R²: {r2_score(y_test, y_pred_pipe):.4f}")

# Cross Validation
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
print(f"CV R² scores: {cv_scores}")
print(f"Mean CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
```

### Polynomial Regression

```python
# ─────────────────────────────────────────────
# Polynomial Regression (degree=2)
# ─────────────────────────────────────────────
poly_pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('model', Ridge(alpha=1.0))
])

poly_pipeline.fit(X_train, y_train)
y_pred_poly = poly_pipeline.predict(X_test)
print(f"\nPolynomial (degree=2) R²: {r2_score(y_test, y_pred_poly):.4f}")
```

### Visualization

```python
# ─────────────────────────────────────────────
# Visualization Dashboard
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Actual vs Predicted
axes[0, 0].scatter(y_test, y_pred, alpha=0.3, color='steelblue')
axes[0, 0].plot([y_test.min(), y_test.max()],
                [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 0].set_title("Actual vs Predicted")
axes[0, 0].set_xlabel("Actual"); axes[0, 0].set_ylabel("Predicted")

# 2. Residual Plot
residuals = y_test - y_pred
axes[0, 1].scatter(y_pred, residuals, alpha=0.3, color='orange')
axes[0, 1].axhline(0, color='red', linestyle='--')
axes[0, 1].set_title("Residuals vs Predicted")
axes[0, 1].set_xlabel("Predicted"); axes[0, 1].set_ylabel("Residuals")

# 3. Residual Histogram
axes[1, 0].hist(residuals, bins=50, edgecolor='k', color='green', alpha=0.7)
axes[1, 0].set_title("Residuals Distribution")
axes[1, 0].set_xlabel("Residual"); axes[1, 0].set_ylabel("Count")

# 4. Feature Coefficients
coef_df = pd.DataFrame({
    'Feature': data.feature_names,
    'Coefficient': lr.coef_
}).sort_values('Coefficient')
axes[1, 1].barh(coef_df['Feature'], coef_df['Coefficient'], color='purple', alpha=0.7)
axes[1, 1].axvline(0, color='red', linestyle='--')
axes[1, 1].set_title("Feature Coefficients")

plt.tight_layout()
plt.savefig("linear_regression_analysis.png", dpi=150)
plt.show()
```

### Alpha vs R² Plot (Regularization Tuning)

```python
# ─────────────────────────────────────────────
# Regularization Path
# ─────────────────────────────────────────────
alphas = np.logspace(-4, 4, 100)
ridge_scores = []
lasso_scores = []

for alpha in alphas:
    r = Ridge(alpha=alpha)
    r.fit(X_train_scaled, y_train)
    ridge_scores.append(r2_score(y_test, r.predict(X_test_scaled)))
    
    try:
        l = Lasso(alpha=alpha, max_iter=10000)
        l.fit(X_train_scaled, y_train)
        lasso_scores.append(r2_score(y_test, l.predict(X_test_scaled)))
    except:
        lasso_scores.append(np.nan)

plt.figure(figsize=(10, 5))
plt.semilogx(alphas, ridge_scores, label='Ridge', color='blue')
plt.semilogx(alphas, lasso_scores, label='Lasso', color='red')
plt.xlabel("Alpha (Regularization Strength)")
plt.ylabel("R² Score")
plt.title("Regularization Path: R² vs Alpha")
plt.legend(); plt.grid(True)
plt.show()
```

---

## 14. Common Mistakes & Tips

### ❌ Mistake 1: Not Scaling Features
```python
# Wrong
model.fit(X_train, y_train)   # features in different scales

# Correct
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
model.fit(X_scaled, y_train)
```

### ❌ Mistake 2: Fitting Scaler on Test Data
```python
# Wrong — data leakage!
X_test_scaled = scaler.fit_transform(X_test)

# Correct — only transform test
X_test_scaled = scaler.transform(X_test)
```

### ❌ Mistake 3: Using Linear Regression for Non-Linear Data
```python
# Check relationship first
plt.scatter(X, y)   # if it looks curved → use PolynomialFeatures or tree-based models
```

### ❌ Mistake 4: Ignoring Multicollinearity
```python
# Check VIF (Variance Inflation Factor)
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
print(vif_data)
# VIF > 10 → multicollinearity problem → drop or combine features
```

### ❌ Mistake 5: Not Checking Outliers
```python
# Outliers heavily affect linear regression
import scipy.stats as stats
z_scores = np.abs(stats.zscore(X))
X_clean = X[(z_scores < 3).all(axis=1)]   # remove outliers
```

### ✅ Tips

- Always do **EDA** (Exploratory Data Analysis) before modeling
- Use **cross-validation** instead of a single train/test split
- Try **Ridge first**, then Lasso if feature selection is needed
- For large datasets, use **SGDRegressor** (much faster)
- Use **Pipeline** to avoid data leakage and simplify code
- Plot **learning curves** to diagnose underfitting/overfitting

---

## 15. Interview Questions

**Q1: What is the difference between MSE and MAE?**
> MSE penalizes large errors more (squared), making it sensitive to outliers. MAE treats all errors equally and is more robust.

**Q2: When would you use Lasso over Ridge?**
> Lasso when you want automatic feature selection (it can set weights to zero). Ridge when all features are useful but need shrinkage.

**Q3: What happens if learning rate is too high?**
> The cost function diverges (increases instead of decreasing). The weights overshoot the minimum.

**Q4: Why add a column of 1s to X?**
> To absorb the bias term `w₀` into the matrix multiplication `XW`, simplifying the math.

**Q5: Is Linear Regression affected by outliers?**
> Yes, heavily. MSE squares the errors so outliers dominate. Use MAE or Huber loss for robustness.

**Q6: What is the Normal Equation and when NOT to use it?**
> It computes exact weights in one step: `W = (XᵀX)⁻¹ Xᵀy`. Avoid it for large `n` (features) because matrix inversion is O(n³) — too slow.

**Q7: What is multicollinearity?**
> When two or more features are highly correlated. It makes weights unstable and hard to interpret. Detected via VIF > 10 or correlation matrix.

**Q8: Difference between parameters and hyperparameters?**
> **Parameters** (weights `W`) are learned during training. **Hyperparameters** (learning rate, alpha, batch size) are set before training.

---

## 📎 Quick Reference Cheat Sheet

```
Model:        ŷ = Xw + b
Loss:         J = (1/2m) Σ(ŷ - y)²
Update:       W := W - α · (1/m) · Xᵀ(Xw - y)
Normal Eq:    W = (XᵀX)⁻¹Xᵀy
Ridge:        J + λΣwⱼ²
Lasso:        J + λΣ|wⱼ|
ElasticNet:   J + λ₁Σ|wⱼ| + λ₂Σwⱼ²
R²:           1 - SS_res/SS_tot
```

---

## 🔗 Resources

- [Scikit-learn Docs — LinearRegression](https://scikit-learn.org/stable/modules/linear_model.html)
- [StatQuest — Linear Regression (YouTube)](https://www.youtube.com/watch?v=nk2CQITm_eo)
- [Andrew Ng — ML Course Notes](https://cs229.stanford.edu/notes2022fall/main_notes.pdf)
- [Elements of Statistical Learning (ESL) — Free PDF](https://hastie.su.domains/ElemStatLearn/)

---

*Made with 💙 for ML/DL learners — covers theory, math, code, and hyperparameters completely.*
