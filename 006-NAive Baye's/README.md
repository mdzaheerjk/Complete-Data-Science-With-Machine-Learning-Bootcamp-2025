# 🟣 Naive Bayes — Complete ML/DL Job-Ready Notes

> **"Given what I know about this class, how probable is this new data point?"**
> Naive Bayes is a **probabilistic, generative, eager learning** classifier based on Bayes' Theorem with a strong (naive) assumption of **conditional independence** between features.

---

## 📌 Table of Contents

1. [Intuition](#1-intuition)
2. [Bayes' Theorem — The Foundation](#2-bayes-theorem--the-foundation)
3. [The Naive Assumption](#3-the-naive-assumption)
4. [Full Mathematical Derivation](#4-full-mathematical-derivation)
5. [Types of Naive Bayes](#5-types-of-naive-bayes)
6. [Log-Probability Trick (Numerical Stability)](#6-log-probability-trick-numerical-stability)
7. [Laplace Smoothing](#7-laplace-smoothing)
8. [Bias-Variance in Naive Bayes](#8-bias-variance-in-naive-bayes)
9. [Hyperparameters](#9-hyperparameters)
10. [Python from Scratch](#10-python-from-scratch)
11. [Scikit-learn Implementation](#11-scikit-learn-implementation)
12. [Full Pipeline with Best Practices](#12-full-pipeline-with-best-practices)
13. [Text Classification (NLP Use Case)](#13-text-classification-nlp-use-case)
14. [Evaluation Metrics](#14-evaluation-metrics)
15. [Pros and Cons](#15-pros-and-cons)
16. [Interview Questions](#16-interview-questions)
17. [Resources](#17-resources)

---

## 1. Intuition

**Simple statement:** Naive Bayes asks — "Given a new data point, what is the probability it belongs to each class? Pick the class with the highest probability."

**Real-world analogy:**
- **Spam filter:** Given the words in an email ("FREE", "WINNER", "CLAIM"), what's the probability it's spam vs not spam?
- **Medical diagnosis:** Given symptoms (fever=yes, cough=yes), what disease is most probable?
- **Sentiment analysis:** Given words in a review, is it positive or negative?

**Key properties:**
- **Probabilistic** → Outputs class probabilities, not just labels
- **Generative** → Models $P(X|Y)$ and $P(Y)$, not directly $P(Y|X)$
- **Eager learner** → Builds a model during training (unlike KNN)
- **Naive** → Assumes ALL features are conditionally independent given the class
- **Fast** → Training is O(N·d), Prediction is O(d·C) — extremely scalable

---

## 2. Bayes' Theorem — The Foundation

### The Formula

$$P(Y | X) = \frac{P(X | Y) \cdot P(Y)}{P(X)}$$

| Term | Name | Meaning |
|------|------|---------|
| $P(Y \mid X)$ | **Posterior** | Probability of class Y given features X (what we want) |
| $P(X \mid Y)$ | **Likelihood** | Probability of seeing features X if class is Y |
| $P(Y)$ | **Prior** | Probability of class Y before seeing any data |
| $P(X)$ | **Evidence** | Probability of features X (normalizing constant) |

### Simple Example

```
Problem: Is an email spam?

Prior:     P(spam) = 0.3,  P(not spam) = 0.7
Likelihood:
  P("FREE" | spam)     = 0.8
  P("FREE" | not spam) = 0.1

P(X) = P("FREE") = P("FREE"|spam)·P(spam) + P("FREE"|not spam)·P(not spam)
     = 0.8×0.3 + 0.1×0.7 = 0.24 + 0.07 = 0.31

Posterior:
  P(spam | "FREE")     = (0.8 × 0.3) / 0.31 = 0.774
  P(not spam | "FREE") = (0.1 × 0.7) / 0.31 = 0.226

→ Prediction: SPAM  ✅
```

---

## 3. The Naive Assumption

### What is it?

**Naive Bayes assumes all features are conditionally independent given the class:**

$$P(X_1, X_2, ..., X_d \mid Y) = \prod_{j=1}^{d} P(X_j \mid Y)$$

### Why "Naive"?

In reality, features are often correlated:
- "FREE" and "WIN" often appear together in spam
- Height and weight are correlated

But despite this wrong assumption, Naive Bayes works surprisingly well in practice — especially for text classification.

### Why does it still work?

We don't need the probabilities to be perfectly calibrated — we just need the **ranking** to be correct:

$$\hat{y} = \arg\max_c P(Y=c) \prod_{j=1}^{d} P(X_j | Y=c)$$

As long as the correct class gets the highest score, the prediction is correct.

---

## 4. Full Mathematical Derivation

### Classification Rule

We want to find the class $c$ that maximizes the posterior:

$$\hat{y} = \arg\max_{c \in C} P(Y=c \mid X_1, X_2, ..., X_d)$$

**Step 1:** Apply Bayes' theorem:

$$= \arg\max_{c} \frac{P(X_1, X_2, ..., X_d \mid Y=c) \cdot P(Y=c)}{P(X_1, X_2, ..., X_d)}$$

**Step 2:** Drop $P(X)$ — it's the same for all classes (doesn't affect argmax):

$$= \arg\max_{c} \ P(X_1, X_2, ..., X_d \mid Y=c) \cdot P(Y=c)$$

**Step 3:** Apply the Naive (independence) assumption:

$$= \arg\max_{c} \ P(Y=c) \cdot \prod_{j=1}^{d} P(X_j \mid Y=c)$$

**Step 4:** Take log to avoid numerical underflow (products → sums):

$$= \arg\max_{c} \left[ \log P(Y=c) + \sum_{j=1}^{d} \log P(X_j \mid Y=c) \right]$$

### Parameter Estimation from Training Data

**Prior:**
$$P(Y=c) = \frac{\text{count of samples with class } c}{N}$$

**Likelihood (Categorical/Bernoulli):**
$$P(X_j = v \mid Y=c) = \frac{\text{count}(X_j = v \text{ AND } Y=c)}{\text{count}(Y=c)}$$

**Likelihood (Gaussian):**
$$P(X_j \mid Y=c) = \frac{1}{\sqrt{2\pi\sigma_{jc}^2}} \exp\left(-\frac{(X_j - \mu_{jc})^2}{2\sigma_{jc}^2}\right)$$

where $\mu_{jc}$ and $\sigma_{jc}^2$ are the mean and variance of feature $j$ in class $c$.

---

## 5. Types of Naive Bayes

### Overview Table

| Type | Likelihood Model | Feature Type | Best For |
|------|-----------------|--------------|----------|
| **Gaussian NB** | Normal distribution | Continuous | Iris, numeric features |
| **Multinomial NB** | Multinomial distribution | Count data | TF-IDF, word counts |
| **Bernoulli NB** | Bernoulli distribution | Binary (0/1) | Word presence/absence |
| **Complement NB** | Complement of Multinomial | Count data | Imbalanced text |
| **Categorical NB** | Categorical distribution | Categorical | Nominal features |

---

### 5.1 Gaussian Naive Bayes

**When:** Continuous features (assume Gaussian distribution per class)

**Likelihood:**
$$P(X_j \mid Y=c) = \frac{1}{\sqrt{2\pi\sigma_{jc}^2}} \exp\left(-\frac{(X_j - \mu_{jc})^2}{2\sigma_{jc}^2}\right)$$

**Training:** Compute $\mu_{jc}$ and $\sigma_{jc}^2$ for each feature $j$ and class $c$

```python
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB(var_smoothing=1e-9)
gnb.fit(X_train, y_train)

# Learned parameters
print(gnb.theta_)        # means:    shape (n_classes, n_features)
print(gnb.var_)          # variances: shape (n_classes, n_features)
print(gnb.class_prior_)  # P(Y=c) for each class
```

---

### 5.2 Multinomial Naive Bayes

**When:** Count/frequency features (word counts, TF-IDF)

**Likelihood:**
$$P(X \mid Y=c) = \frac{(\sum_j X_j)!}{\prod_j X_j!} \prod_{j=1}^{d} \hat{\theta}_{jc}^{X_j}$$

where $\hat{\theta}_{jc} = \frac{N_{jc} + \alpha}{N_c + \alpha \cdot d}$ (with Laplace smoothing $\alpha$)

```python
from sklearn.naive_bayes import MultinomialNB

mnb = MultinomialNB(alpha=1.0)  # alpha = Laplace smoothing
mnb.fit(X_train_counts, y_train)

# Learned parameters
print(mnb.class_log_prior_)        # log P(Y=c)
print(mnb.feature_log_prob_)       # log P(X_j | Y=c)
```

---

### 5.3 Bernoulli Naive Bayes

**When:** Binary features (word present=1, absent=0)

**Likelihood:**
$$P(X \mid Y=c) = \prod_{j=1}^{d} \hat{\theta}_{jc}^{X_j} (1 - \hat{\theta}_{jc})^{(1-X_j)}$$

Key difference from Multinomial: **explicitly penalizes absent features**

```python
from sklearn.naive_bayes import BernoulliNB

bnb = BernoulliNB(alpha=1.0, binarize=0.0)
# binarize: threshold to convert continuous to binary (None if already binary)
bnb.fit(X_train_binary, y_train)
```

---

### 5.4 Complement Naive Bayes

**When:** Imbalanced text datasets (outperforms Multinomial NB)

**Idea:** Instead of estimating $P(X \mid Y=c)$, estimate $P(X \mid Y \neq c)$ — use complement classes.

$$\hat{\theta}_{jc} = \frac{\alpha + \sum_{i: y_i \neq c} X_{ij}}{\alpha \cdot d + \sum_{i: y_i \neq c} \sum_k X_{ik}}$$

```python
from sklearn.naive_bayes import ComplementNB

cnb = ComplementNB(alpha=1.0, norm=False)
cnb.fit(X_train_counts, y_train)
```

---

### 5.5 Categorical Naive Bayes

**When:** Categorical (nominal) features

```python
from sklearn.naive_bayes import CategoricalNB

catnb = CategoricalNB(alpha=1.0)
catnb.fit(X_train_categorical, y_train)
```

---

## 6. Log-Probability Trick (Numerical Stability)

**Problem:** Multiplying many small probabilities → numerical underflow

```python
# BAD: Numerical underflow
P = P(Y=c) * P(x1|c) * P(x2|c) * ... * P(x1000|c)
# = 0.3 * 0.001 * 0.002 * ... → reaches 0.0 (float underflow)

# GOOD: Use log-sum (converts products to sums)
log_P = log(P(Y=c)) + log(P(x1|c)) + log(P(x2|c)) + ... + log(P(x1000|c))
# Stays in numerical range
```

**In code:**

```python
import numpy as np

def log_posterior(X, prior, likelihood_params):
    log_prior = np.log(prior)
    log_likelihood = np.sum(np.log(likelihood_params))  # sum, not product!
    return log_prior + log_likelihood
```

Scikit-learn handles this automatically — all internal computations use log-probabilities.

---

## 7. Laplace Smoothing

### The Zero-Frequency Problem

**Problem:** If a word never appeared with a class in training, its probability = 0, and the entire product = 0.

```
Training:  P("unicorn" | spam) = 0/100 = 0.0
Test:      "WIN a unicorn NOW"
Product:   P(spam) × P("WIN"|spam) × P("unicorn"|spam) × P("NOW"|spam)
         = 0.3 × 0.8 × 0.0 × 0.7 = 0.0  ← Kills everything!
```

### Laplace Smoothing Fix

Add pseudo-count $\alpha$ to every feature count:

$$P(X_j = v \mid Y=c) = \frac{\text{count}(X_j=v, Y=c) + \alpha}{\text{count}(Y=c) + \alpha \cdot |V|}$$

where $|V|$ = vocabulary size (number of unique values)

```
Without smoothing (α=0): P("unicorn"|spam) = 0/100 = 0.0
With smoothing    (α=1): P("unicorn"|spam) = (0+1)/(100 + 10000) = 0.0001
                         → Not zero! Doesn't kill the product.
```

```python
# In sklearn, alpha controls Laplace smoothing
from sklearn.naive_bayes import MultinomialNB

mnb = MultinomialNB(alpha=1.0)   # alpha=1.0 → standard Laplace smoothing
mnb = MultinomialNB(alpha=0.1)   # smaller alpha → less smoothing
mnb = MultinomialNB(alpha=0.0)   # no smoothing → risk of zero probabilities
```

**Additive smoothing general form:**
$$\hat{\theta}_{jc} = \frac{N_{jc} + \alpha}{N_c + \alpha \cdot d}$$

---

## 8. Bias-Variance in Naive Bayes

| Property | Value | Explanation |
|----------|-------|-------------|
| **Bias** | High | Strong independence assumption is almost always wrong |
| **Variance** | Low | Simple model, few parameters, generalizes well |
| **Overall** | High bias, low variance | Works best when assumption approximately holds |

```
High Bias  → Underfitting possible if features are highly correlated
Low Variance → Doesn't overfit, works well with small training data

Unlike KNN or Decision Trees, Naive Bayes has no "complexity" knob —
its bias is structural (baked into the independence assumption).
```

**Note:** Naive Bayes has no gradient descent. Parameters are computed analytically (closed-form) from counting + Gaussian statistics.

**Why no gradient descent?**
- Likelihood has a closed-form solution
- $\mu_{jc}$ = sample mean, $\sigma_{jc}^2$ = sample variance → just compute directly
- Priors = class frequencies → just count

---

## 9. Hyperparameters

### Gaussian NB

```python
GaussianNB(
    priors=None,          # Prior probabilities P(Y=c). None = estimated from data
    var_smoothing=1e-9    # Portion of largest variance added to variances
                          # Prevents division by zero in likelihood computation
)
```

| Param | Default | Effect |
|-------|---------|--------|
| `priors` | `None` (from data) | Set manually for class imbalance handling |
| `var_smoothing` | `1e-9` | Increase if features have very small variance |

### Multinomial NB

```python
MultinomialNB(
    alpha=1.0,            # Laplace/Lidstone smoothing (0=none, 1=Laplace)
    force_alpha=True,     # Force alpha even if alpha=0 causes issues
    fit_prior=True,       # Whether to learn class prior probabilities
    class_prior=None      # Manual class priors (if fit_prior=False)
)
```

| Param | Default | Effect |
|-------|---------|--------|
| `alpha` | `1.0` | **Most important.** 0→no smoothing, 1→Laplace, 0.5→Lidstone |
| `fit_prior` | `True` | False → uniform prior (1/K for each class) |
| `class_prior` | `None` | Set manually to handle class imbalance |

### Bernoulli NB

```python
BernoulliNB(
    alpha=1.0,            # Smoothing parameter
    force_alpha=True,
    binarize=0.0,         # Threshold to binarize continuous features
                          # None → assume already binary
    fit_prior=True,
    class_prior=None
)
```

| Param | Default | Effect |
|-------|---------|--------|
| `binarize` | `0.0` | Values > threshold → 1, else → 0. None if already binary |

### Complement NB

```python
ComplementNB(
    alpha=1.0,            # Smoothing
    force_alpha=True,
    fit_prior=True,
    class_prior=None,
    norm=False            # Normalize second weights (True recommended for long docs)
)
```

### Categorical NB

```python
CategoricalNB(
    alpha=1.0,            # Smoothing
    force_alpha=True,
    fit_prior=True,
    class_prior=None,
    min_categories=None   # Minimum number of categories per feature
)
```

---

## 10. Python from Scratch

### Gaussian Naive Bayes from Scratch

```python
import numpy as np
from collections import defaultdict

class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes from scratch.
    
    Training:  O(N * d) — compute mean/variance per class
    Prediction: O(d * C) — compute likelihood for each class
    """
    
    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        
        # Store parameters per class
        self.priors_ = {}
        self.means_ = {}
        self.vars_ = {}
        
        for c in self.classes_:
            X_c = X[y == c]
            
            # Prior: P(Y=c) = count(Y=c) / N
            self.priors_[c] = len(X_c) / n_samples
            
            # Mean and variance for each feature per class
            self.means_[c] = X_c.mean(axis=0)
            self.vars_[c] = X_c.var(axis=0) + 1e-9  # var_smoothing
        
        return self
    
    def _gaussian_log_likelihood(self, X, mean, var):
        """
        Log of Gaussian PDF:
        log P(x|μ,σ²) = -0.5*log(2πσ²) - (x-μ)²/(2σ²)
        """
        log_coeff = -0.5 * np.log(2 * np.pi * var)
        log_exp = -0.5 * ((X - mean) ** 2) / var
        return log_coeff + log_exp  # shape: (n_features,)
    
    def _log_posterior(self, x):
        """Compute log posterior for each class for single sample x."""
        log_posteriors = {}
        for c in self.classes_:
            # log P(Y=c)
            log_prior = np.log(self.priors_[c])
            # Σ log P(x_j | Y=c) — independence assumption: sum log-likelihoods
            log_likelihood = np.sum(
                self._gaussian_log_likelihood(x, self.means_[c], self.vars_[c])
            )
            log_posteriors[c] = log_prior + log_likelihood
        return log_posteriors
    
    def predict(self, X):
        X = np.array(X)
        predictions = []
        for x in X:
            log_posts = self._log_posterior(x)
            predictions.append(max(log_posts, key=log_posts.get))
        return np.array(predictions)
    
    def predict_proba(self, X):
        X = np.array(X)
        probas = []
        for x in X:
            log_posts = self._log_posterior(x)
            # Convert log-posteriors to probabilities via softmax
            log_vals = np.array([log_posts[c] for c in self.classes_])
            log_vals -= log_vals.max()  # numerical stability
            probs = np.exp(log_vals)
            probs /= probs.sum()
            probas.append(probs)
        return np.array(probas)
    
    def score(self, X, y):
        return np.mean(self.predict(X) == np.array(y))


# ---- Multinomial Naive Bayes from Scratch ----

class MultinomialNaiveBayes:
    """
    Multinomial Naive Bayes with Laplace smoothing.
    Best for: word count / TF-IDF features.
    """
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Laplace smoothing
    
    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        
        self.log_priors_ = {}
        self.log_likelihoods_ = {}
        
        for c in self.classes_:
            X_c = X[y == c]
            
            # log P(Y=c) = log(count(c) / N)
            self.log_priors_[c] = np.log(len(X_c) / n_samples)
            
            # Laplace-smoothed log likelihood:
            # log P(X_j | Y=c) = log((count(X_j, c) + α) / (count(c) + α*d))
            feature_counts = X_c.sum(axis=0) + self.alpha          # N_jc + α
            total_count = feature_counts.sum()                       # N_c + α*d
            self.log_likelihoods_[c] = np.log(feature_counts / total_count)
        
        return self
    
    def predict(self, X):
        X = np.array(X)
        predictions = []
        for x in X:
            scores = {}
            for c in self.classes_:
                # log P(Y=c) + Σ x_j * log P(X_j | Y=c)
                scores[c] = self.log_priors_[c] + np.dot(x, self.log_likelihoods_[c])
            predictions.append(max(scores, key=scores.get))
        return np.array(predictions)
    
    def score(self, X, y):
        return np.mean(self.predict(X) == np.array(y))


# ---- Test Both ----
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Gaussian NB — continuous features
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

gnb_scratch = GaussianNaiveBayes()
gnb_scratch.fit(X_train, y_train)
print(f"Gaussian NB (scratch) Accuracy: {gnb_scratch.score(X_test, y_test):.4f}")
```

---

## 11. Scikit-learn Implementation

### Gaussian NB — Continuous Features

```python
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ── Data ──────────────────────────────────────────────────────
iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── NOTE: Gaussian NB does NOT require feature scaling
# (it models each feature's distribution independently)
# But scaling can still help in practice for numerical stability

# ── Model ──────────────────────────────────────────────────────
gnb = GaussianNB(var_smoothing=1e-9)
gnb.fit(X_train, y_train)

# ── Predict ────────────────────────────────────────────────────
y_pred = gnb.predict(X_test)
y_proba = gnb.predict_proba(X_test)

print(f"Test Accuracy: {gnb.score(X_test, y_test):.4f}")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# ── Cross Validation ───────────────────────────────────────────
cv_scores = cross_val_score(gnb, X, y, cv=5, scoring='accuracy')
print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── Inspect Learned Parameters ─────────────────────────────────
print("\nClass Priors:", gnb.class_prior_)
print("Feature Means per Class:\n", gnb.theta_)
print("Feature Variances per Class:\n", gnb.var_)

# ── Confusion Matrix ───────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title("Confusion Matrix — Gaussian NB")
plt.tight_layout()
plt.show()
```

### Multinomial NB — Text/Count Features

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# ── Fetch 4 categories from 20 Newsgroups ─────────────────────
categories = ['alt.atheism', 'soc.religion.christian',
              'comp.graphics', 'sci.med']

train = fetch_20newsgroups(subset='train', categories=categories)
test  = fetch_20newsgroups(subset='test',  categories=categories)

# ── Pipeline: Text → Counts → TF-IDF → Multinomial NB ─────────
text_pipeline = Pipeline([
    ('vect', CountVectorizer()),         # word → count matrix
    ('tfidf', TfidfTransformer()),       # counts → TF-IDF
    ('clf', MultinomialNB(alpha=0.1)),   # classify
])

text_pipeline.fit(train.data, train.target)
y_pred = text_pipeline.predict(test.data)

print(f"Accuracy: {text_pipeline.score(test.data, test.target):.4f}")
print(classification_report(test.target, y_pred,
                             target_names=train.target_names))
```

### Bernoulli NB — Binary Features

```python
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer

# ── Binary vectorizer (presence/absence, not counts) ──────────
bvect = CountVectorizer(binary=True)
X_train_bin = bvect.fit_transform(train.data)
X_test_bin = bvect.transform(test.data)

bnb = BernoulliNB(alpha=1.0, binarize=None)  # already binary
bnb.fit(X_train_bin, train.target)
print(f"Bernoulli NB Accuracy: {bnb.score(X_test_bin, test.target):.4f}")
```

### Complement NB — Imbalanced Text

```python
from sklearn.naive_bayes import ComplementNB

cnb = ComplementNB(alpha=0.5, norm=True)
cnb.fit(X_train_bin, train.target)
print(f"Complement NB Accuracy: {cnb.score(X_test_bin, test.target):.4f}")
```

---

## 12. Full Pipeline with Best Practices

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.decomposition import PCA
import numpy as np

# ── Pipeline for Gaussian NB (continuous features) ───────────
gaussian_pipe = Pipeline([
    ('scaler', StandardScaler()),          # optional but helps stability
    ('gnb', GaussianNB())
])

param_grid_gnb = {
    'gnb__var_smoothing': np.logspace(-12, -6, 7)  # [1e-12, 1e-11, ... 1e-6]
}

grid_gnb = GridSearchCV(
    gaussian_pipe, param_grid_gnb,
    cv=5, scoring='accuracy', n_jobs=-1, verbose=1
)
grid_gnb.fit(X_train, y_train)
print(f"Best var_smoothing: {grid_gnb.best_params_}")
print(f"Best CV Score:      {grid_gnb.best_score_:.4f}")

# ── Pipeline for Multinomial NB (text features) ───────────────
from sklearn.feature_extraction.text import TfidfVectorizer

text_pipe = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),           # unigrams and bigrams
        min_df=2,                      # ignore rare words
        sublinear_tf=True,             # log(tf)
    )),
    ('mnb', MultinomialNB())
])

param_grid_mnb = {
    'tfidf__ngram_range': [(1,1), (1,2)],
    'tfidf__max_features': [5000, 10000, 50000],
    'mnb__alpha': [0.01, 0.1, 0.5, 1.0, 2.0],
    'mnb__fit_prior': [True, False],
}

grid_mnb = GridSearchCV(
    text_pipe, param_grid_mnb,
    cv=5, scoring='f1_macro', n_jobs=-1
)
grid_mnb.fit(train.data, train.target)
print(f"Best params: {grid_mnb.best_params_}")
print(f"Best F1:     {grid_mnb.best_score_:.4f}")
```

---

## 13. Text Classification (NLP Use Case)

**Naive Bayes is the go-to baseline for text classification.**

### Complete Email Spam Classifier

```python
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline

# Sample data (replace with real spam dataset)
emails = [
    "FREE WINNER CLAIM YOUR PRIZE NOW",
    "Win a million dollars click here free",
    "Meeting at 3pm please confirm attendance",
    "Your invoice is attached please review",
    "URGENT: You have won a lottery prize",
    "Hi, can we schedule a call this week?",
    "Buy cheap drugs online no prescription",
    "Quarterly report attached for your review",
]
labels = [1, 1, 0, 0, 1, 0, 1, 0]  # 1=spam, 0=ham

X_train, X_test, y_train, y_test = train_test_split(
    emails, labels, test_size=0.25, random_state=42
)

# Build pipeline
spam_clf = Pipeline([
    ('tfidf', TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        ngram_range=(1, 2),
        max_features=50000
    )),
    ('clf', MultinomialNB(alpha=0.1))
])

spam_clf.fit(X_train, y_train)

# Predict
test_emails = ["FREE MONEY WINNER CLAIM", "Please see attached report"]
predictions = spam_clf.predict(test_emails)
probabilities = spam_clf.predict_proba(test_emails)

for email, pred, prob in zip(test_emails, predictions, probabilities):
    label = "SPAM" if pred == 1 else "HAM"
    print(f"'{email[:30]}...' → {label} (confidence: {max(prob):.2%})")

# Feature importance — most spam-indicative words
vectorizer = spam_clf.named_steps['tfidf']
classifier = spam_clf.named_steps['clf']

feature_names = vectorizer.get_feature_names_out()
# Difference in log-likelihoods between spam and ham
log_prob_diff = classifier.feature_log_prob_[1] - classifier.feature_log_prob_[0]
top_spam_words = feature_names[np.argsort(log_prob_diff)[-10:]]
top_ham_words  = feature_names[np.argsort(log_prob_diff)[:10]]

print(f"\nTop spam words: {top_spam_words}")
print(f"Top ham words:  {top_ham_words}")
```

### Updating the Model with New Data (Partial Fit)

```python
# Naive Bayes supports online/incremental learning!
from sklearn.naive_bayes import MultinomialNB

mnb = MultinomialNB(alpha=1.0)

# Train in batches (e.g., streaming data)
for i in range(0, len(X_train_counts), batch_size):
    X_batch = X_train_counts[i:i+batch_size]
    y_batch = y_train[i:i+batch_size]
    mnb.partial_fit(X_batch, y_batch, classes=np.unique(y_train))
```

---

## 14. Evaluation Metrics

### Classification Metrics

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, classification_report,
    ConfusionMatrixDisplay, RocCurveDisplay
)

y_pred  = gnb.predict(X_test)
y_proba = gnb.predict_proba(X_test)

# Binary case
print(f"Accuracy:   {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision:  {precision_score(y_test, y_pred):.4f}")
print(f"Recall:     {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:   {f1_score(y_test, y_pred):.4f}")
print(f"ROC AUC:    {roc_auc_score(y_test, y_proba[:,1]):.4f}")
print(f"Log Loss:   {log_loss(y_test, y_proba):.4f}")

# Multi-class
print(f"ROC AUC (OvR, macro): {roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro'):.4f}")
print(classification_report(y_test, y_pred))

# Probability Calibration Check
from sklearn.calibration import CalibrationDisplay
CalibrationDisplay.from_estimator(gnb, X_test, y_test, n_bins=10)
plt.title("Probability Calibration — Gaussian NB")
plt.show()
```

**Note:** Naive Bayes probabilities are often **poorly calibrated** (too extreme), even when classification accuracy is good. Use `CalibratedClassifierCV` if you need well-calibrated probabilities.

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_gnb = CalibratedClassifierCV(GaussianNB(), cv=5, method='isotonic')
calibrated_gnb.fit(X_train, y_train)
```

---

## 15. Pros and Cons

### ✅ Advantages

| Advantage | Explanation |
|-----------|-------------|
| **Extremely Fast** | O(N·d) training, O(d·C) prediction |
| **Works with Small Data** | Just needs counts — doesn't need thousands of samples |
| **Handles High Dimensions** | Scales to millions of features (text!) |
| **Multi-class Natural** | Works for any number of classes directly |
| **Incremental Learning** | `partial_fit()` for streaming/online learning |
| **Interpretable** | Can inspect P(X|Y) and P(Y) directly |
| **Robust to Irrelevant Features** | They just contribute near-equal probabilities |
| **No Feature Scaling Needed** | (for Multinomial/Bernoulli variants) |

### ❌ Disadvantages

| Disadvantage | Explanation |
|--------------|-------------|
| **Independence Assumption** | Almost never holds in reality |
| **Zero Frequency Problem** | Needs Laplace smoothing |
| **Poor Probability Calibration** | Probabilities too extreme (near 0 or 1) |
| **Can't Model Feature Interactions** | "spam" AND "urgent" together ≠ "spam" alone |
| **Gaussian Assumption** | GaussianNB fails on skewed/multimodal distributions |
| **Not Good for Regression** | Classification only (unless modified) |

### When to Use Naive Bayes

✅ **Use when:**
- Text classification (spam, sentiment, news categories)
- Baseline model / quick prototype
- Very high-dimensional data (10k+ features)
- Small training data
- Real-time streaming / online learning
- Multi-class problems

❌ **Avoid when:**
- Features are highly correlated
- You need well-calibrated probabilities
- Complex non-linear relationships exist
- Regression problem

---

## 16. Interview Questions

**Q1: Why is it called "Naive" Bayes?**
> Because it naively assumes all features are conditionally independent given the class, which is almost never true. Despite this, it works surprisingly well.

**Q2: What is the Zero-Frequency Problem and how do you fix it?**
> If a word/feature never appears with a class in training data, its probability = 0, making the entire product 0. Fix: Laplace smoothing adds a pseudo-count α to every feature count.

**Q3: Does Naive Bayes require feature scaling?**
> Multinomial and Bernoulli NB — No, they work with counts/binary. Gaussian NB — Scaling is not required (it models distributions), but can help numerical stability. Never required as strongly as KNN/SVM.

**Q4: What's the difference between Gaussian, Multinomial, and Bernoulli NB?**
> Gaussian: continuous features, assumes normal distribution. Multinomial: count features (word frequencies), uses multinomial distribution. Bernoulli: binary features (word presence/absence), explicitly penalizes absent features.

**Q5: Why are Naive Bayes probabilities poorly calibrated?**
> The independence assumption causes the model to be overconfident — probabilities tend toward 0 or 1 even when true probability is moderate. Use `CalibratedClassifierCV` if calibrated probabilities are needed.

**Q6: Is there gradient descent in Naive Bayes?**
> No. Parameters are computed analytically in closed form: priors from class counts, means/variances from sample statistics. No optimization needed.

**Q7: How does Naive Bayes handle missing values?**
> Naturally! Missing features are simply omitted from the product. The algorithm works with available features only.

**Q8: What is the difference between MultinomialNB and ComplementNB?**
> MultinomialNB estimates P(feature | class). ComplementNB estimates P(feature | NOT class). CNB performs better on imbalanced datasets.

**Q9: What does `fit_prior=False` do?**
> Sets equal prior probability for all classes (1/K). Use this when you want to ignore class imbalance or when you believe all classes are equally likely.

**Q10: Can Naive Bayes do online/incremental learning?**
> Yes! `partial_fit()` allows training on batches of data — ideal for streaming data or when dataset doesn't fit in memory.

**Q11: Naive Bayes vs Logistic Regression — when to use which?**
> Naive Bayes: small data, high-dimensional (text), fast training, streaming. Logistic Regression: larger data, better probability calibration, handles feature correlations, generally more accurate when data is sufficient.

**Q12: What is the log-probability trick?**
> When multiplying many small probabilities (< 1), floating point underflow can make the product = 0. Taking log converts products to sums, staying numerically stable.

---

## 17. Resources

### 📘 Andrew Ng — ML Specialization (Coursera)
- **Course 1, Week 3**: Classification and probability-based models
- **NLP Specialization (separate)** — Naive Bayes for sentiment analysis (Week 2)
  - Exact Bayes formula derivation
  - Log-likelihood ratio for text classification
  - Laplace smoothing in practice
- 🔗 https://www.coursera.org/specializations/machine-learning-introduction
- 🔗 https://www.coursera.org/specializations/natural-language-processing

### 📗 Hands-On Machine Learning (Aurélien Géron)
- **Chapter 3**: Classification — covers evaluation metrics applicable to NB
- **Chapter 4**: Training models — generative vs discriminative framing
- Naive Bayes used as spam classifier example throughout
- **Exercise**: Titanic survival with Naive Bayes
- 🔗 https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/

### 🎬 StatQuest with Josh Starmer (YouTube)
- **"Naive Bayes, Clearly Explained"** → Core intuition and formula
- **"Gaussian Naive Bayes, Clearly Explained"** → Continuous features
- **"Bayes' Theorem, Clearly Explained"** → Foundation
- **"How Naive Bayes is used for NLP"** → Spam/text application
- 🔗 https://www.youtube.com/@statquest

### 📙 Introduction to Statistical Learning (ISLP — Python Edition)
- **Chapter 4.4**: Naive Bayes Classifier (4th edition onward)
- **Chapter 4**: Linear Discriminant Analysis (related generative model)
- **Chapter 4.5**: Comparison of classification methods
- Discussion of when NB beats LDA and logistic regression
- Free PDF: 🔗 https://www.statlearning.com/

### 📜 Scikit-learn Documentation
- GaussianNB: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
- MultinomialNB: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
- BernoulliNB: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html
- ComplementNB: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.ComplementNB.html
- User Guide: https://scikit-learn.org/stable/modules/naive_bayes.html
- Text feature extraction: https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction

### 📺 Additional Resources
- **3Blue1Brown** — "Bayes Theorem" video → best visual intuition for the formula
- **Sebastian Raschka** — "Naive Bayes and Text Classification" blog post
- **Chris Manning (Stanford)** — NLP lecture notes on NB classifiers
- Original Paper: **"A Bayesian Approach to Filtering Junk E-Mail"** (Sahami et al., 1998)

---

## 🎯 Quick Cheat Sheet

```
Algorithm:      Probabilistic, generative, eager
Training:       O(N * d) — count/compute statistics
Prediction:     O(d * C) — compute posterior for each class
Memory:         O(d * C) — stores parameters only (very efficient!)

Variants:
  GaussianNB    → continuous features (Iris, tabular numeric)
  MultinomialNB → word counts / TF-IDF (text classification)
  BernoulliNB   → word presence/absence (short texts, binary)
  ComplementNB  → imbalanced text datasets (recommended over Multinomial)
  CategoricalNB → nominal categorical features

Key formula:
  ŷ = argmax_c [ log P(Y=c) + Σ log P(X_j | Y=c) ]

Smoothing:  alpha=1.0 (Laplace) prevents zero-probability features
Priors:     fit_prior=True (from data) or set manually for imbalance

No gradient descent — parameters computed analytically:
  Priors   → class frequency
  Means    → sample mean per class per feature (Gaussian)
  Variances→ sample variance + var_smoothing (Gaussian)
  Counts   → Laplace-smoothed feature counts (Multinomial/Bernoulli)

Bias-Variance: High bias (naive assumption), Low variance (few params)
Scaling:       Not required for Multinomial/Bernoulli; optional for Gaussian
```

---

*Notes compiled from: Andrew Ng ML + NLP Specializations · Hands-On ML (Géron) · StatQuest · ISLP (James et al.) · Scikit-learn Docs*
