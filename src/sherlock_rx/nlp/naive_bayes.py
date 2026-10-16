"""Multinomial Naive Bayes, from scratch.

    log P(c | d) ∝ log P(c) + Σ_t  x_t · log P(t | c)
    P(t | c)     = (count(t, c) + α) / (Σ_t' count(t', c) + α·|V|)      (Laplace / Lidstone)

Scores stay in log space; probabilities come from a log-sum-exp softmax so long
utterances never underflow. Works on any non-negative sparse matrix (counts or TF-IDF).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


class MultinomialNB:
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.classes_: np.ndarray = np.array([])
        self.class_log_prior_: np.ndarray = np.array([])
        self.feature_log_prob_: np.ndarray = np.array([])

    def fit(self, x: sp.csr_matrix, y: np.ndarray | list[str]) -> MultinomialNB:
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        onehot = (y[:, None] == self.classes_[None, :]).astype(np.float64)  # [n, C]
        counts = np.asarray((sp.csr_matrix(onehot.T) @ x).todense())  # [C, V]
        smoothed = counts + self.alpha
        self.feature_log_prob_ = np.log(smoothed) - np.log(smoothed.sum(axis=1, keepdims=True))
        self.class_log_prior_ = np.log(onehot.sum(axis=0) / len(y))
        return self

    def joint_log_likelihood(self, x: sp.csr_matrix) -> np.ndarray:
        jll: np.ndarray = np.asarray(x @ self.feature_log_prob_.T) + self.class_log_prior_
        return jll

    def predict_proba(self, x: sp.csr_matrix) -> np.ndarray:
        jll = self.joint_log_likelihood(x)
        jll -= jll.max(axis=1, keepdims=True)  # log-sum-exp trick
        p = np.exp(jll)
        return np.asarray(p / p.sum(axis=1, keepdims=True))

    def predict(self, x: sp.csr_matrix) -> np.ndarray:
        return np.asarray(self.classes_[self.joint_log_likelihood(x).argmax(axis=1)])
