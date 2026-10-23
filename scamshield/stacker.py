"""Sign-constrained logistic regression regularised towards domain-knowledge priors.

Why not a plain LogisticRegression stacker?
  * The first version used one, and on our (partly synthetic) data it learned a
    NEGATIVE weight for "asks you to share OTP/PIN". That's because the rules
    are collinear with the text model and with each other — a known failure
    mode of learning many weights from little data.
  * Rules that never fire in training (e.g. international sender) get weight 0
    under normal L2, so the rule is silently useless.

Fix: minimise   mean_log_loss(w) + (lambda/2) * ||w - w_prior||^2
     subject to w >= 0 for scam signals and w <= 0 for trust signals.
With little evidence a weight stays at its expert prior; with lots of evidence
the data moves it. (Equivalent to MAP estimation with a Gaussian prior centred
on the expert weights, plus sign constraints.)
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


class PriorLogisticRegression:
    def __init__(self, prior: np.ndarray, lower: np.ndarray, upper: np.ndarray, lam: float = 0.05,
                 class_weight: str | None = "balanced"):
        self.prior, self.lower, self.upper = np.asarray(prior, float), lower, upper
        self.lam, self.class_weight = lam, class_weight

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        sw = np.ones_like(y)
        if self.class_weight == "balanced":
            for c in (0, 1):
                sw[y == c] = len(y) / (2 * max((y == c).sum(), 1))
        sw /= sw.sum()

        def loss(params):
            w, b = params[:-1], params[-1]
            z = X @ w + b
            # numerically stable log(1+exp(z)) - y*z
            ll = np.logaddexp(0, z) - y * z
            p = 1 / (1 + np.exp(-z))
            g = X.T @ (sw * (p - y)) + self.lam * (w - self.prior)
            return (sw * ll).sum() + 0.5 * self.lam * ((w - self.prior) ** 2).sum(), np.append(g, (sw * (p - y)).sum())

        x0 = np.append(np.clip(self.prior, self.lower, self.upper), 0.0)
        bounds = list(zip(self.lower, self.upper, strict=True)) + [(None, None)]
        res = minimize(loss, x0, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 2000})
        self.coef_, self.intercept_ = res.x[:-1][None, :], np.array([res.x[-1]])
        return self

    def decision_function(self, X):
        return np.asarray(X, float) @ self.coef_[0] + self.intercept_[0]

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-self.decision_function(X)))
        return np.column_stack([1 - p, p])
