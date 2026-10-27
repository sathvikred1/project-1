"""Binary logistic loss with an unregularized intercept."""
from dataclasses import dataclass

import numpy as np
from scipy.special import expit
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class LogisticProblem:
    x_train: np.ndarray
    y_train: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray
    l2: float = 0.001

    @property
    def dimension(self):
        return self.x_train.shape[1]

    def gradient(self, parameters, indices):
        x, y = self.x_train[indices], self.y_train[indices]
        gradient = x.T @ (expit(x @ parameters) - y) / len(indices)
        gradient[:-1] += self.l2 * parameters[:-1]
        return gradient

    def loss(self, parameters, validation=False):
        x, y = ((self.x_validation, self.y_validation) if validation
                else (self.x_train, self.y_train))
        logits = x @ parameters
        # logaddexp avoids overflow for large positive or negative logits.
        return float(np.mean(np.logaddexp(0, logits) - y * logits)
                     + 0.5 * self.l2 * np.dot(parameters[:-1], parameters[:-1]))

    def evaluate(self, parameters):
        return {
            "training_loss": self.loss(parameters),
            "validation_loss": self.loss(parameters, validation=True),
            "training_accuracy": float(np.mean((self.x_train @ parameters >= 0) == self.y_train)),
            "validation_accuracy": float(np.mean((self.x_validation @ parameters >= 0) == self.y_validation)),
        }


def make_problem(samples=6000, features=30, dataset_seed=2026, l2=0.001):
    x, y = make_classification(
        n_samples=samples, n_features=features, n_informative=20,
        n_redundant=5, n_clusters_per_class=2, class_sep=1.1,
        flip_y=0.04, random_state=dataset_seed,
    )
    xt, xv, yt, yv = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=dataset_seed + 1,
    )
    scaler = StandardScaler().fit(xt)  # Validation never enters preprocessing fit.
    xt, xv = scaler.transform(xt), scaler.transform(xv)
    # The final coordinate is the intercept; it receives no L2 penalty.
    xt = np.ascontiguousarray(np.column_stack([xt, np.ones(len(xt))]))
    xv = np.ascontiguousarray(np.column_stack([xv, np.ones(len(xv))]))
    return LogisticProblem(xt, yt, xv, yv, l2)
