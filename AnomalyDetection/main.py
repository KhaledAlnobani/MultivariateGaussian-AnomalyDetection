import numpy as np
import matplotlib.pyplot as plt

X_train = np.load("data/X_part1.npy")
# validation set
X_val = np.load("data/X_val_part1.npy")
y_val = np.load("data/y_val_part1.npy")



def estimate_gaussian(X):

    m, n = X.shape
    mu = 1 / m * np.sum(X, axis=0)
    var = 1 / m * np.sum((X - mu) ** 2, axis=0)

    return mu, var


def multivariate_gaussian(X, mu, var):
    """
    Computes the probability
    density function of the examples X under the multivariate gaussian
    distribution with parameters mu and var. If var is a matrix, it is
    treated as the covariance matrix. If var is a vector, it is treated
    as the var values of the variances in each dimension (a diagonal
    covariance matrix
    """

    k = len(mu)

    if var.ndim == 1:
        var = np.diag(var)

    X = X - mu
    p = (2 * np.pi) ** (-k / 2) * np.linalg.det(var) ** (-0.5) * \
        np.exp(-0.5 * np.sum(np.matmul(X, np.linalg.pinv(var)) * X, axis=1))

    return p


def select_threshold(y, p):
    best_f1 = 0
    best_epsilon = 0
    step_size = (max(p) - min(p)) / 1000

    for epsilon in np.arange(min(p), max(p), step_size):
        predictions = (p < epsilon)
        tp = np.sum((predictions == 1) & (y == 1))
        fn = np.sum((predictions == 0) & (y == 1))
        fp = np.sum((predictions == 1) & (y == 0))

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)

        if f1 > best_f1:
            best_f1 = f1
            best_epsilon = epsilon

    return best_epsilon, best_f1


mu, var = estimate_gaussian(X_train)
p = multivariate_gaussian(X_val, mu, var)
epsilon, F1 = select_threshold(y_val, p)

outliers = p < epsilon

fig, ax = plt.subplots()

ax.scatter(X_train[:, 0], X_train[:, 1], marker="x", c="b")
ax.plot(X_train[outliers, 0], X_train[outliers, 1], 'ro',
        markersize=14, markerfacecolor='none', markeredgewidth=1)
ax.set_xlabel('X1')
ax.set_ylabel('X2')

fig.savefig("figures/fig1.png")