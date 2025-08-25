import numpy as np

np.random.seed(42)
import matplotlib.pyplot as plt

# Create 3 blobs with different means and covariances
means = [(-4, -4), (4, -4), (0, 5)]
covs = [np.eye(2) * 1, np.eye(2) * 1.5, np.array([[2, 0.5], [0.5, 1]])]
data = np.vstack(
    [np.random.multivariate_normal(mu, cov, size=200) for mu, cov in zip(means, covs)]
)
labels = np.concatenate([[i] * 200 for i in range(len(means))])

x, y = np.linspace(-8, 8, 300), np.linspace(-8, 8, 300)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

# Precompute inverses for Mahalanobis
inv_covs = [np.linalg.inv(cov) for cov in covs]

# Compute dist for each metric to each blob center
Z_euclals = [np.sqrt((X - mu[0]) ** 2 + (Y - mu[1]) ** 2) for mu in means]
Z_eucl = np.min(Z_euclals, axis=0)

# Cosine: pick reference direction per blob (mu vector)
Z_cosals = []
for mu in means:
    ref = np.array(mu)
    vecs = pos - mu
    norms = np.linalg.norm(vecs, axis=-1) * np.linalg.norm(ref)
    Z = 1 - np.sum(vecs * ref, axis=-1) / norms
    Z_cosals.append(np.nan_to_num(Z))
Z_cos = np.min(Z_cosals, axis=0)

# Mahalanobis:
Z_mahals = []
for mu, inv_cov in zip(means, inv_covs):
    diff = pos - mu
    Zm = np.einsum("...i,ij,...j->...", diff, inv_cov, diff)
    Z_mahals.append(Zm)
Z_mah = np.min(Z_mahals, axis=0)


titles = ["Euclidean", "Cosine dist", "Mahalanobis"]
Zs = [Z_eucl, Z_cos, Z_mah]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, Z, title in zip(axes, Zs, titles):
    ax.contour(X, Y, Z, levels=15, cmap="cividis")
    ax.scatter(data[:, 0], data[:, 1], c=labels, s=10, cmap="tab10", alpha=0.6)
    ax.set_title(title)
    ax.set_aspect("equal")
plt.tight_layout()
plt.show()
