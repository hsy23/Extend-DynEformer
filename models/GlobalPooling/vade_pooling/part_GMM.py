from sklearn.metrics import adjusted_rand_score
import numpy as np
from sklearn.mixture import GaussianMixture


class ImprovedIncrementalGMM:
    def __init__(self, n_components, dim, alpha=0.1):
        self.n_components = n_components
        self.alpha = alpha  # Learning rate for incremental update
        self.weights_ = np.ones(n_components) / n_components
        self.means_ = np.random.randn(n_components, dim)  # Initialize with random values
        self.covariances_ = np.ones((n_components, dim))  # Initialize to identity

    def partial_fit(self, X):
        # E-step: Estimate posterior probabilities
        posterior_prob = self._estimate_posterior(X)

        # M-step: Update model parameters
        Nk = np.sum(posterior_prob, axis=0)  # Sum along the rows
        self.weights_ = (1 - self.alpha) * self.weights_ + self.alpha * Nk / X.shape[0]

        new_means = np.sum(posterior_prob[:, :, np.newaxis] * X[:, np.newaxis, :], axis=0) / Nk[:, np.newaxis]
        new_covariances = np.sum(posterior_prob[:, :, np.newaxis] * (X[:, np.newaxis, :] - new_means) ** 2,
                                 axis=0) / Nk[:, np.newaxis]

        self.means_ = (1 - self.alpha) * self.means_ + self.alpha * new_means
        self.covariances_ = (1 - self.alpha) * self.covariances_ + self.alpha * new_covariances

    def _estimate_posterior(self, X):
        n_samples, n_features = X.shape
        X = X[:, np.newaxis, :]  # Reshape X to (n_samples, 1, n_features)

        log_prob = np.log(self.weights_) - 0.5 * np.sum(
            np.log(self.covariances_) + (X - self.means_) ** 2 / self.covariances_, axis=2)

        log_prob -= np.max(log_prob, axis=1, keepdims=True)  # Subtract the maximum for numerical stability
        prob = np.exp(log_prob)
        prob /= np.sum(prob, axis=1, keepdims=True)  # Normalize along the rows
        return prob

    def predict(self, X):
        posterior_prob = self._estimate_posterior(X)
        return np.argmax(posterior_prob, axis=1)


if __name__ == "__main__":
    # Generate synthetic data
    X1 = np.random.normal(loc=-1, scale=1, size=(100, 3))
    X2 = np.random.normal(loc=1, scale=1, size=(100, 3))

    # Test ImprovedIncrementalGMM
    improved_gmm = ImprovedIncrementalGMM(n_components=3, alpha=0.5, dim=3)
    improved_gmm.partial_fit(X1)
    improved_gmm.partial_fit(X2)
    improved_res = improved_gmm.predict(X1)

    # Test sklearn's GaussianMixture
    gmm = GaussianMixture(n_components=3, covariance_type='diag')
    gmm.fit(np.concatenate([X1, X2], 0))
    sklearn_res = gmm.predict(X1)

    # Calculate the Adjusted Rand Index
    ari_score = adjusted_rand_score(improved_res, sklearn_res)
    print(ari_score)
    # print("Weights:", gmm.weights_)
    # print("Means:", gmm.means_)
    # print("Covariances:", gmm.covariances_)
