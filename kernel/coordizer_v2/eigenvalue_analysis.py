import numpy as np

from kernel.coordizer_v2.compress import compress
from kernel.coordizer_v2.harvest import HarvestResult


def synthetic_pipeline_test():
    n_samples = 1000
    n_features = 64
    token_fingerprints = {}
    for i in range(n_samples):
        token_fingerprints[i] = np.random.dirichlet(alpha=np.ones(n_features))
    harvest = HarvestResult(token_fingerprints=token_fingerprints, vocab_size=n_features)
    compressed_result = compress(harvest)
    eigenvalues = compressed_result.eigenvalues
    return eigenvalues


if __name__ == "__main__":
    eigenvalues = synthetic_pipeline_test()
    print("Synthetic pipeline test eigenvalues:", eigenvalues)
