import numpy as np
from scipy.spatial import KDTree

from .distribution_utils import get_distribution, is_return_ratio


class PatchSampler:
    def __init__(self, distribution: str='batch_128', patch_sample_method="nearest", min_samples=2):
        self.distribution = distribution
        self.distribution_func = get_distribution(distribution)
        self.patch_sample_method = patch_sample_method
        self.min_samples = min_samples

    @staticmethod
    def sample_nearest_patch(coords, num_samples, center_idx=None):
        num_samples = min(len(coords), num_samples)

        if num_samples == len(coords):
            return np.arange(len(coords))

        # Build a KDTree for efficient nearest neighbor searches
        tree = KDTree(coords)
        
        if center_idx is None:
            # Randomly choose an index as the starting point for the patch
            center_idx = np.random.randint(0, len(coords))
        center_coord = coords[center_idx]
        
        # Query the nearest 'num_samples' points including the center itself
        _, idx_nearest = tree.query(center_coord, k=num_samples)

        # Fetch the coordinates of these nearest points
        return idx_nearest

    def get_distribution_expectation(self):
        return np.mean([self.distribution() for _ in range(10000)])

    def __call__(self, coords):
        # sample the number of spots
        if not is_return_ratio(self.distribution):
            total_samples = max(self.min_samples, int(self.distribution_func()))
        else:
            total_samples = max(self.min_samples, int(len(coords) * self.distribution_func()))
        
        # sample spots
        if self.patch_sample_method == "nearest":
            return PatchSampler.sample_nearest_patch(coords, total_samples)
        elif self.patch_sample_method == "random":
            return np.random.choice(len(coords), total_samples)


if __name__ == "__main__":
    # Test the patch sampler
    coords = np.random.rand(100, 2)
    sampler = PatchSampler("beta_3_1")
    print(sampler(coords).shape)
    print(sampler.get_distribution_expectation())