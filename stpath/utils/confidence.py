# source: https://github.com/mahmoodlab/Patho-Bench/blob/main/patho_bench/experiments/BaseExperiment.py
import numpy as np


class Confidence_Utils:
    def __init__(self):
        pass

    def bootstrap(self, labels, preds, num_bootstraps=100):
        '''
        Perform bootstrapping and calculate 95% CI.
        '''
        bootstraps = []
        for _ in range(num_bootstraps):
            idx = np.random.choice(len(labels), len(labels), replace=True)
            labels_ = np.array([labels[i] for i in idx])
            preds_ = np.array([preds[i] for i in idx])
            bootstraps.append((labels_, preds_))
        return bootstraps
    
    def get_95_ci(self, all_scores_across_folds):
        '''
        Calculate 95% CI.

        Args:
            all_scores_across_folds (list[dict]): List of scores across bootstraps, keys are metric names.
            
        Returns:
            mean (float): Mean of bootstrapped scores.
            lower (float): Lower bound of 95% CI.
            upper (float): Upper bound of 95% CI.
        '''
        summary = {}
        for key in all_scores_across_folds[0].keys():
            if any([score[key] is None for score in all_scores_across_folds]): # Sometimes scores are mathematically noncomputable, indicated by None
                summary[key] = {
                    'mean': None,
                    'lower': None,
                    'upper': None,
                    'formatted': 'N/A'
                }
                continue
            mean = np.mean([score[key] for score in all_scores_across_folds])
            lower = np.percentile([score[key] for score in all_scores_across_folds], 2.5)
            upper = np.percentile([score[key] for score in all_scores_across_folds], 97.5)
            summary[key] = {
                'mean': mean,
                'lower': lower,
                'upper': upper,
                'formatted': f'{mean:.3f} ({lower:.3f}-{upper:.3f})'
            }

        return summary


if __name__ == '__main__':
    # shape [3, 5]
    labels = np.random.random((3, 5))
    preds = np.random.random((3, 5))
    print(labels.shape, preds.shape)
    print(labels)
    print(preds)
    confidence_utils = Confidence_Utils()
    bootstraps = confidence_utils.bootstrap(labels, preds, num_bootstraps=100)
    print(len(bootstraps))
    print(bootstraps[0][0], bootstraps[0][1])
