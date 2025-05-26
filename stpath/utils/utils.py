import torch
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import random
import datetime
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, homogeneity_score, normalized_mutual_info_score


def create_row_index_tensor(csr_matrix):
    indptr = csr_matrix.indptr
    row_indices = []
    for row in range(len(indptr) - 1):
        start = indptr[row]
        end = indptr[row + 1]
        row_indices.extend([row] * (end - start))
    return np.array(row_indices)


def clustering_metrics(target, pred, mode="AMI"):
    """
    Evaluate clustering performance.
    
    Parameters
    ----------
    target
        Key in `adata.obs` where ground-truth spatial domain labels are stored.
    pred
        Key in `adata.obs` where clustering assignments are stored.
        
    Returns
    -------
    ami
        Adjusted mutual information score.
    ari
        Adjusted Rand index score.
    homo
        Homogeneity score.
    nmi
        Normalized mutual information score.
    """
    if mode == "AMI":
        return adjusted_mutual_info_score(target, pred)
    elif mode == "ARI":
        return adjusted_rand_score(target, pred)
    elif mode == "Homo":
        return homogeneity_score(target, pred)
    elif mode == "NMI":
        return normalized_mutual_info_score(target, pred)


class WarmupScheduler(LambdaLR):
    def __init__(self, optimizer, warmup_steps, base_lr):
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        super().__init__(optimizer, self.lr_lambda)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return 1.0


def symbol2ensembl_id(gene_symbol, species='human'):
    import mygene
    
    try:
        mg = mygene.MyGeneInfo()
        gene_info = mg.query(gene_symbol, species=species, fields='ensembl.gene')
        if gene_info['total'] == 0:
            return None
        return gene_info['hits'][0].get('ensembl')['gene']
    except Exception as e:
        return None


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def merge_fold_results(arr):
    aggr_dict = {}
    for dict in arr:
        for item in dict['pearson_corrs']:
            gene_name = item['name']
            correlation = item['pearson_corr']
            aggr_dict[gene_name] = aggr_dict.get(gene_name, []) + [correlation]
    
    aggr_results = []
    all_corrs = []
    for key, value in aggr_dict.items():
        aggr_results.append({
            "name": key,
            "pearson_corrs": value,
            "mean": np.mean(value),
            "std": np.std(value)
        })
        all_corrs += value
        
    mean_per_split = [d['pearson_mean'] for d in arr]    
        
    return {"pearson_corrs": aggr_results, "pearson_mean": np.mean(mean_per_split), "pearson_std": np.std(mean_per_split), "mean_per_split": mean_per_split}


def get_current_time():
    now = datetime.datetime.now()
    year = now.year % 100  # convert to 2-digit year
    month = now.month
    day = now.day
    hour = now.hour
    minute = now.minute
    second = now.second
    return f"{year:02d}-{month:02d}-{day:02d}-{hour:02d}-{minute:02d}-{second:02d}"


if __name__ == "__main__":
    print(symbol2ensembl_id("CDK1"))
    print(symbol2ensembl_id("0610005C13Rik", species='mouse'))
