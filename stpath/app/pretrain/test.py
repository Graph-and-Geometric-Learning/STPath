import torch
import numpy as np
from time import time
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import auc, precision_recall_curve, roc_curve, f1_score


def metric_func(preds_all: np.ndarray, y_test: np.ndarray, gene_idx: list, gene_names: list, n_hvgs=None):
    errors = []
    r2_scores = []
    pearson_corrs = []
    pearson_genes = []
    if n_hvgs is None:
        n_hvgs = [len(gene_idx)]
    
    n_nan_genes = 0
    for i, target in enumerate(gene_idx):
        preds = preds_all[:, target]
        target_vals = y_test[:, target]

        errors.append(float(np.mean((preds - target_vals) ** 2)))
        r2_scores.append(float(1 - np.sum((target_vals - preds) ** 2) / np.sum((target_vals - np.mean(target_vals)) ** 2)))
        pearson_corr, _ = pearsonr(target_vals, preds)
        pearson_corrs.append(pearson_corr)

        if np.isnan(pearson_corr):
            n_nan_genes += 1

        score_dict = {
            'name': gene_names[i],
            'pearson_corr': float(pearson_corr),
        }
        pearson_genes.append(score_dict)

    if n_nan_genes > 0:
        print(f"Warning: {n_nan_genes} genes have NaN Pearson correlation")

    pearson_dict = {}
    for n_hvg in n_hvgs[:-1]:
        pearson_dict = {
            f"pearson_mean_{n_hvg}": float(np.mean(pearson_corrs[:n_hvg])),
            f"pearson_std_{n_hvg}": float(np.std(pearson_corrs[:n_hvg])),
        }

    res_dict = {'l2_errors': list(errors), 
            'r2_scores': list(r2_scores),
            'pearson_corrs': pearson_genes,
            'pearson_mean': float(np.mean(pearson_corrs[:n_hvgs[-1]])),
            'pearson_std': float(np.std(pearson_corrs[:n_hvgs[-1]])),
            'l2_error_q1': float(np.percentile(errors, 25)),
            'l2_error_q2': float(np.median(errors)),
            'l2_error_q3': float(np.percentile(errors, 75)),
            'r2_score_q1': float(np.percentile(r2_scores, 25)),
            'r2_score_q2': float(np.median(r2_scores)),
            'r2_score_q3': float(np.percentile(r2_scores, 75))
        }
    res_dict.update(pearson_dict)
    return res_dict


@torch.no_grad()
def test(args, model, loader, return_all=False):
    model.eval()
    all_pred, all_gt = [], []
    res_dict = {}

    n_eval_hvg = args.n_eval_hvg if hasattr(args, 'n_eval_hvg') else None

    for step, batch in enumerate(loader):
        batch = [x.to(args.device) for x in batch]
        img_features, coords, gene_exp, _, tech_ids, organ_ids, _, _, batch_idx = batch
        assert batch_idx.max() == 0, "Batch size must be 1 for inference"

        # generate masked gene expression
        masked_ge_tokens = loader.dataset.generate_masked_ge_tokens(img_features.shape[0])
        masked_ge_tokens = masked_ge_tokens.to(img_features.device)

        pred = model.prediction_head(
            img_tokens=img_features,
            coords=coords,
            ge_tokens=masked_ge_tokens,
            batch_idx=batch_idx,
            tech_tokens=tech_ids,
            organ_tokens=organ_ids,
        )[:, loader.dataset.get_hvg_names(step)]
        gene_exp = gene_exp[:, loader.dataset.get_hvg_names(step)]

        # batch size == 1 by default
        cur_pred = pred.cpu().numpy()
        cur_gt = gene_exp.cpu().numpy()
    
        cur_res_dict = metric_func(cur_pred, cur_gt, list(range(cur_pred.shape[1])), loader.dataset.get_hvg_names(step), n_hvgs=n_eval_hvg)
        cur_res_dict.update({'n_test': len(cur_gt)})
        res_dict[loader.dataset.get_dataset_name(step)] = cur_res_dict

        all_pred.append(cur_pred)
        all_gt.append(cur_gt)
    
    # test the performance on all datasets
    all_pearson = []
    all_hvg_pearson = {}
    for name in res_dict.keys():
        all_pearson.extend([d["pearson_corr"] for d in res_dict[name]['pearson_corrs']])
        if n_eval_hvg is not None:
            for n_hvg in args.n_eval_hvg[:-1]:
                if n_hvg not in all_hvg_pearson:
                    all_hvg_pearson[n_hvg] = []
                all_hvg_pearson[n_hvg].extend([d["pearson_corr"] for d in res_dict[name]['pearson_corrs'][:n_hvg]])

    # all_pearson = np.concatenate(all_pearson)
    res_dict["all"] = {
        "pearson_mean": float(np.mean(all_pearson)),
        "pearson_std": float(np.std(all_pearson)),
    }

    if n_eval_hvg is not None:
        pearson_dict = {}
        for n_hvg in args.n_eval_hvg[:-1]:
            pearson_dict.update({
                f"pearson_mean_{n_hvg}": float(np.mean(all_hvg_pearson[n_hvg])),
                f"pearson_std_{n_hvg}": float(np.std(all_hvg_pearson[n_hvg])),
            })
        res_dict["all"].update(pearson_dict)

    if return_all:
        return res_dict, {'preds_all': all_pred, 'targets_all': all_gt}
    return res_dict
