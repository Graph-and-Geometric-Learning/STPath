import torch
import numpy as np
import scanpy as sc
import pandas as pd
import anndata as ad
from tqdm import tqdm
from scipy.stats import pearsonr
from stpath.utils import clustering_metrics
from stpath.utils.confidence import Confidence_Utils


def mean_pearson(preds_all: np.ndarray, y_test: np.ndarray):
    return float(np.mean([pearsonr(y_test[:, i], preds_all[:, i])[0] for i in range(y_test.shape[1])]))


def metric_func(preds_all: np.ndarray, y_test: np.ndarray, gene_idx: list, gene_names: list):
    errors = []
    r2_scores = []
    pearson_corrs = []
    pearson_genes = []
    
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

    return {'l2_errors': list(errors), 
            'r2_scores': list(r2_scores),
            'pearson_corrs': pearson_genes,
            'pearson_mean': float(np.mean(pearson_corrs)),
            'pearson_std': float(np.std(pearson_corrs)),
            'l2_error_q1': float(np.percentile(errors, 25)),
            'l2_error_q2': float(np.median(errors)),
            'l2_error_q3': float(np.percentile(errors, 75)),
            'r2_score_q1': float(np.percentile(r2_scores, 25)),
            'r2_score_q2': float(np.median(r2_scores)),
            'r2_score_q3': float(np.percentile(r2_scores, 75))
        }


@torch.no_grad()
def test(args, model, loader, hvg_gene_names, hvg_gene_ids, return_all=False):
    model.eval()
    all_pred, all_gt = [], []
    res_dict = {}

    all_coords = []
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
        )[:, hvg_gene_ids]
        gene_exp = gene_exp[:, hvg_gene_ids]

        # batch size == 1 by default
        cur_pred = pred.cpu().numpy()
        cur_gt = gene_exp.cpu().numpy()

        cur_res_dict = metric_func(cur_pred, cur_gt, list(range(len(hvg_gene_ids))), hvg_gene_names)
        cur_res_dict.update({'n_test': len(cur_gt)})
        res_dict[loader.dataset.get_dataset_name(step)] = cur_res_dict

        all_pred.append(cur_pred)
        all_gt.append(cur_gt)
        all_coords.append(coords.cpu().numpy())

    # test the performance on all datasets
    all_pred_numpy = np.concatenate(all_pred, axis=0)
    all_gt_numpy = np.concatenate(all_gt, axis=0)
    cur_res_dict = metric_func(all_pred_numpy, all_gt_numpy, list(range(len(hvg_gene_ids))), hvg_gene_names)
    cur_res_dict.update({'n_test': len(all_gt_numpy)})
    res_dict["all"] = cur_res_dict
    
    # clustering evaluation
    adata_pred_all = ad.AnnData(X=all_pred_numpy)
    sc.tl.pca(adata_pred_all)
    sc.pp.neighbors(adata_pred_all, n_pcs=30, n_neighbors=30)
    sc.tl.leiden(adata_pred_all)

    adata_gt_all = ad.AnnData(X=all_gt_numpy)
    sc.tl.pca(adata_gt_all)
    sc.pp.neighbors(adata_gt_all, n_pcs=30, n_neighbors=30)
    sc.tl.leiden(adata_gt_all)

    res_dict["all"]["ari"] = float(clustering_metrics(adata_gt_all.obs['leiden'], adata_pred_all.obs['leiden'], "ARI"))
    res_dict["all"]["ami"] = float(clustering_metrics(adata_gt_all.obs['leiden'], adata_pred_all.obs['leiden'], "AMI"))
    res_dict["all"]["homo"] = float(clustering_metrics(adata_gt_all.obs['leiden'], adata_pred_all.obs['leiden'], "Homo"))
    res_dict["all"]["nmi"] = float(clustering_metrics(adata_gt_all.obs['leiden'], adata_pred_all.obs['leiden'], "NMI"))

    """boostrap confidence interval"""
    confidence_utils = Confidence_Utils()

    all_pred_numpy_hvg, all_gt_numpy_hvg = all_pred_numpy[:, :], all_gt_numpy[:, :]
    all_bootstraps = confidence_utils.bootstrap(all_gt_numpy_hvg, all_pred_numpy_hvg, num_bootstraps=100)  # a list of (labels, preds) tuples
    metrics = [{"Pearson": mean_pearson(labels, preds)} for labels, preds in tqdm(all_bootstraps, desc=f'Computing bootstraps for PCC', ncols=100)]
    res_dict["all"]["pearson_ci"] = confidence_utils.get_95_ci(metrics)

    all_gt_leiden, all_pred_leiden = adata_gt_all.obs['leiden'], adata_pred_all.obs['leiden']
    all_bootstraps = confidence_utils.bootstrap(all_gt_leiden, all_pred_leiden, num_bootstraps=100)
    metrics = [
        {"ari_ci": float(clustering_metrics(labels, preds, "ARI")),
        "ami_ci": float(clustering_metrics(labels, preds, "AMI")),
        "homo_ci": float(clustering_metrics(labels, preds, "Homo")),
        "nmi_ci": float(clustering_metrics(labels, preds, "NMI"))
        } for labels, preds in tqdm(all_bootstraps, desc=f'Computing bootstraps for Clustering', ncols=100)
    ]
    all_95_ci = confidence_utils.get_95_ci(metrics)
    res_dict["all"]["ari_ci"] = all_95_ci["ari_ci"]
    res_dict["all"]["ami_ci"] = all_95_ci["ami_ci"]
    res_dict["all"]["homo_ci"] = all_95_ci["homo_ci"]
    res_dict["all"]["nmi_ci"] = all_95_ci["nmi_ci"]

    if return_all:
        all_adata = []
        for i in range(len(loader)):
            adata = ad.AnnData(X=all_pred[i])
            adata.uns['sample_id'] = loader.dataset.get_dataset_name(i)
            adata.layers['groundtruth'] = all_gt[i]
            adata.obsm['coordinates'] = all_coords[i]
            adata.var_names = pd.Index(hvg_gene_names)
            # adata.obs_names = pd.Index(loader.dataset.barcodes)
            all_adata.append(adata)
        return res_dict, all_adata
    return res_dict
