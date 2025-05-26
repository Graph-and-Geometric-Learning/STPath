import os
import json
import wandb
import argparse
import numpy as np
import pandas as pd

import torch

from stpath.data.sampling_utils import PatchSampler
from stpath.data.normalize_utils import get_normalize_method
from stpath.tokenization import GeneExpTokenizer, ImageTokenizer, IDTokenizer, TokenizerTools, AnnotationTokenizer

from stpath.model.model import STFM
from stpath.data.dataset import DatasetPath
from stpath.app.task.imputation.metric import test
from stpath.utils import set_random_seed, get_current_time
from stpath.data.imputation_dataset import SpotImputationDataset

import warnings
warnings.filterwarnings("ignore")


def main(args, hvg_list, niche_dict, test_sample_ids, tokenizer):
    normalize_method = get_normalize_method(args.normalize_method)

    # validation set
    sample_id_paths = [
        DatasetPath(
            name=sample_id,
            source=niche_dict[sample_id]["source"],
            embed_path=os.path.join(args.source_dataroot, f"embeddings/{sample_id}", args.feature_encoder, f"fp32/{sample_id}.h5"),
            h5ad_path=os.path.join(args.source_dataroot, f"st/{sample_id}.h5ad"),
            split_path=os.path.join(args.split_save_path, f"{args.dataset}/{sample_id}_new.json"),
        ) for sample_id in test_sample_ids
    ]
    test_dataset = SpotImputationDataset(
        sample_id_paths,
        meta_data_dict=niche_dict, 
        normalize_method=normalize_method, 
        tokenizer=tokenizer, 
        patch_sampler=PatchSampler("constant_1.0"), 
    )

    device = args.device
    model = STFM(args).to(device)
    model.load_state_dict(torch.load(args.checkpoint_path))
    print(f"Model loaded from {args.checkpoint_path}")

    hvg_gene_ids = tokenizer.ge_tokenizer.symbol2id(hvg_list)

    val_perf_dict_all = {} 
    for masked_ratio in args.masked_ratios:
        test_dataset.load_masked_index(masked_ratio)

        val_perf_dict, pred_dump = test(args, model, test_dataset, hvg_list, hvg_gene_ids, return_all=True)
        for patch_name, dataset_res in val_perf_dict.items():
            os.makedirs(os.path.join(args.save_dir, str(masked_ratio)), exist_ok=True)

            with open(os.path.join(args.save_dir, str(masked_ratio), f'{patch_name}_results.json'), 'w') as f:
                json.dump(dataset_res, f, sort_keys=True, indent=4)
            torch.save(pred_dump, os.path.join(args.save_dir, str(masked_ratio), 'results.pth'))
        
        val_perf_dict_all[masked_ratio] = val_perf_dict
    return val_perf_dict_all


def run(args):
    np.random.seed(args.seed)

    split_dir = os.path.join(args.source_dataroot, "hest-bench", args.dataset, 'splits')
    train_df = pd.read_csv(os.path.join(split_dir, f'train_{0}.csv'))
    test_df = pd.read_csv(os.path.join(split_dir, f'test_{0}.csv'))
    benchmark_sample_ids = train_df['sample_id'].tolist() + test_df['sample_id'].tolist()

    with open(os.path.join(args.source_dataroot, "hest-bench", args.dataset, "var_50genes.json")) as f:
        hvg_list = json.load(f)['genes']

    meta_data = pd.read_csv(os.path.join(args.source_dataroot, 'HEST_v1_1_0.csv'))
    niche_dict = {
        row["id"]: {"tech": row["st_technology"], "organ": row["organ"], "specie": row["species"], "source": "hest"}
        for _, row in meta_data.iterrows()
    }

    tokenizer = TokenizerTools(
        ge_tokenizer=GeneExpTokenizer(args.gene_voc_path),
        image_tokenizer=ImageTokenizer(feature_dim=args.feature_dim),
        tech_tokenizer=IDTokenizer(id_type="tech"), 
        specie_tokenizer=IDTokenizer(id_type="specie"), 
        organ_tokenizer=IDTokenizer(id_type="organ"),
        cancer_anno_tokenizer=AnnotationTokenizer(id_type="disease"),
        domain_anno_tokenizer=AnnotationTokenizer(id_type="domain"),
    )

    args.n_genes = tokenizer.ge_tokenizer.n_tokens
    args.n_tech = tokenizer.tech_tokenizer.n_tokens
    args.n_species = tokenizer.specie_tokenizer.n_tokens
    args.n_organs = tokenizer.organ_tokenizer.n_tokens
    args.n_cancer_annos = tokenizer.cancer_anno_tokenizer.n_tokens
    args.n_domain_annos = tokenizer.domain_anno_tokenizer.n_tokens
    print(f"n_genes: {args.n_genes}, n_tech: {args.n_tech}, n_species: {args.n_species}, n_organs: {args.n_organs}, n_cancer_annos: {args.n_cancer_annos}, n_domain_annos: {args.n_domain_annos}")

    return main(args, hvg_list, niche_dict, benchmark_sample_ids, tokenizer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--use_wandb', default=False)
    parser.add_argument('--datasets', nargs='+', default=["LUNG"], help="LUNG, READ, HCC")
    parser.add_argument('--source_dataroot', default="/home/ti.huang/project/single_cell_dataset/hest")
    parser.add_argument('--save_dir', type=str, default="/home/ti.huang/project/stfm/stpath/finetune/spot_imputation/test")
    parser.add_argument('--feature_encoder', type=str, default='gigapath', help="uni_v1_official | resnet50_trunc | ciga | gigapath")
    parser.add_argument('--normalize_method', type=str, default="log1pv2", help="log1pv2 | log1pv2_smooth")
    parser.add_argument('--exp_code', type=str, default="hest_stimagev3")

    parser.add_argument('--checkpoint_dir', type=str, default='/home/ti.huang/project/stfm/stpath/backup')
    parser.add_argument('--checkpoint_ids', nargs='+', type=str, default=['stfm'])

    # dataset hyperparameters
    parser.add_argument('--masked_ratios', nargs='+', default=[0.95, 0.9, 0.8])
    parser.add_argument('--split_save_path', type=str, default='/home/ti.huang/STFlow_pretrain/token/spot_imputation')
    parser.add_argument('--gene_voc_path', type=str, default="/home/ti.huang/STPath/utils_data/symbol2ensembl.json", help='hest | stimage_1k4m')

    # training hyperparameters
    parser.add_argument('--device', type=int, default=0)

    # model hyperparameters
    parser.add_argument('--backbone', type=str, default="spatial_transformer")
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--attn_dropout', type=float, default=0.1)
    parser.add_argument('--mlp_ratio', type=float, default=2.0)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--feature_dim', type=int, default=1024, help="uni:1024, ciga:512")
    parser.add_argument('--norm', type=str, default='layer', help="batch | layer")
    parser.add_argument('--activation', type=str, default='gelu', help="relu | gelu | swiglu")
    args = parser.parse_args()

    args.feature_dim = {
        "uni_v1_official": 1024,
        "gigapath": 1536,
        "ciga": 512,
    }[args.feature_encoder]

    set_random_seed(args.seed)

    if args.exp_code is None:
        exp_code = f"{args.backbone}::{get_current_time()}"
    else:
        exp_code = args.exp_code + f"_{args.feature_encoder}" + f"_{args.backbone}::{get_current_time()}"
    
    save_dir = os.path.join(args.save_dir, exp_code)
    os.makedirs(save_dir, exist_ok=True)

    if args.datasets[0] == "all":
        args.datasets = ["LUNG", "HCC", "COAD", "SKCM", "PAAD", "READ", "LYMPH_IDC", "PRAD", "IDC", "CCRCC"]

    for dataset in args.datasets:
        current_save_dir = os.path.join(save_dir, f"{dataset}")
        os.makedirs(current_save_dir, exist_ok=True)
        os.makedirs(os.path.join(args.split_save_path, f"{dataset}"), exist_ok=True)

        args.dataset = dataset

        print(args)
        print(f"Save dir: {save_dir}")

        if dataset == "IDC":
            args.device = "cpu"

        checkpoint_perfs = []
        for checkpoint_id in args.checkpoint_ids:
            args.checkpoint_path = os.path.join(args.checkpoint_dir, f"{checkpoint_id}.pth")

            args.save_dir = os.path.join(current_save_dir, f'{checkpoint_id}')
            os.makedirs(args.save_dir, exist_ok=True)

            with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
                json.dump(vars(args), f, sort_keys=True, indent=4)

            result = run(args)

            res_all_masked_ratios = []
            for masked_ratio, res in result.items():
                res_all_masked_ratios.append({
                    'checkpoint_id': checkpoint_id,
                    'masked_ratio': masked_ratio,
                    'pearson_mean': res['all']['pearson_mean'], 
                    'pearson_ci': res['all']['pearson_ci'],
                    'pearson_std': res['all']['pearson_std'], 
                    'ari': res['all']['ari'],
                    'ari_ci': res['all']['ari_ci'],
                    'ami': res['all']['ami'],
                    'ami_ci': res['all']['ami_ci'],
                    'homo': res['all']['homo'],
                    'homo_ci': res['all']['homo_ci'],
                    'nmi': res['all']['nmi'],
                    'nmi_ci': res['all']['nmi_ci'],
                })
            checkpoint_perfs.append(res_all_masked_ratios)

        with open(os.path.join(current_save_dir, 'dataset_results.json'), 'w') as f:
            json.dump(checkpoint_perfs, f, sort_keys=True, indent=4)
