import os
import json
import wandb
import argparse
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm

import torch

from stpath.data.sampling_utils import PatchSampler
from stpath.data.distribution_utils import get_distribution
from stpath.data.normalize_utils import get_normalize_method
from stpath.tokenization import GeneExpTokenizer, ImageTokenizer, IDTokenizer, TokenizerTools, AnnotationTokenizer

from stpath.utils import set_random_seed, get_current_time, WarmupScheduler
from stpath.data.dataset import DatasetPath, STDataset, padding_batcher
from stpath.model.model import STFM
from stpath.app.pretrain.test import test

import warnings
warnings.filterwarnings("ignore")


def dataset_preparation(args, niche_dict, pretrain_sample_ids, val_sample_ids, tokenizer):
    normalize_method = get_normalize_method(args.normalize_method)

    print("Dataset Loading")
    pretrain_sample_id_paths = [
        DatasetPath(
            name=sample_id,
            source=niche_dict[sample_id]["source"],
            embed_path=os.path.join(args.source_dataroot, niche_dict[sample_id]["source"], f"embeddings/{sample_id}", args.feature_encoder, f"fp32/{sample_id}.h5"),
            h5ad_path=os.path.join(args.source_dataroot, niche_dict[sample_id]["source"], f"st/{sample_id}.h5ad"),
        ) for sample_id in pretrain_sample_ids
    ]

    train_dataset = STDataset(
        dataset_list=pretrain_sample_id_paths,
        meta_data_dict=niche_dict,
        patch_sampler=PatchSampler(args.patch_distribution, patch_sample_method=args.patch_sample_method),
        normalize_method=normalize_method,
        load_first=args.load_first,  # True means load all the pretraining samples in memory first, which might takes a while
        is_pretrain=True,
        tokenizer=tokenizer,
        masked_ratio_sampler=get_distribution(args.mask_distribution),
        n_hvg=args.n_hvg,
        use_hvg=args.use_hvg,
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=padding_batcher(), num_workers=args.num_workers)

    sample_id_paths = [
        DatasetPath(
            name=sample_id,
            source=niche_dict[sample_id]["source"],
            embed_path=os.path.join(args.source_dataroot, niche_dict[sample_id]["source"], f"embeddings/{sample_id}", args.feature_encoder, f"fp32/{sample_id}.h5"),
            h5ad_path=os.path.join(args.source_dataroot, niche_dict[sample_id]["source"], f"st/{sample_id}.h5ad"),
        ) for sample_id in val_sample_ids
    ]
    val_dataset = STDataset(
        dataset_list=sample_id_paths,
        meta_data_dict=niche_dict,
        patch_sampler=PatchSampler("constant_1.0"),
        normalize_method=normalize_method,
        load_first=True,
        is_pretrain=False,
        tokenizer=tokenizer,
        n_hvg=args.n_hvg,
        use_hvg=args.use_hvg,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, collate_fn=padding_batcher(), num_workers=args.num_workers
    )
    
    return train_loader, val_loader


def main(args, niche_dict, pretrain_sample_ids, val_sample_ids, tokenizer, val_save_dir, checkpoint_save_dir):
    train_loader, val_loader = dataset_preparation(args, niche_dict, pretrain_sample_ids, val_sample_ids, tokenizer)

    device = args.device
    model = STFM(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.warmup_steps > 0:
        scheduler = WarmupScheduler(optimizer, warmup_steps=args.warmup_steps, base_lr=args.lr)
    else:
        scheduler = None

    print("Training")
    best_pearson, best_val_dict = -1, None
    early_stop_step = 0
    epoch_iter = tqdm(range(1, args.epochs + 1), ncols=100)
    for epoch in epoch_iter:
        avg_loss = 0
        model.train()

        for step, batch in enumerate(train_loader):
            # try: 
            batch = [x.to(device) for x in batch]

            img_features, coords, masked_gene_exp, \
                obs_gene_ids, tech_ids, organ_ids, \
                    _, _, batch_idx, masked_tokens = batch

            loss = model(
                img_tokens=img_features,
                coords=coords,
                ge_tokens=masked_gene_exp,
                batch_idx=batch_idx,
                obs_gene_ids=obs_gene_ids,
                tech_tokens=tech_ids,
                organ_tokens=organ_ids,
                ge_masked_tokens=masked_tokens,
            )

            optimizer.zero_grad()
            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            if args.use_wandb:
                wandb.log({f"Train/loss": loss.cpu().item()})

            avg_loss += loss.cpu().item()
            
            # except Exception as e:
            #     print(f"Error: {e}")

        avg_loss /= len(train_loader)
        epoch_iter.set_description(f"epoch: {epoch}, avg_loss: {avg_loss:.3f}")

        if args.save_step > 0 and (epoch == 10 or epoch % args.save_step == 0):
            torch.save(model.state_dict(), os.path.join(checkpoint_save_dir, f"{epoch}.pth"))

        if epoch % args.eval_step == 0 or epoch == args.epochs:
            val_perf_dict, pred_dump = test(args, model, val_loader, return_all=True)

            if val_perf_dict["all"]['pearson_mean'] > best_pearson:
                best_pearson = val_perf_dict["all"]['pearson_mean']
                best_val_dict = val_perf_dict

                for patch_name, dataset_res in val_perf_dict.items():
                    with open(os.path.join(val_save_dir, f'{patch_name}_results.json'), 'w') as f:
                        json.dump(dataset_res, f, sort_keys=True, indent=4)

                torch.save(model.state_dict(), os.path.join(checkpoint_save_dir, f"best.pth"))

                # save_pkl(os.path.join(val_save_dir, 'inference_dump.pkl'), pred_dump)
                early_stop_step = 0
            else:
                early_stop_step += 1
                if early_stop_step >= 20:
                    print("Early stopping")
                    break

            if args.use_wandb:
                wandb.log({
                        f"Val/all/gene/pearson_mean": val_perf_dict["all"]['pearson_mean'], 
                        f"Val/all/gene/pearson_std": val_perf_dict["all"]['pearson_std'],
                    })
                if len(args.n_eval_hvg) > 1:
                    pearson_dict = {}
                    for n_hvg in args.n_eval_hvg[:-1]:
                        pearson_dict[f"Val/all/gene/pearson_mean_{n_hvg}"] = val_perf_dict["all"][f'pearson_mean_{n_hvg}']
                        pearson_dict[f"Val/all/gene/pearson_std_{n_hvg}"] = val_perf_dict["all"][f'pearson_std_{n_hvg}']
                    wandb.log(pearson_dict)

                for patch_name, dataset_res in val_perf_dict.items():
                    if patch_name == "all":
                        continue
                    wandb.log({
                        f"Val/{patch_name}/pearson_mean": dataset_res['pearson_mean'],
                        f"Val/{patch_name}/pearson_std": dataset_res['pearson_std'],
                        f"Val/{patch_name}/l2_error_q1": dataset_res['l2_error_q1'],
                        f"Val/{patch_name}/l2_error_q2": dataset_res['l2_error_q2'],
                        f"Val/{patch_name}/l2_error_q3": dataset_res['l2_error_q3'],
                        f"Val/{patch_name}/r2_score_q1": dataset_res['r2_score_q1'],
                        f"Val/{patch_name}/r2_score_q2": dataset_res['r2_score_q2'],
                        f"Val/{patch_name}/r2_score_q3": dataset_res['r2_score_q3'],
                    })

    return best_val_dict["all"]


def run(args):
    np.random.seed(args.seed)

    meta_data_dict = {}
    all_pretrain_sample_ids = []
    niche_dict = {}

    for dataset in args.datasets:
        if dataset == "hest":
            # obtain sample ids used for pretraining
            meta_data = pd.read_csv(os.path.join(args.source_dataroot, 'hest/HEST_v1_1_0.csv'))

            # keep the samples with the species in args.species
            species_map = {
                "human": "Homo sapiens",
                "mouse": "Mus musculus"
            }
            species = [species_map[specie] for specie in args.species]
            meta_data = meta_data[meta_data["species"].isin(species)]
            
            all_samples_ids = meta_data["id"].tolist()
            
            # filter out the samples from hest-bench
            benchmark_sample_ids = []
            for benchmark_dataset in ["LUNG", "HCC", "COAD", "SKCM", "PAAD", "READ", "LYMPH_IDC", "PRAD", "IDC", "CCRCC"]:
                benchmark_sample_ids.extend(
                    [f.split(".")[0] for f in os.listdir(os.path.join(args.source_dataroot, "hest/hest-bench", f'{benchmark_dataset}/adata'))]
                )
            pretrain_sample_ids = list(set(all_samples_ids) - set(benchmark_sample_ids))
            if os.path.exists(args.val_sample_path):
                with open(args.val_sample_path, 'r') as f:
                    val_sample_ids = [x.strip() for x in f.readlines()]
                print(f"Loading val samples from {args.val_sample_path}")
            else:
                # randomly pick 5% of the samples for validation
                val_sample_ids = np.random.choice(pretrain_sample_ids, int(len(pretrain_sample_ids) * 0.05), replace=False)
                with open(args.val_sample_path, 'w') as f:
                    for sample_id in val_sample_ids:
                        f.write(f"{sample_id}\n")

            pretrain_sample_ids = list(set(pretrain_sample_ids) - set(val_sample_ids))
            niche_dict.update({
                row["id"]: {"tech": row["st_technology"], "organ": row["organ"], "specie": row["species"], "source": "hest"}
                for _, row in meta_data.iterrows()
            })

        elif dataset == "stimage_1k4m":
            meta_data = pd.read_csv(os.path.join(args.source_dataroot, 'stimage_1k4m/meta/meta_all_gene.csv'))
            meta_data = meta_data[meta_data["species"].isin(args.species)]

            all_samples_ids = meta_data["slide"].tolist()
            pretrain_sample_ids = all_samples_ids
            niche_dict.update({
                row["slide"]: {"tech": row["tech"], "organ": row["tissue"], "specie": row["species"], "source": "stimage_1k4m"}
                for _, row in meta_data.iterrows()
            })

        all_pretrain_sample_ids.extend(pretrain_sample_ids)
        meta_data_dict[dataset] = meta_data

    # remove the samples that are overlapped with the HEST
    with open(os.path.join("/home/ti.huang/STPath/utils_data/stimage_overlap_samples.json"), 'r') as f:
        toremove_sample_ids = json.load(f)["id"]
    # remove the samples in stimage that has annotations
    with open(os.path.join("/home/ti.huang/STPath/utils_data/stimage_cancer_annos_eval.json"), 'r') as f:
        toremove_sample_ids.extend(json.load(f)["all"])
    with open(os.path.join("/home/ti.huang/STPath/utils_data/stimage_domain_annos_eval.json"), 'r') as f:
        toremove_sample_ids.extend(json.load(f)["all"])
    # remove the samples in hest that are from the same sources as the samples with annotations in stimage
    with open(os.path.join("/home/ti.huang/STPath/utils_data/hest_anno_samples.json"), 'r') as f:
        toremove_sample_ids.extend(json.load(f)["all"])

    print(f"Remove {len(toremove_sample_ids)} samples from {len(all_pretrain_sample_ids)} samples")
    all_pretrain_sample_ids = list(set(all_pretrain_sample_ids) - set(toremove_sample_ids))
    print(f"Total {len(all_pretrain_sample_ids)} samples")

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
    print(f"n_genes: {args.n_genes}, n_tech: {args.n_tech}, n_species: {args.n_species}, n_organs: {args.n_organs}")

    val_save_dir = os.path.join(args.save_dir, 'result')
    os.makedirs(val_save_dir, exist_ok=True)
    checkpoint_save_dir = args.save_dir

    val_sample_ids = val_sample_ids[:2]
    all_pretrain_sample_ids = all_pretrain_sample_ids[:10]
    # all_pretrain_sample_ids = ["MISC128", "GSE243981_GSM7845914", "TENX68", "GSE223559_GSM6963122"]

    results = main(
        args, niche_dict, 
        all_pretrain_sample_ids, 
        val_sample_ids, 
        tokenizer, 
        val_save_dir, 
        checkpoint_save_dir
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--use_wandb', default=False)
    parser.add_argument('--source_dataroot', default="/home/ti.huang/project/single_cell_dataset")
    parser.add_argument('--datasets', nargs='+', default=["hest", "stimage_1k4m"], help='hest | stimage_1k4m')
    parser.add_argument('--val_sample_path', type=str, default='/home/ti.huang/STPath/utils_data/val_sample.txt')
    parser.add_argument('--feature_encoder', type=str, default='gigapath', help="uni_v1_official | resnet50_trunc | ciga | gigapath")
    parser.add_argument('--normalize_method', type=str, default="log1pv2", help="log1pv2 | mehrtash")
    parser.add_argument('--save_dir', type=str, default="/home/ti.huang/project/stfm/stpath/pretrain/test")
    parser.add_argument('--exp_code', type=str, default="stpath")

    # dataset hyperparameters
    parser.add_argument('--species', nargs='+', default=["human"], help="human | mouse | all")
    parser.add_argument('--status', type=str, default="all", help="Cancer | Healthy | all")
    parser.add_argument('--gene_voc_path', type=str, default="/home/ti.huang/STPath/utils_data/symbol2ensembl.json")
    parser.add_argument('--load_first', default=False, help="load all the pretraining samples in memory first or not")

    # pretraining hyperparameters
    parser.add_argument('--mask_distribution', default='beta_10_1')
    parser.add_argument('--n_hvg', type=int, default=200)
    parser.add_argument('--n_eval_hvg', nargs='+', type=int, default=[50, 100, 200])
    parser.add_argument('--use_hvg', type=bool, default=True, help="predict HVG during pretraining")

    # training hyperparameters
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--warmup_steps', type=int, default=-1)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--clip_norm', type=float, default=1.)
    parser.add_argument('--save_step', type=int, default=50)
    parser.add_argument('--eval_step', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=5, help='Number of workers for dataloader')
    parser.add_argument('--loss_func', type=str, default='mse', help="mse | mae | pearson")
    parser.add_argument('--patch_distribution', type=str, default='uniform_64_1024', help="uniform | batch_64 | batch_128 | uniform_64_1024")
    parser.add_argument('--patch_sample_method', type=str, default='nearest', help="nearest | random")

    # model hyperparameters
    parser.add_argument('--backbone', type=str, default="spatial_transformer", help="spatial_transformer | seq_transformer | transformer | mlp | transformer_v2")
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--pairwise_hidden_dim', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--attn_dropout', type=float, default=0.)
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
    args.save_dir = os.path.join(args.save_dir, exp_code)
    os.makedirs(args.save_dir, exist_ok=True)
    
    if args.use_wandb:
        wandb.init(project="spatial_transcriptomics", name=exp_code)
        wandb.config.update(args)

    print(f"Save dir: {args.save_dir}")
    print(args)

    with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)
    run(args)

    if args.use_wandb:
        wandb.finish()
