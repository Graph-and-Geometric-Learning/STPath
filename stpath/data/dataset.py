import os
import json
import pickle
import random
import numpy as np
from time import time
from tqdm import tqdm
from typing import List
from pathlib import Path
import scanpy as sc

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from stpath.hest_utils.st_dataset import load_adata
from stpath.hest_utils.file_utils import read_assets_from_h5
from stpath.tokenization import TokenizerTools


def rescale_coords(coords, new_max=100):
    """
    Rescale coordinates to a specified range while maintaining their shape.
    
    Parameters:
        coords (torch.Tensor): A tensor of shape (n, 2), where each row contains (x, y) coordinates.
        new_max (float): The maximum value for the new scaled coordinates.
    
    Returns:
        torch.Tensor: The rescaled coordinates.
    """
    # Find the minimum and maximum values of the coordinates
    min_coords = torch.min(coords, dim=0).values
    max_coords = torch.max(coords, dim=0).values

    # Calculate the range of the coordinates
    coord_range = max_coords - min_coords

    # Rescale the coordinates to the range [0, new_max]
    scaled_coords = (coords - min_coords) / coord_range * new_max

    return scaled_coords


class DatasetPath:
    name: str | None = None
    source: str | None = None
    embed_path: str | None = None
    h5ad_path: str | None = None

    def __init__(self, name, source, embed_path, h5ad_path, **kwargs):
        self.name = name
        self.source = source
        self.embed_path = embed_path
        self.h5ad_path = h5ad_path

        for k, v in kwargs.items():
            setattr(self, k, v)


class STData:
    features: torch.Tensor | None = None
    coords: torch.Tensor | None = None
    gene_exp: torch.Tensor | None = None
    obs_gene_ids: torch.Tensor | None = None  # observed gene ids in this sample
    hvg_ids: torch.Tensor | None = None
    tech_ids: torch.Tensor | None = None
    specie_ids: torch.Tensor | None = None
    organ_ids: torch.Tensor | None = None
    cancer_anno_ids: torch.Tensor | None = None
    domain_anno_ids: torch.Tensor | None = None
    
    def __init__(self, features, coords, gene_exp, obs_gene_ids, hvg_ids, tech_ids, specie_ids, organ_ids, cancer_anno_ids=None, domain_anno_ids=None, **kwargs):
        self.features = features
        self.coords = coords
        self.gene_exp = gene_exp
        self.obs_gene_ids = obs_gene_ids
        self.hvg_ids = hvg_ids
        self.tech_ids = tech_ids
        self.specie_ids = specie_ids
        self.organ_ids = organ_ids
        self.cancer_anno_ids = cancer_anno_ids
        self.domain_anno_ids = domain_anno_ids

        # decenter and rescale to align different resolution
        self.coords[:, 0] = self.coords[:, 0] - self.coords[:, 0].min()
        self.coords[:, 1] = self.coords[:, 1] - self.coords[:, 1].min()
        self.coords = rescale_coords(self.coords)

        for k, v in kwargs.items():
            setattr(self, k, v)

    def __len__(self):
        return len(self.features)

    def splice_gene_exp(self, index):
        if self.gene_exp.is_sparse:
            return self.gene_exp.to_dense().float()[index]
        else:
            return self.gene_exp[index]

    def chunk(self, index):
        length = len(index)

        return self.features[index], self.coords[index], \
            self.splice_gene_exp(index), \
            self.obs_gene_ids[None, ...].expand(length, -1), self.hvg_ids[None, ...].expand(length, -1), \
            self.tech_ids[index], self.specie_ids[index], self.organ_ids[index], \
            self.cancer_anno_ids[index] if self.cancer_anno_ids is not None else None, \
            self.domain_anno_ids[index] if self.domain_anno_ids is not None else None


class MaskToken:
    token_type: str | None = None
    position: torch.Tensor | None = None  # id of the special position
    groundtruth: torch.Tensor | None = None

    def __init__(self, token_type, position, groundtruth):
        self.token_type = token_type
        self.position = position
        self.groundtruth = groundtruth

    def to(self, device):
        self.position = self.position.to(device)
        self.groundtruth = self.groundtruth.to(device)
        return MaskToken(self.token_type, self.position, self.groundtruth)


class STDataset(Dataset):
    def __init__(
        self, 
        dataset_list: List[DatasetPath],
        meta_data_dict: dict, 
        normalize_method, 
        tokenizer: TokenizerTools, 
        patch_sampler, 
        load_first=True,
        is_pretrain=False,
        masked_ratio_sampler=None,
        n_hvg=100,
        use_hvg=False,
    ):
        super().__init__()

        self.dataset_list = dataset_list
        self.st_datasets = []
        self.patch_sampler = patch_sampler
        self.tokenizer = tokenizer
        self.meta_data_dict = meta_data_dict
        self.normalize_method = normalize_method
        self.is_pretrain = is_pretrain
        self.masked_ratio_sampler = masked_ratio_sampler
        self.n_hvg = n_hvg
        self.use_hvg = use_hvg
        self.load_first = load_first

        if load_first:
            # load all datasets in advance
            # but this will take a lot of memory if the number of datasets is large
            self.st_datasets = []
            for i, dataset in enumerate(tqdm(self.dataset_list, ncols=130, desc="Loading datasets")):
                try:
                    stdata = self.load_adata(dataset, tokenizer, normalize_method, meta_data_dict, n_hvg=n_hvg)
                except Exception as e:
                    print(f"{i}, {dataset.name}: {e}")
                    continue

                self.st_datasets.append(stdata)

    def load_adata(self, dataset, tokenizer, normalize_method, meta_data_dict, n_hvg=100):
        data_dict, _ = read_assets_from_h5(dataset.embed_path)
        coords = torch.from_numpy(data_dict["coords"]).float()
        embeddings = torch.from_numpy(data_dict["embeddings"]).float()

        if dataset.source == "hest":
            barcodes = data_dict["barcodes"].flatten().astype(str).tolist()
        elif dataset.source == "stimage_1k4m":
            barcodes = data_dict["barcodes"].astype('U').tolist()
            barcodes = [i[0] for i in barcodes]
        else:
            raise ValueError(f"Unknown dataset source: {dataset.source}")

        adata = load_adata(dataset.h5ad_path, barcodes=barcodes, normalize_method=normalize_method, return_df=False)
        gene_exp, obs_gene_ids = tokenizer.ge_tokenizer.encode(adata)

        hvg_gene_names, hvg_ids = tokenizer.ge_tokenizer.get_hvg(adata, n_top_genes=n_hvg)
        tech = tokenizer.tech_tokenizer.encode(meta_data_dict[dataset.name]["tech"], align_first=True)
        specie = tokenizer.specie_tokenizer.encode(meta_data_dict[dataset.name]["specie"], align_first=True)
        organ = tokenizer.organ_tokenizer.encode(meta_data_dict[dataset.name]["organ"], align_first=True)

        n_spots = gene_exp.shape[0]
        tech = torch.full((n_spots,), tech, dtype=torch.long)
        specie = torch.full((n_spots,), specie, dtype=torch.long)
        organ = torch.full((n_spots,), organ, dtype=torch.long)

        if "annotation" in adata.obs:
            cancer_anno_ids = tokenizer.cancer_anno_tokenizer.encode(
                adata.obs["annotation"].tolist(), align_first=True
            )
            domain_anno_ids = tokenizer.domain_anno_tokenizer.encode(
                adata.obs["annotation"].tolist(), align_first=True
            )
            cancer_anno_ids = torch.tensor(cancer_anno_ids, dtype=torch.long)
            domain_anno_ids = torch.tensor(domain_anno_ids, dtype=torch.long)
        else:
            cancer_anno_ids = torch.full((n_spots,), tokenizer.cancer_anno_tokenizer.pad_token)
            domain_anno_ids = torch.full((n_spots,), tokenizer.domain_anno_tokenizer.pad_token)

        return STData(
                features=embeddings,
                coords=coords,
                gene_exp=gene_exp,
                obs_gene_ids=obs_gene_ids,
                hvg_ids=hvg_ids,
                tech_ids=tech,
                specie_ids=specie,
                organ_ids=organ,
                cancer_anno_ids=cancer_anno_ids,
                domain_anno_ids=domain_anno_ids,
                sample_id=dataset.name,
            )

    def __len__(self):
        return len(self.dataset_list)

    def get_dataset_name(self, idx):
        return self.dataset_list[idx].name

    def generate_masked_ge_tokens(self, n_spots):
        mask_token = self.tokenizer.ge_tokenizer.mask_token.float()
        mask_token.repeat(n_spots, 1)
        return mask_token

    def generate_pad_tech_tokens(self, n_spots):
        return torch.tensor([self.tokenizer.tech_tokenizer.pad_token_id] * n_spots, dtype=torch.long)

    def generate_masked_anno_tokens(self, n_spots):
        return torch.full((n_spots,), self.tokenizer.cancer_anno_tokenizer.mask_token)

    def replace_gene_masked_tokens(self, token_ids, tokens, mask_ratio, tokenizer):
        spot_ids = torch.arange(tokens.size(0))
        
        mask_num = int(max(1, mask_ratio * len(spot_ids)))
        mask_ids = spot_ids[torch.randperm(len(spot_ids))[:mask_num]]

        gt_tokens = tokens.clone()[mask_ids]
        masked_tokens = tokens.clone()

        # directly replace with mask token
        # masked_tokens[mask_ids] = tokenizer.mask_token

        # 90% of the time, replace with mask token
        all_mask_num = int(0.90 * len(mask_ids))
        all_mask_ids, part_mask_ids = mask_ids[:all_mask_num], mask_ids[all_mask_num:]
        masked_tokens[all_mask_ids] = tokenizer.mask_token

        # for the rest 10% of the time, mask 80% of the observed genes
        drop_cols = torch.randperm(token_ids.shape[1])[:int(token_ids.shape[1] * 0.5)]  # mask 80% of the observed genes
        masked_observed_genes = token_ids[part_mask_ids][:, drop_cols]
        tobe_maskes_rows = masked_tokens[part_mask_ids]
        tobe_maskes_rows[torch.arange(masked_observed_genes.shape[0]).unsqueeze(-1), masked_observed_genes] = 0
        masked_tokens[part_mask_ids] = tobe_maskes_rows
        return masked_tokens, gt_tokens, mask_ids

    def __getitem__(self, idx):
        if self.load_first:
            st_dataset = self.st_datasets[idx]
        else:
            dataset = self.dataset_list[idx]
            st_dataset = self.load_adata(dataset, self.tokenizer, self.normalize_method, self.meta_data_dict)

        sampled_idx = self.patch_sampler(st_dataset.coords)
        features, coords, gene_exp, obs_gene_ids, obs_hvg_gene_ids, \
            tech_ids, specie_ids, organ_ids, cancer_annos, domain_annos = st_dataset.chunk(sampled_idx)

        if not self.is_pretrain:
            return features, coords, gene_exp, \
                    obs_hvg_gene_ids if self.use_hvg else obs_gene_ids, \
                        tech_ids, organ_ids, cancer_annos, domain_annos
        else:
            if random.random() > 0.2:
                # 80% of the time during training, use HVGs
                obs_gene_ids = obs_hvg_gene_ids if self.use_hvg else obs_gene_ids

            # randomly drops 80% tech_ids to mimic the situation where technology information is missing
            tech_ids = (tech_ids * (torch.rand_like(tech_ids.float()) > 0.8)).long()
            
            masked_ratio = self.masked_ratio_sampler()
            gene_exp, gt_gene_exp, ge_mask_ids = self.replace_gene_masked_tokens(obs_gene_ids, gene_exp, masked_ratio, self.tokenizer.ge_tokenizer)
            pretrain_data = (gt_gene_exp, ge_mask_ids)

            return features, coords, gene_exp, obs_gene_ids, tech_ids, organ_ids, cancer_annos, domain_annos, pretrain_data


def padding_batcher():
    def batcher_dev(batch):
        features = [d[0] for d in batch]
        coords = [d[1] for d in batch]
        gene_exp = [d[2] for d in batch]
        obs_gene_ids = [d[3] for d in batch]
        tech_ids = [d[4] for d in batch]
        organ_ids = [d[5] for d in batch]
        cancer_annos = [d[6] for d in batch]
        domain_annos = [d[7] for d in batch]

        batch_idx_list = []
        for i, tensor in enumerate(features):
            batch_idx_list.append(torch.full((tensor.shape[0],), i, dtype=torch.long))
        batch_idx = torch.cat(batch_idx_list)

        features = torch.cat(features, dim=0)
        coords = torch.cat(coords, dim=0)
        tech_ids = torch.cat(tech_ids, dim=0)
        organ_ids = torch.cat(organ_ids, dim=0)
        gene_exp = torch.cat(gene_exp, dim=0)
        cancer_annos = torch.cat(cancer_annos, dim=0)
        domain_annos = torch.cat(domain_annos, dim=0)

        max_n = max([x.size(1) for x in obs_gene_ids])
        obs_gene_ids = torch.cat([F.pad(x, (0, max_n - x.size(1))) for x in obs_gene_ids])
        
        # pretraining setup
        if len(batch[0]) == 9:
            # for gene expression prediction
            gt_tokens = [d[8][0] for d in batch]
            mask_ids = [d[8][1] for d in batch]
            gt_tokens = torch.cat(gt_tokens, dim=0)

            n_feats = np.cumsum([d[0].size(0) for d in batch])
            for i in range(1, len(n_feats)):
                if len(mask_ids[i]) > 0:
                    mask_ids[i] += n_feats[i - 1]
            mask_ids = torch.cat(mask_ids, dim=0)
            masked_tokens = MaskToken('gene', mask_ids, gt_tokens)

            return features, coords, gene_exp, obs_gene_ids, tech_ids, organ_ids, cancer_annos, domain_annos, batch_idx, masked_tokens

        return features, coords, gene_exp, obs_gene_ids, tech_ids, organ_ids, cancer_annos, domain_annos, batch_idx
    return batcher_dev
