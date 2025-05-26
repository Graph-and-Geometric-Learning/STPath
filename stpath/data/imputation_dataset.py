import os
import json
import numpy as np
from typing import List

import torch

from stpath.tokenization import TokenizerTools
from stpath.data.sampling_utils import PatchSampler
from stpath.data.dataset import DatasetPath, STDataset


class SpotImputationDataset(STDataset):
    def __init__(
        self, 
        dataset_list: List[DatasetPath], 
        meta_data_dict: dict, 
        normalize_method, 
        tokenizer: TokenizerTools, 
        patch_sampler, 
    ):
        super().__init__(dataset_list, meta_data_dict, normalize_method, tokenizer, patch_sampler, load_first=True)

        for i, dataset in enumerate(dataset_list):
            # generate the split path
            split_path = dataset.split_path
            if not os.path.exists(split_path):
                coords = self.st_datasets[i].coords
                rightest_coord = np.where(coords[:, 0] == coords[:, 0].max())[0][0]
                split_dict = {}
                for masked_ratio in [0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
                    masked_index = PatchSampler.sample_nearest_patch(coords, int(len(coords) * masked_ratio), rightest_coord)
                    split_dict[masked_ratio] = masked_index.tolist()
                with open(split_path, 'w') as f:
                    json.dump(split_dict, f, indent=4)

    def load_masked_index(self, masked_ratio=0.8):
        self.masked_index = []
        for i, dataset in enumerate(self.dataset_list):
            with open(dataset.split_path, 'r') as f:
                split_dict = json.load(f)
            self.masked_index.append(split_dict[str(masked_ratio)])

    def __getitem__(self, idx):
        features, coords, gene_exp, obs_gene_ids, _, \
            tech_ids, specie_ids, organ_ids, \
                cancer_annos, domain_annos = self.st_datasets[idx].chunk(self.patch_sampler(self.st_datasets[idx].coords))

        # gt_tokens = gene_exp.clone()[self.masked_index]
        masked_gene_exp = gene_exp.clone()
        masked_gene_exp[self.masked_index[idx]] = self.tokenizer.ge_tokenizer.mask_token

        return features, coords, masked_gene_exp, obs_gene_ids, tech_ids, specie_ids, organ_ids, cancer_annos, domain_annos, \
                    gene_exp, torch.tensor(self.masked_index[idx], dtype=torch.long)
