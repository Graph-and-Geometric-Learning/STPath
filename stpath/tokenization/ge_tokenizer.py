import json
import numpy as np
from time import time
from typing import Any
import scanpy as sc

import torch
import torch.nn.functional as F
from torch_geometric.utils import coalesce

from stpath.utils import create_row_index_tensor
from stpath.tokenization.tokenizer_base import TokenizerBase


class GeneExpTokenizer(TokenizerBase):
    def __init__(
        self,
        symbol2gene_path,
    ):

        self.symbol2gene = json.load(open(symbol2gene_path, "r"))
        gene2id = list(set(self.symbol2gene.values()))
        gene2id.sort()  # make sure the order is fixed
        self.gene2id = {gene_id: i for i, gene_id in enumerate(gene2id)}

        # add 2 for pad and mask token
        self.gene2id = {gene: token_id + 2 for gene, token_id in self.gene2id.items()}

    def get_available_symbols(self):
        return [g for g in list(self.symbol2gene.keys()) if self.symbol2gene[g] in self.gene2id]

    def get_available_genes(self):
        # rank the genes by their id
        all_genes = sorted(self.gene2id.keys(), key=lambda x: self.gene2id[x])
        # remove pad and mask token
        return all_genes[2:]

    def convert_gene_exp_to_one_hot_tensor(self, n_genes, gene_exp, gene_ids):
        # gene_exp: [N_cells, N_genes], gene_ids: [N_genes]
        gene_ids = gene_ids.clamp(min=0, max=n_genes)[None, ...].expand(gene_exp.shape[0], -1)
        gene_exp_onehot = torch.zeros(gene_exp.size(0), n_genes, device=gene_exp.device)
        gene_exp_onehot.scatter_(dim=1, index=gene_ids, src=gene_exp)
        return gene_exp_onehot

    def symbol2id(self, symbol_list, return_valid_positions=False):
        # if return_valid_positions is True, return the positions of the valid symbols in the input list
        res = [self.gene2id[self.symbol2gene[symbol]] for symbol in symbol_list if symbol in self.symbol2gene]
        if len(res) != len(symbol_list):
            print(f"Warning: {len(symbol_list) - len(res)} symbols are not in the tokenizer.")
        
        if return_valid_positions:
            valid_positions = [i for i, symbol in enumerate(symbol_list) if symbol in self.symbol2gene]
            return res, valid_positions
        else:
            return res

    def shift_token_id(self, n_shift):
        self.gene2id = {gene:token_id+n_shift for gene, token_id in self.gene2id.items()}

    def subset_gene_ids(self, symbol_list):
        gene_ids = [self.symbol2gene[s] for s in symbol_list if s in self.symbol2gene]
        # only keep the valid gene ids in self.gene2ids
        self.gene2id = {gene_id:i for i, gene_id in enumerate(gene_ids)}

    def get_hvg(self, adata, n_top_genes=100):
        if "raw_hvg_names" in adata.uns:
            hvg_names = adata.uns["raw_hvg_names"][:n_top_genes]
        else:
            all_gene_names = set(self.get_available_symbols())
            available_genes = [g for g in adata.var_names.tolist() if g in all_gene_names]
            adata = adata.copy()[:, available_genes]

            sc.pp.filter_genes(adata, min_cells=np.ceil(0.1 * len(adata.obs)))
            sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
            hvg_names = adata.var_names[adata.var['highly_variable']][:n_top_genes].tolist()

        hvg_names = [g for g in hvg_names if g in self.symbol2gene and self.symbol2gene[g] is not None]
        hvg_gene_ids = np.array([self.gene2id[self.symbol2gene[g]] for g in hvg_names if self.symbol2gene[g] in self.gene2id])
        return hvg_names, torch.from_numpy(hvg_gene_ids).long()

    def encode(self, adata, return_sparse=False):
        gene_list = [g for g in adata.var_names.tolist() if g in self.symbol2gene and self.symbol2gene[g] is not None and self.symbol2gene[g] in self.gene2id]
        adata = adata[:, gene_list]

        if not isinstance(adata.X, np.ndarray):
            row = create_row_index_tensor(adata.X)
            col = adata.X.indices
            act_gene_exp = adata.X.data  # nonzero values
        else:
            row, col = np.nonzero(adata.X)
            act_gene_exp = adata.X[row, col]

        obs_gene_ids = np.array([self.gene2id[self.symbol2gene[g]] for g in adata.var_names.tolist()])  # TODO: there might be some genes sharing the same id so some expression values will be missing
        col = obs_gene_ids[col]
        obs_gene_ids = torch.from_numpy(obs_gene_ids).long()
        act_gene_exp = torch.from_numpy(act_gene_exp).float()

        indices = torch.stack([torch.from_numpy(row), torch.from_numpy(col)], dim=0).long()

        if return_sparse:
            # remove duplicate indices
            indices, act_gene_exp = coalesce(
                edge_index=indices,
                edge_attr=act_gene_exp,
                reduce="max",  # pick the maximum value if there are multiple values for the same edge
            )
            gene_exp = torch.sparse_coo_tensor(indices, act_gene_exp, size=(adata.shape[0], self.n_tokens))
        else:
            gene_exp = adata.to_df().values
            gene_exp = torch.from_numpy(gene_exp).float()
            gene_exp = self.convert_gene_exp_to_one_hot_tensor(self.n_tokens, gene_exp, obs_gene_ids)

        # gene_exp: [n_cells, n_genes], obs_gene_ids: [-1]
        return gene_exp, obs_gene_ids

    @property
    def n_tokens(self):
        return max(self.gene2id.values()) + 1

    @property
    def mask_token(self) -> str:
        return F.one_hot(torch.tensor(1), num_classes=self.n_tokens).float()

    @property
    def mask_token_id(self) -> int:
        return 1

    @property
    def pad_token(self) -> str:
        return F.one_hot(torch.tensor(0), num_classes=self.n_tokens).float()

    @property
    def pad_token_id(self) -> int:
        return 0
