import os
import json
import numpy as np
import pandas as pd
import anndata as ad
from typing import List

import torch

from stpath.model.model import STFM
from stpath.data.dataset import rescale_coords
from stpath.model.nn_utils.config import ModelConfig
from stpath.tokenization import GeneExpTokenizer, ImageTokenizer, IDTokenizer, TokenizerTools, AnnotationTokenizer


class STPathInference:
    def __init__(self, gene_voc_path, model_weight_path, device=0):
        self.device = device

        # Setup the tokenizer and model
        self.tokenizer, self.model = self.setup(gene_voc_path, model_weight_path, device)
    
    def setup(self, gene_voc_path, model_weight_path, device):
        # loading the tokenizer
        tokenizer = TokenizerTools(
            ge_tokenizer=GeneExpTokenizer(gene_voc_path),
            image_tokenizer=ImageTokenizer(feature_dim=1536),
            tech_tokenizer=IDTokenizer(id_type="tech"), 
            specie_tokenizer=IDTokenizer(id_type="specie"), 
            organ_tokenizer=IDTokenizer(id_type="organ"),
            cancer_anno_tokenizer=AnnotationTokenizer(id_type="disease"),
            domain_anno_tokenizer=AnnotationTokenizer(id_type="domain"),
        )

        n_genes = tokenizer.ge_tokenizer.n_tokens
        n_tech = tokenizer.tech_tokenizer.n_tokens
        n_species = tokenizer.specie_tokenizer.n_tokens
        n_organs = tokenizer.organ_tokenizer.n_tokens
        n_cancer_annos = tokenizer.cancer_anno_tokenizer.n_tokens
        n_domain_annos = tokenizer.domain_anno_tokenizer.n_tokens
        print(f"n_genes: {n_genes}, n_tech: {n_tech}, n_species: {n_species}, n_organs: {n_organs}, n_cancer_annos: {n_cancer_annos}, n_domain_annos: {n_domain_annos}")

        # loading the model
        config = ModelConfig.get_default_config()
        config.feature_dim = 1536
        config.activation = "gelu"
        config.n_genes = n_genes
        config.n_tech = n_tech
        config.n_species = n_species
        config.n_organs = n_organs
        config.backbone = "spatial_transformer"
    
        model = STFM(config).to(device)
        model.load_state_dict(torch.load(model_weight_path))
        print(f"Model loaded from {model_weight_path}")
        model.eval()

        return tokenizer, model

    @torch.no_grad()
    def inference(
        self, 
        coords: np.ndarray, 
        img_features: np.ndarray, 
        context_ids: np.ndarray=None,
        context_gene_exps: np.ndarray=None,
        context_gene_names: List=None,
        organ_type: str=None, 
        tech_type: str=None, 
        save_gene_names: List=None,
    ):
        coords = torch.from_numpy(coords).to(self.device)
        coords = self._normalize_coords(coords)

        img_features = torch.from_numpy(img_features).to(self.device)
        masked_ge_tokens = self._generate_masked_ge_tokens(img_features.shape[0])
        masked_ge_tokens = masked_ge_tokens.to(self.device)

        if context_gene_exps is not None:
            assert context_ids is not None, "context_ids must be provided if context_gene_exps is provided."
            assert context_gene_names is not None, "context_gene_names must be provided if context_gene_exps is provided."
            print(f"Replacing masked gene tokens with context gene expressions for {len(context_ids)} spots.")

            context_gene_exps = torch.from_numpy(context_gene_exps).to(self.device)
            context_ids = torch.from_numpy(context_ids).to(self.device)
            
            context_gene_exps = self._log1p(context_gene_exps)  # log1p transformation
            context_gene_ids, valid_ids = self.tokenizer.ge_tokenizer.symbol2id(context_gene_names, return_valid_positions=True)
            context_gene_exps = context_gene_exps[:, valid_ids]
            context_gene_ids = torch.tensor(context_gene_ids, dtype=torch.long, device=self.device)  # remove the genes that are not in the tokenizer
            
            assert context_gene_exps.shape[1] == len(context_gene_ids), "Mismatch between context_gene_exps and context_gene_ids."
            context_gene_exps = self.tokenizer.ge_tokenizer.convert_gene_exp_to_one_hot_tensor(
                self.tokenizer.ge_tokenizer.n_tokens, context_gene_exps, context_gene_ids
            )
            
            # replace the masked ge tokens with context gene exp according to context_ids
            masked_ge_tokens[context_ids, :] = context_gene_exps

        if organ_type is None:
            organ = self.tokenizer.organ_tokenizer.encode("Others", align_first=True)
        else:
            organ = self.tokenizer.organ_tokenizer.encode(organ_type, align_first=True)
        organ_ids = torch.full((img_features.shape[0],), organ, dtype=torch.long, device=self.device)

        if tech_type is None:
            tech_ids = self._generate_pad_tech_tokens(img_features.shape[0])
            tech_ids = tech_ids.to(self.device)
        else:
            tech = self.tokenizer.tech_tokenizer.encode(tech_type, align_first=True)
            tech_ids = torch.full((img_features.shape[0],), tech, dtype=torch.long, device=self.device)

        print("Starting inference...")
        pred = self.model.prediction_head(
            img_tokens=img_features,
            coords=coords,
            ge_tokens=masked_ge_tokens,
            batch_idx=torch.zeros(img_features.shape[0], dtype=torch.long).to(self.device),
            tech_tokens=tech_ids,
            organ_tokens=organ_ids,
        )
        pred = pred.cpu().numpy()

        print("Return results...")
        if save_gene_names is not None:
            gene_ids = self.tokenizer.ge_tokenizer.symbol2id(save_gene_names)
            pred = pred[:, gene_ids]
            gene_names = save_gene_names
        else:
            gene_names = self.tokenizer.ge_tokenizer.get_available_genes()  # all 38984 genes

        adata = ad.AnnData(X=pred)
        adata.obsm['coordinates'] = coords.cpu().numpy()
        adata.var_names = pd.Index(gene_names)
        return adata

    def _generate_masked_ge_tokens(self, n_spots):
        mask_token = self.tokenizer.ge_tokenizer.mask_token.float()
        return mask_token.repeat(n_spots, 1)

    def _generate_pad_tech_tokens(self, n_spots):
        return torch.tensor([self.tokenizer.tech_tokenizer.pad_token_id] * n_spots, dtype=torch.long)

    def _normalize_coords(self, coords):
        # coords: [-1, 2]
        coords[:, 0] = coords[:, 0] - coords[:, 0].min()
        coords[:, 1] = coords[:, 1] - coords[:, 1].min()
        return rescale_coords(coords)

    def _log1p(self, x):
        """Log1p function that handles both numpy arrays and torch tensors."""
        if isinstance(x, np.ndarray):
            return np.log1p(x)
        elif isinstance(x, torch.Tensor):
            return torch.log1p(x)
        else:
            raise TypeError(f"Unsupported type: {type(x)}. Expected numpy array or torch tensor.")
