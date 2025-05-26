import h5py
import numpy as np
import scanpy as sc
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset

from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, RobustScaler


def normalize_adata(adata: sc.AnnData, smooth=False) -> sc.AnnData:
    """
    Normalize each spot by total gene counts + Logarithmize each spot
    """
    filtered_adata = adata.copy()
    filtered_adata.X = filtered_adata.X.astype(np.float64)

    if smooth:
        adata_df = adata.to_df()
        for index, df_row in adata.obs.iterrows():
            row = int(df_row['array_row'])
            col = int(df_row['array_col'])
            neighbors_index = adata.obs[((adata.obs['array_row'] >= row - 1) & (adata.obs['array_row'] <= row + 1)) & \
                ((adata.obs['array_col'] >= col - 1) & (adata.obs['array_col'] <= col + 1))].index
            neighbors = adata_df.loc[neighbors_index]
            nb_neighbors = len(neighbors)
            
            avg = neighbors.sum() / nb_neighbors
            filtered_adata[index] = avg
    
    # Logarithm of the expression
    sc.pp.log1p(filtered_adata)
    return filtered_adata


def load_adata(expr_path, genes = None, barcodes = None, normalize_method=normalize_adata, return_df=True):
    adata = sc.read_h5ad(expr_path)
    if barcodes is not None:
        adata = adata[barcodes]
    if normalize_method is not None:
        adata = normalize_method(adata)
    if genes is not None:
        adata = adata[:, genes]
    if return_df:
        return adata.to_df()
    return adata


def load_tiles(h5_path):
    with h5py.File(h5_path, 'r') as f:
        barcodes = f['barcode'][:].astype(str).flatten().tolist()
        coords = f['coords'][:]
        tiles = f['img'][:]
    return barcodes, coords, tiles


class H5TileDataset(Dataset):
    def __init__(self, h5_path, img_transform=None, chunk_size=1000):
        self.h5_path = h5_path
        self.img_transform = img_transform
        self.chunk_size = chunk_size
    
        with h5py.File(h5_path, 'r') as f:
            barcode = f['barcode'] if 'barcode' in f else f['barcodes']
                
            if chunk_size == -1:
                self.n_chunks = 1
                self.chunk_size = len(barcode)
            else:
                self.n_chunks = int(np.ceil(len(barcode) / chunk_size))
        
    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        start_idx = idx * self.chunk_size
        end_idx = (idx + 1) * self.chunk_size
        with h5py.File(self.h5_path, 'r') as f:
            barcodes = f['barcode'] if 'barcode' in f else f['barcodes']

            imgs = f['img'][start_idx:end_idx]
            barcodes = barcodes[start_idx:end_idx].flatten().tolist()
            coords = f['coords'][start_idx:end_idx]
            
        if self.img_transform:
            imgs = torch.stack([self.img_transform(Image.fromarray(img)) for img in imgs])
                    
        return {'imgs': imgs, 'barcodes': barcodes, 'coords': coords}
