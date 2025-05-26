import os
import json
import wandb
import argparse
import numpy as np
import scanpy as sc
import pandas as pd
from time import time
from tqdm import tqdm
import scipy.sparse as sp

import torch
from stpath.hest_utils.encoder import load_encoder
from stpath.hest_utils.st_dataset import H5TileDataset
from stpath.hest_utils.file_utils import save_hdf5


class LazyEncoder:
    def __init__(self, name, weights_root, private_weights_root=None, transforms=None, model=None):
        self.name = name
        self.model = model
        self.transforms = transforms
        self.weights_root = weights_root
        self.private_weights_root = private_weights_root
        
    def get_model(self, device):
        if self.model is not None:
            return self.model, self.transforms
        else:
            encoder, img_transforms, _ = load_encoder(self.name, device, self.weights_root, self.private_weights_root)
            return encoder, img_transforms


def embed_tiles(sample_id, 
                dataloader,
                model,
                embedding_save_path,
                device,
                precision=torch.float32,
                use_coords=None):
    
    def post_collate_fn(batch):
        """
        Post collate function to clean up batch
        """
        if batch["imgs"].dim() == 5:
            assert batch["imgs"].size(0) == 1
            batch["imgs"] = batch["imgs"].squeeze(0)
        if batch["coords"].dim() == 3:
            assert batch["coords"].size(0) == 1
            batch["coords"] = batch["coords"].squeeze(0)
        return batch

    """
    Extract embeddings from tiles using encoder and save to h5 file
    """
    model.eval()
    for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Embedding Tiles {sample_id}', ncols=100):
        batch = post_collate_fn(batch)
        imgs = batch['imgs'].to(device)

        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=precision):
            if use_coords:
                embeddings = model(imgs, batch['coords'].to(device))
            else:
                embeddings = model(imgs)
        if batch_idx == 0:
            mode = 'w'
        else:
            mode = 'a'
        asset_dict = {'embeddings': embeddings.cpu().numpy()}
        asset_dict.update({key: np.array(val) for key, val in batch.items() if key != 'imgs' and key != 'barcodes'})
        asset_dict.update({'barcodes': np.array(batch['barcodes'], dtype="<U100").astype('S')})
        # asset_dict.update({'barcodes': np.array(batch["barcodes"], dtype='|S4')})
        save_hdf5(embedding_save_path,
                  asset_dict=asset_dict,
                  mode=mode)
    return embedding_save_path


def embed_sample(args, sample_ids):
    device = 0
    precisions = {'fp16': torch.float16, 
                  'fp32': torch.float32}
    precision = precisions.get(args.precision, torch.float32)

    lazy_enc = LazyEncoder(args.feature_encoder, weights_root=args.weights_root)
    encoder, img_transforms = lazy_enc.get_model(device)

    for sample_id in sample_ids:
        tile_h5_path = os.path.join(args.source_dataroot, "patches", f"{sample_id}.h5")
        assert os.path.isfile(tile_h5_path), f"Tile file {tile_h5_path} does not exist"

        embedding_dir = os.path.join(args.embed_dataroot, sample_id, lazy_enc.name, args.precision)
        os.makedirs(embedding_dir, exist_ok=True)
        embed_path = os.path.join(embedding_dir, f'{sample_id}.h5')

        # if not os.path.isfile(embed_path) or (args.overwrite and not check_align(embed_path, tile_h5_path)):
        if not os.path.isfile(embed_path) or args.overwrite:
            try:
                tile_dataset = H5TileDataset(tile_h5_path, chunk_size=args.batch_size, img_transform=img_transforms)
                tile_dataloader = torch.utils.data.DataLoader(tile_dataset, 
                                                            batch_size=1, 
                                                            shuffle=False,
                                                            num_workers=args.num_workers)
                _ = embed_tiles(sample_id, tile_dataloader, encoder, embed_path, device, precision=precision, use_coords=(lazy_enc.name=="gigapathslide"))
                
                print(f"Embedding {sample_id} done")
            except Exception as e:
                print(f"{sample_id}: {e}")
                continue
        else:
            print(f"Skipping {sample_id} as it already exists")


def build_hvg_and_sparify_gene_exp(args, sample_ids, n_top_genes=200):
    for sample_id in tqdm(sample_ids, ncols=130, total=len(sample_ids)):
        try:
            h5ad_path = os.path.join(args.source_dataroot, f"st/{sample_id}.h5ad")
            adata = sc.read_h5ad(h5ad_path)
            adata_ = adata.copy()
            sc.pp.log1p(adata_)
            sc.pp.filter_genes(adata_, min_cells=np.ceil(0.1 * len(adata_.obs)))
            sc.pp.highly_variable_genes(adata_, n_top_genes=n_top_genes)
            hvg_names = [i for i in adata_.var["dispersions_norm"][adata_.var.highly_variable].sort_values()[::-1].keys()]
            adata.uns["hvg_names"] = hvg_names

            if isinstance(adata.X, np.ndarray):
                adata.X = sp.coo_matrix(adata.X).tocsr()
            if len(adata.X.indptr) != adata.X.shape[0] + 1:
                adata.X = adata.X.tocsr()
            adata.write(os.path.join(args.source_dataroot, f"st/{sample_id}.h5ad"))
            
        except Exception as e:
            print(f"{sample_id}: {e}")
            continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dataroot', default="/home/ti.huang/project/single_cell_dataset/hest/")
    parser.add_argument('--weights_root', type=str, default="/home/ti.huang/project/single_cell_dataset/weights_root")
    parser.add_argument('--embed_dataroot', type=str, default="/home/ti.huang/project/single_cell_dataset/hest/embeddings")
    parser.add_argument('--feature_encoder', type=str, default='gigapath', help="ciga | resnet50_trunc | uni_v1_official | gigapath")
    parser.add_argument('--overwrite', default=True, help='overwrite existing results')

    # training hyperparameters
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--precision', type=str, default='fp32', help='Precision (fp32 or fp16)')
    parser.add_argument('--img_resize', type=int, default=224, help='Image resizing (-1 to not resize)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for dataloader')
    args = parser.parse_args()

    args.feature_dim = {
        "uni_v1_official": 1024,
        "gigapath": 1536,
        "ciga": 512,
    }[args.feature_encoder]

    data_overview = pd.read_csv(os.path.join(args.source_dataroot, "HEST_v1_1_0.csv"))
    human_sample_ids = data_overview[data_overview["species"] == "Homo sapiens"]["id"].tolist()
    mouse_sample_ids = data_overview[data_overview["species"] == "Mus musculus"]["id"].tolist()
    
    # embedding all spots in all samples
    embed_sample(args, sample_ids=human_sample_ids)

    # build hvg and sparify gene exp
    build_hvg_and_sparify_gene_exp(args, human_sample_ids)
