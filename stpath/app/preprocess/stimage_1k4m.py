import os
import argparse
import numpy as np
import anndata as ad
import scanpy as sc
import pandas as pd
from time import time
from tqdm import tqdm
import scipy.sparse as sp
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from stpath.hest_utils.encoder import load_encoder
from stpath.hest_utils.file_utils import save_hdf5


Image.MAX_IMAGE_PIXELS = 3080000000000000000000000000000000000


class ImageDataset(Dataset):
    def __init__(self, coord_csv_path, img_path, img_transform=None):
        
        coord = pd.read_csv(coord_csv_path)
        temp_image = Image.open(img_path)
        temp_image = temp_image.convert('RGB')

        self.crop_images, self.coords = [], []

        r = coord.r[0]
        for i in range(len(coord.index)):
            yaxis = coord.yaxis[i]
            xaxis = coord.xaxis[i]
            spot_name = coord.iloc[i,0]
            image_crop = temp_image.crop((xaxis-r, yaxis-r, xaxis+r, yaxis+r))
            
            self.crop_images.append(image_crop)  
            self.coords.append((xaxis, yaxis))
        
        self.barcodes = coord["Unnamed: 0"]
        self.img_transform = img_transform
        self.coords = torch.tensor(self.coords, dtype=torch.float32)
        
    def __len__(self):
        return len(self.crop_images)

    def __getitem__(self, idx):
        img = self.crop_images[idx]
        coord = self.coords[idx].squeeze(0)
        barcode = self.barcodes[idx]

        if self.img_transform:
            img = self.img_transform(img)

        return img, coord, barcode


def padding_batcher():
    def batcher_dev(batch):
        imgs = [d[0] for d in batch]
        coords = [d[1] for d in batch]
        barcodes = [d[2] for d in batch]
        return {"imgs": torch.stack(imgs), "coords":torch.stack(coords), "barcodes": barcodes}
    return batcher_dev


def merge_annotation_into_adata(args, annotation_path):
    for annotation_file in tqdm(os.listdir(annotation_path), ncols=100, desc="Merging annotation"):
        if annotation_file.endswith(".csv"):
            try:
                sample_id = annotation_file.split(".csv")[0].replace("_anno", "")
                adata = sc.read_h5ad(os.path.join(args.source_dataroot, "st", f"{sample_id}.h5ad"))
                annotation = pd.read_csv(os.path.join(annotation_path, annotation_file), index_col=0)
                annotation.rename(columns={annotation.columns[0]: 'annotation'}, inplace=True)
                annotation["annotation"] = annotation["annotation"].str.lower()
                adata.obs = adata.obs.join(annotation)
                adata.write(os.path.join(args.source_dataroot, "st", f"{sample_id}.h5ad"))
            except Exception as e:
                print(f"{sample_id}: {e}")
                continue


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
        save_hdf5(embedding_save_path,
                  asset_dict=asset_dict,
                  mode=mode)
    return embedding_save_path


def embed_sample(args, sample_ids, techs):

    device = 0
    precisions = {'fp16': torch.float16, 
                  'fp32': torch.float32}
    precision = precisions.get(args.precision, torch.float32)

    lazy_enc = LazyEncoder(args.feature_encoder, weights_root=args.weights_root)
    encoder, _ = lazy_enc.get_model(device)
    img_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for sample_id, tech in zip(sample_ids, techs):

        coord_csv_path = os.path.join(args.source_dataroot, f"{tech}/coord/{sample_id}_coord.csv")
        img_path = os.path.join(args.source_dataroot, f"{tech}/image/{sample_id}.png")

        embedding_dir = os.path.join(args.embed_dataroot, sample_id, lazy_enc.name, args.precision)
        os.makedirs(embedding_dir, exist_ok=True)

        embed_path = os.path.join(embedding_dir, f'{sample_id}.h5')

        if not os.path.isfile(embed_path) or args.overwrite:
            try:
                dataset = ImageDataset(coord_csv_path, img_path, img_transform=img_transforms)
                dataloader = torch.utils.data.DataLoader(dataset, 
                                                        batch_size=args.batch_size, 
                                                        shuffle=False,
                                                        num_workers=args.num_workers,
                                                        collate_fn=padding_batcher())
                _ = embed_tiles(sample_id, dataloader, encoder, embed_path, device, precision=precision, use_coords=(lazy_enc.name=="gigapathslide"))
            except Exception as e:
                print(f"{sample_id}: {e}")
                continue
        else:
            print(f"Skipping {sample_id} as it already exists")


def convert_to_h5ad(sample_ids, sample_techs, data_path, save_path):
    for sample_id, tech in tqdm(zip(sample_ids, sample_techs), ncols=130, desc="Converting to h5ad", total=len(sample_ids)):

        if os.path.isfile(os.path.join(save_path, f"{sample_id}.h5ad")):
            # print(f"Skipping {sample_id}")
            continue

        try:
            sample_id = sample_id.split(".png")[0]
            coord = pd.read_csv(os.path.join(data_path, tech, f"coord/{sample_id}_coord.csv"),index_col=0)
            gene_exp = pd.read_csv(os.path.join(data_path, tech, f"gene_exp/{sample_id}_count.csv"),sep=',',index_col=0)
            gene_exp = gene_exp.loc[coord.index]

            adata = ad.AnnData(X=gene_exp)
            adata.obs["x_coord"] = coord["xaxis"].to_numpy()
            adata.obs["y_coord"] = coord["yaxis"].to_numpy()
            adata.obs["r"] = coord["r"].to_numpy()
            adata.obs["tech"] = tech

            # build hvg
            adata_ = adata.copy()
            sc.pp.log1p(adata_)
            sc.pp.filter_genes(adata_, min_cells=np.ceil(0.1 * len(adata_.obs)))
            sc.pp.highly_variable_genes(adata_, n_top_genes=200)
            hvg_names = [i for i in adata_.var["dispersions_norm"][adata_.var.highly_variable].sort_values()[::-1].keys()]
            adata.uns["hvg_names"] = hvg_names

            if isinstance(adata.X, np.ndarray):
                adata.X = sp.coo_matrix(adata.X).tocsr()
            if len(adata.X.indptr) != adata.X.shape[0] + 1:
                adata.X = adata.X.tocsr()
            adata.write(os.path.join(save_path, f"{sample_id}.h5ad"))
        
        except Exception as e:
            print(f"{sample_id}: {e}")
            continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dataroot', default="/home/ti.huang/project/single_cell_dataset/stimage_1k4m/")
    parser.add_argument('--weights_root', type=str, default="/home/ti.huang/project/single_cell_dataset/weights_root")
    parser.add_argument('--embed_dataroot', type=str, default="/home/ti.huang/project/single_cell_dataset/stimage_1k4m/embeddings")
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

    data_overview = pd.read_csv(os.path.join(args.source_dataroot, "meta/meta_all_gene.csv"))
    human_sample_ids = data_overview[data_overview["species"] == "human"]["slide"].tolist()
    mouse_sample_ids = data_overview[data_overview["species"] == "mouse"]["slide"].tolist()
    others_sample_ids = data_overview[(data_overview["species"] != "human") & (data_overview["species"] != "mouse")]["slide"].tolist()
    
    human_sample_techs = data_overview[data_overview["species"] == "human"]["tech"].tolist()
    mouse_sample_techs = data_overview[data_overview["species"] == "mouse"]["tech"].tolist()
    others_sample_techs = data_overview[(data_overview["species"] != "human") & (data_overview["species"] != "mouse")]["tech"].tolist()

    # merging the annotation into adata
    merge_annotation_into_adata(args, os.path.join(args.source_dataroot, "annotation"))

    # embedding all spots in all samples
    embed_sample(args, sample_ids=human_sample_ids, techs=human_sample_techs)

    # build hvg and sparify gene exp
    os.makedirs(os.path.join(args.source_dataroot, "st"), exist_ok=True)
    convert_to_h5ad(human_sample_ids, human_sample_techs, args.source_dataroot, os.path.join(args.source_dataroot, "st/"))
