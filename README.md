# STPath: A Generative Foundation Model for Integrating Spatial Transcriptomics and Whole Slide Images

This is our PyTorch implementation for the paper:

> Tinglin Huang, Tianyu Liu, Mehrtash Babadi, Rex Ying, and Wengong Jin (2025). STPath: A Generative Foundation Model for Integrating Spatial Transcriptomics and Whole Slide Images. Paper in [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.04.19.649665v2.abstract).

## Dataset Preparation

First, download the datasets from the following links:

```python
import datasets
from huggingface_hub import snapshot_download

# download the STimage-1K4M dataset
snapshot_download(
    repo_id="jiawennnn/STimage-1K4M", 
    repo_type='dataset', 
    local_dir='your_dir/stimage_1k4m',
)

# download the HEST-1K dataset
datasets.load_dataset(
    'MahmoodLab/hest', 
    cache_dir="your_dir/hest",
    patterns='st/*'
)

# download the HEST-bench dataset for gene expression prediction
snapshot_download(
    repo_id="MahmoodLab/hest-bench", 
    repo_type='dataset', 
    local_dir='your_dir/hest-bench',
    ignore_patterns=['fm_v1/*']
)
```

Then, download the pre-trained model weight of Gigapath following the official repository [link](https://github.com/prov-gigapath/prov-gigapath).


## Requirements

The code has been tested running under Python 3.10.16. The required packages are as follows:
- pytorch == 2.3.1
- torch_geometric == 2.6.1
- einops == 0.8.0

Once you finished these installation, please run install the package by running:
```
pip install -e .
```

## Usage

We provide an easy-to-use interface for users to perform inference on the pre-trained model, which can be found in `app/pipeline/inference.py`. Specifically, the following code snippet shows how to use it:

```python
from stpath.app.pipeline.inference import STPathInference

agent = STPathInference(
    gene_voc_path='STPath_dir/utils_data/symbol2ensembl.json',
    model_weight_path='your_dir/stpath.pkl', 
    device=0
)

pred_adata = agent.inference(
    coords=coords, # [number_of_spots, 2]
    img_features=embeddings,  # [number_of_spots, 1536], the image features extracted using Gigapath 
    organ_type="Kidney",  # Default is None
    tech_type="Visium",  # Default is None
    save_gene_names=hvg_list  # a list of gene names to save in the adata, e.g., ['GATA3', 'UBLE2C', ...]. None will save all genes in the model.
)

# save adata
pred_adata.write_h5ad(f"your_dir/pred_{sample_id}.h5ad")
```

The vocabularies for organs and technologies can be found in the following locations:
* [organ vocabulary](https://github.com/Graph-and-Geometric-Learning/STPath/blob/main/stpath/utils/constants.py#L98)
* [tech vocabulary](https://github.com/Graph-and-Geometric-Learning/STPath/blob/main/stpath/utils/constants.py#L20) 

If the organ type or the tech type is unknown, you can set them to `None` in the inference function. Besides, the predicted gene expression values are log1p-transformed (`log(1 + x)`), consistent with the transformation applied during the training of STPath.


### Example of Inference

Here, we provide an example of how to perform inference on a [sample](https://github.com/Graph-and-Geometric-Learning/STPath/tree/main/example_data) from the HEST dataset:

```python
from scipy.stats import pearsonr
from stpath.hest_utils.st_dataset import load_adata
from stpath.hest_utils.file_utils import read_assets_from_h5

sample_id = "INT2"
source_dataroot = "STPath_dir"  # the root directory of the STPath repository
with open(os.path.join(source_dataroot, "example_data/var_50genes.json")) as f:
    hvg_list = json.load(f)['genes']

data_dict, _ = read_assets_from_h5(os.path.join(source_dataroot, f"{sample_id}.h5"))  # load the data from the h5 file
coords = data_dict["coords"]
embeddings = data_dict["embeddings"]
barcodes = data_dict["barcodes"].flatten().astype(str).tolist()
adata = sc.read_h5ad(os.path.join(source_dataroot, f"{sample_id}.h5ad"))[barcodes, :]

# The return pred_adata includes the expressions of the genes in hvg_list, which is a list of highly variable genes.
pred_adata = agent.inference(
    coords=coords, 
    img_features=embeddings, 
    organ_type="Kidney", 
    tech_type="Visium",
    save_gene_names=hvg_list  # we only need the highly variable genes for evaluation
)

# calculate the Pearson correlation coefficient between the predicted and ground truth gene expression
all_pearson_list = []
gt = np.log1p(adata[:, hvg_list].X.toarray())  # sparse -> dense
# go through each gene in the highly variable genes list
for i in range(len(hvg_list)):
    pearson_corr, _ = pearsonr(gt[:, i], pred_adata.X[:, i])
    all_pearson_list.append(pearson_corr.item())
print(f"Pearson correlation for {sample_id}: {np.mean(all_pearson_list)}")  # 0.1562
```

### In-context Learning

STPath also support in-context learning, which allows users to provide the expression of a few spots to guide the model to predict the expression of other spots:

```python
from stpath.data.sampling_utils import PatchSampler

rightest_coord = np.where(coords[:, 0] == coords[:, 0].max())[0][0]
masked_ids = PatchSampler.sample_nearest_patch(coords, int(len(coords) * 0.95), rightest_coord)  # predict the expression of the 95% spots
context_ids = np.setdiff1d(np.arange(len(coords)), masked_ids)  # the index not in masked_ids will be used as context
context_gene_exps = adata.X.toarray()[context_ids]
context_gene_names = adata.var_names.tolist()

pred_adata = agent.inference(
    coords=coords, 
    img_features=embeddings, 
    context_ids=context_ids,  # the index of the context spots
    context_gene_exps=context_gene_exps,   # the expression of the context spots
    context_gene_names=context_gene_names,   # the gene names of the context spots
    organ_type="Kidney", 
    tech_type="Visium", 
    save_gene_names=hvg_list,
)

all_pearson_list = []
gt = np.log1p(adata[:, hvg_list].X.toarray())[masked_ids, :]  # groundtruth expression of the spots in masked_ids
pred = pred_adata.X[masked_ids, :]  # predicted expression of the spots in masked_ids
for i in range(len(hvg_list)):
    pearson_corr, _ = pearsonr(gt[:, i], pred[:, i])
    all_pearson_list.append(pearson_corr.item())
print(f"Pearson correlation for {sample_id}: {np.mean(all_pearson_list)}")  # 0.2449
```

## TODO

* Dataset preprocessing pipeline [x]
* Upload pretrained weight
* Training pipeline
* Evaluation pipeline
    * Gene expression prediction [x]
    * Imputation [x]
    * Spatial clustering
    * Biomarker analysis
    * Weakly-supervised classification
* An easy-to-use interface for users to perform inference [x]


## Reference

If you find our work useful in your research, please consider citing our paper:

```
@inproceedings{huang2025stflow,
  title={Scalable Generation of Spatial Transcriptomics from Histology Images via Whole-Slide Flow Matching},
  author={Huang, Tinglin and Liu, Tianyu and Babadi, Mehrtash and Jin, Wengong and Ying, Rex},
  booktitle={International Conference on Machine Learning},
  year={2025}
}

@article{huang2025stpath,
  title={STPath: A Generative Foundation Model for Integrating Spatial Transcriptomics and Whole Slide Images},
  author={Huang, Tinglin and Liu, Tianyu and Babadi, Mehrtash and Ying, Rex and Jin, Wengong},
  journal={bioRxiv},
  pages={2025--04},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}
```