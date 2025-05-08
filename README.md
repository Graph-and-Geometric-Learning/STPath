# STPath: A Generative Foundation Model for Integrating Spatial Transcriptomics and Whole Slide Images

This is our PyTorch implementation for the paper:

> Tinglin Huang, Tianyu Liu, Mehrtash Babadi, Rex Ying, and Wengong Jin (2025). STPath: A Generative Foundation Model for Integrating Spatial Transcriptomics and Whole Slide Images. Paper in [bioRxiv](https://www.biorxiv.org/content/biorxiv/early/2025/04/24/2025.04.19.649665.full.pdf).

## Dataset Preparation

First, download the datasets from the following links:

```
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

## TODO

* Dataset preprocessing pipeline
* Training pipeline
* Evaluation pipeline
    * Gene expression prediction
    * Spatial clustering
    * Imputation
    * Biomarker analysis
    * Weakly-supervised classification
* An easy-to-use interface for users to perform inference