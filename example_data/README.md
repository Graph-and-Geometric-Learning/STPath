This folder provides an example demonstrating how to use STPath:
- `var_50genes.json`: A JSON file containing the top 50 highly variable genes. [Source](https://huggingface.co/datasets/MahmoodLab/hest-bench/blob/main/CCRCC/var_50genes.json).
- `INT2.h5ad`: An HDF5 file containing the spatial transcriptomics data for the INT2 sample. The original file can be found in [Source](https://huggingface.co/datasets/MahmoodLab/hest/blob/main/st/INT2.h5ad). We additionally sparsified the gene expression matrix to reduce file size and computed the highly variable genes (HVGs). The processing steps and generation of logits are detailed in the [code](https://github.com/Graph-and-Geometric-Learning/STPath/tree/main/app/preprocess/hest.py).
- `INT2.h5`: Visual features for each spatial spot, extracted using the pathology foundation model. The extraction pipeline is also available in the [code](https://github.com/Graph-and-Geometric-Learning/STPath/tree/main/app/preprocess/hest.py). You can load the data with the following code:

```
from stpath.hest_utils.file_utils import read_assets_from_h5
data_dict, _ = read_assets_from_h5(os.path.join(source_dataroot, f"{sample_id}.h5"))
print(data_dict.keys())  # dict_keys(['coords', 'embeddings', 'barcodes'])
```