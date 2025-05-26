import scanpy as sc


def get_normalize_method(normalize_method, **kwargs):
    if normalize_method is None:
        return None
    elif normalize_method == "log1p":
        return log1p
    elif normalize_method == "log1pv2":
        return log1p_rm_special_genes
    elif normalize_method == "log1p_w_scaling":
        return log1p_w_scaling
    else:
        raise ValueError(f"Unknown normalize method: {normalize_method}")


def identity(adata):
    return adata.copy()


def log1p(adata):
    process_data = adata.copy()
    sc.pp.log1p(process_data)
    return process_data


def log1p_rm_special_genes(adata):
    # adata.obs_names_make_unique()
    process_data = adata.copy()
    sc.pp.filter_genes(process_data, min_counts=3)  # filter genes by counts
    process_data.var_names_make_unique()
    sc.pp.log1p(process_data)
    return process_data


# log1p-normalized with a scale factor of 100
def log1p_w_scaling(adata):
    process_data = adata.copy()
    sc.pp.normalize_total(process_data, target_sum=100, inplace=True)
    sc.pp.log1p(process_data)
    return process_data
