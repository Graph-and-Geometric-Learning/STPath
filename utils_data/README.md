The samples included in following four files should be removed during pretraining:
- `stimage_overlap_samples.json`: The samples from STImage-1K4M that overlap with those in the HEST dataset.
    - The key `publications` is the source publications of these samples, which serve as the criterion for identifying overlaps.
- `stimage_domain_annos_eval.json`: The samples with domain annotations (brain layers) in STImage-1K4M for evaluation.
- `stimage_cancer_annos_eval.json`: The samples with disease annotations (tumor subtypes) in STImage-1K4M for evaluation.
- `hest_anno_samples.json`: The samples from HEST that originate from the same publications as annotated samples in STImage-1K4M.
