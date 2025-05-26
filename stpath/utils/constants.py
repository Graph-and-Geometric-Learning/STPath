import os
from functools import cache
from pathlib import Path

# gene_voc = {
#     "hest": "/home/ti.huang/STPath/utils_data/hest_symbol2ensembl.json",
#     "stimage_1k4m": "/home/ti.huang/STPath/utils_data/stimage_symbol2ensembl.json",
# }


# technology
tech_align_mapping = {
    "ST": "Spatial Transcriptomics",
    "Visium": "Visium",
    "Xenium": "Xenium",
    "VisiumHD": "Visium HD",
    "Visium HD": "Visium HD",
    "Spatial Transcriptomics": "Spatial Transcriptomics",
}
tech_voc = ["<pad>", "Spatial Transcriptomics", "Visium", "Xenium", "Visium HD"]


# species
specie_align_mapping = {
    "Mus musculus": "Mus musculus",
    "Homo sapiens": "Homo sapiens",
    "human": "Homo sapiens",
    "mouse": "Mus musculus",
    "plant": "others",
    "rattus norvegicus": "others",
    "human & mouse": "others",
    "pig": "others",
    "fish": "others",
    "frog": "others",
    "ambystoma mexicanum": "others"
}
specie_voc = ["<pad>", "<mask>", "<unk>", "Mus musculus", "Homo sapiens", "others"]


organ_align_mapping = {
    "Spinal cord": "Spinal cord",
    "Brain": "Brain", 
    "Breast": "Breast",
    "Bowel": "Bowel",
    "Skin": "Skin",          
    "Heart": "Heart",         
    "Kidney": "Kidney",        
    "Prostate": "Prostate",
    "Lung": "Lung",          
    "Liver": "Liver",
    "Uterus": "Uterus",
    "Bone": "Bone",
    "Muscle": "Muscle",
    "Eye": "Eye",
    "Pancreas": "Pancreas",

    "breast": "Breast",
    "brain": "Brain",
    "kidney": "Kidney",
    "heart": "Heart",
    "skin": "Skin",
    "liver": "Liver",
    "pancreas": "Pancreas",
    "mouth": "Mouth",
    "ovary": "Ovary",
    "prostate": "Prostate",
    "glioma": "Glioma",
    "glioblastoma": "Glioblastoma",
    "stomach": "Stomach",
    "colon": "Colon",
    "lung": "Lung",
    "pancreas": "Pancreas",
    "muscle": "Muscle",
    
    "Bladder": "Others",
    "Lymphoid": "Others",
    "Cervix": "Others",
    "Lymph node": "Others",
    "Ovary": "Others",
    "Embryo": "Others",
    "Lung/Brain": "Others",
    "Kidney/Brain": "Others",
    "Placenta": "Others",
    "Whole organism": "Others",
    "thymus": "Others",
    "joint": "Others",
    "undifferentiated pleomorphic sarcoma": "Others",
    "largeintestine": "Others",
    "lacrimal gland": "Others",
    "leiomyosarcoma": "Others",
    "endometrium": "Others",
    "brain+kidney": "Others",
    "cerebellum": "Others",
    "cervix": "Others",
    "colorectal": "Others",
    "lymphnode": "Others",
}
organ_voc = ["<pad>", "<mask>", "<unk>", "Spinal cord", "Brain", "Breast", "Bowel", "Skin", "Heart", "Kidney", "Prostate", 
             "Lung", "Liver", "Uterus", "Bone", "Muscle", "Eye", "Pancreas", "Mouth", "Ovary", "Glioma", "Glioblastoma",
             "Stomach", "Colon", "Others"]

# annotation
cancer_annotation_align_mapping = {
    'invasive': 'tumor',
    'invasive cancer': 'tumor',
    'tumor': 'tumor',
    'surrounding tumor': 'tumor',
    'immune infiltrate': 'tumor',
    'cancer in situ': 'tumor',
    'tumor stroma with inflammation': 'tumor',
    'tumor cells': 'tumor',
    'tumour cells': 'tumor',
    'tumor stroma fibrous': 'tumor',
    'tumor stroma': 'tumor',
    'fibrosis': 'tumor',
    'high tils stroma': 'tumor',
    'in situ carcinoma*': 'tumor',
    'in situ carcinoma': 'tumor',
    'tumor cells ?': 'tumor',
    'hyperplasia': 'tumor',
    'tumor cells - spindle cells': 'tumor',
    'tumour stroma': 'tumor',
    'necrosis': 'tumor',
    'tumor_edge_5': 'tumor',
    'idc_4': 'tumor',
    'idc_3': 'tumor',
    'idc_2': 'tumor',
    'tumor_edge_3': 'tumor',
    'idc_5': 'tumor',
    'dcis/lcis_4': 'tumor',
    'idc_7': 'tumor',
    'tumor_edge_1': 'tumor',
    'dcis/lcis_1': 'tumor',
    'dcis/lcis_2': 'tumor',
    'tumor_edge_4': 'tumor',
    'idc_1': 'tumor',
    'benign': 'tumor',
    'gg4 cribriform': 'tumor',
    'gg2': 'tumor',
    'chronic inflammation': 'tumor',
    'gg1': 'tumor',
    'transition_state': 'tumor',
    'benign*': 'tumor',
    'gg4': 'tumor',
    'pin': 'tumor',
    'inflammation': 'tumor',
    'dcis/lcis_3': 'tumor',
    'tumor_edge_2': 'tumor',
    'dcis/lcis_5': 'tumor',
    'idc_6': 'tumor',
    'tumor_edge_6': 'tumor',
    "tls": "tumor",
    "t_agg": "tumor",

    'healthy': 'healthy',
    'non tumor': 'healthy',
    'normal': 'healthy',
    'breast glands': 'healthy',
    'connective tissue': 'healthy',
    'adipose tissue': 'healthy',
    'artifacts': 'healthy',
    'normal epithelium': 'healthy',
    'lymphocytes': 'healthy',
    'healthy_2': 'healthy',
    'vascular': 'healthy',
    'peripheral nerve': 'healthy',
    'lymphoid stroma': 'healthy',
    'fibrous stroma': 'healthy',
    'fibrosis (peritumoral)': 'healthy',
    'artefacts': 'healthy',
    'endothelial': 'healthy',
    'healthy_1': 'healthy',
    'nerve': 'healthy',
    'fat': 'healthy',
    'stroma': 'healthy',
    'exclude': 'healthy',
    'vessel': 'healthy',
    "no_tls": "healthy",
}

cancer_annotation_voc = ["<pad>", "<mask>", "<unk>", "healthy", "tumor"]

domain_annotation_align_mapping = {
    "l1": "l1",
    "l2": "l2",
    "l3": "l3",
    "l4": "l4",
    "l5": "l5",
    "l6": "l6",
    "vm": "vm",
}

domain_annotation_voc = ["<pad>", "<mask>", "<unk>", "l1", "l2", "l3", "l4", "l5", "l6", "vm"]
