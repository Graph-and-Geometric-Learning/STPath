from .ge_tokenizer import GeneExpTokenizer
from .id_tokenizer import IDTokenizer
from .image_tokenizer import ImageTokenizer
from .annotation_tokenizer import AnnotationTokenizer


class TokenizerTools:
    ge_tokenizer: GeneExpTokenizer | None = None
    image_tokenizer: ImageTokenizer | None = None
    tech_tokenizer: IDTokenizer | None = None
    specie_tokenizer: IDTokenizer | None = None
    organ_tokenizer: IDTokenizer | None = None
    cancer_anno_tokenizer: AnnotationTokenizer | None = None
    domain_anno_tokenizer: AnnotationTokenizer | None = None
    
    def __init__(self, ge_tokenizer, image_tokenizer, tech_tokenizer, 
                 specie_tokenizer, organ_tokenizer, 
                 cancer_anno_tokenizer, domain_anno_tokenizer, **kwargs):

        self.ge_tokenizer = ge_tokenizer
        self.image_tokenizer = image_tokenizer
        self.tech_tokenizer = tech_tokenizer
        self.specie_tokenizer = specie_tokenizer
        self.organ_tokenizer = organ_tokenizer
        self.cancer_anno_tokenizer = cancer_anno_tokenizer
        self.domain_anno_tokenizer = domain_anno_tokenizer

        for k, v in kwargs.items():
            setattr(self, k, v)