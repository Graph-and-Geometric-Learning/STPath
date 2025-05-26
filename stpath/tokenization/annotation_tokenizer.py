import torch
import torch.nn.functional as F

from stpath.tokenization.tokenizer_base import TokenizerBase
import stpath.utils.constants as C 


class AnnotationTokenizer(TokenizerBase):
    def __init__(
        self,
        id_type="disease",
    ):
        if id_type == "disease":
            token_aligner = C.cancer_annotation_align_mapping
            all_tokens = C.cancer_annotation_voc
        elif id_type == "domain":
            token_aligner = C.domain_annotation_align_mapping
            all_tokens = C.domain_annotation_voc

        self.token2id = {token: i for i, token in enumerate(all_tokens)}
        self.token_aligner = token_aligner
    
    def tokenize(self, token):
        if token not in self.token2id:
            return self.token2id["<unk>"]
        return self.token2id[token]

    def align(self, input):
        if isinstance(input, str):
            return self.token_aligner[input] if input in self.token_aligner else "<unk>"
        elif isinstance(input, list):
            return [self.token_aligner[token] if token in self.token_aligner else "<unk>" for token in input]

    def encode(self, input, align_first=False):
        if align_first:
            input = self.align(input)

        if isinstance(input, str):
            return self.token2id[input]
        elif isinstance(input, list):
            return [self.token2id[token] for token in input]

    @property
    def n_tokens(self):
        return len(self.token2id)

    @property
    def mask_token(self) -> str:
        return self.token2id["<mask>"]

    @property
    def pad_token(self) -> str:
        return self.token2id["<pad>"]

    @property
    def unk_token(self) -> str:
        return self.token2id["<unk>"]
