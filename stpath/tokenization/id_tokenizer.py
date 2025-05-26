import torch
import torch.nn.functional as F

from stpath.tokenization.tokenizer_base import TokenizerBase
import stpath.utils.constants as C 


class IDTokenizer(TokenizerBase):
    def __init__(
        self,
        id_type="tech",
    ):
        if id_type == "tech":
            token_aligner = C.tech_align_mapping
            all_tokens = C.tech_voc
        elif id_type == "specie":
            token_aligner = C.specie_align_mapping
            all_tokens = C.specie_voc
        elif id_type == "organ":
            token_aligner = C.organ_align_mapping
            all_tokens = C.organ_voc

        self.token2id = {token: i for i, token in enumerate(all_tokens)}
        self.id2token = {i: token for token, i in self.token2id.items()}
        self.token_aligner = token_aligner
    
    def tokenize(self, token):
        if token not in self.token2id:
            return self.token2id["<unk>"]
        return self.token2id[token]

    def align(self, input):
        if isinstance(input, str):
            return self.token_aligner[input] if input in self.token_aligner else "<pad>"
        elif isinstance(input, list):
            return [self.token_aligner[token] if token in self.token_aligner else "<pad>" for token in sequence]

    def encode(self, input, align_first=False):
        if align_first:
            input = self.align(input)

        if isinstance(input, str):
            return self.token2id[input]
        elif isinstance(input, list):
            return [self.token2id[token] for token in input]

    def decode(self, input):
        if isinstance(input, int):
            return self.id2token[input]
        elif isinstance(input, list):
            return [self.id2token[i] for i in input]

    @property
    def n_tokens(self):
        return len(self.token2id)

    @property
    def mask_token(self) -> str:
        return "<mask>"

    @property
    def mask_token_id(self) -> int:
        return self.token2id[self.mask_token]

    @property
    def pad_token(self) -> str:
        return "<pad>"

    @property
    def pad_token_id(self) -> int:
        return self.token2id[self.pad_token]

    @property
    def unk_token(self) -> str:
        return "<unk>"

    @property
    def unk_token_id(self) -> int:
        return self.token2id[self.unk_token]


if __name__ == "__main__":
    tokenizer = IDTokenizer()
    sequence = ["Visium", "Xenium", "VisiumHD"]
    aligned_sequence = tokenizer.align(sequence)
    print(aligned_sequence)
    tokens = tokenizer.encode(aligned_sequence)
    print(tokens)
