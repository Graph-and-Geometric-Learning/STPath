import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from time import time

from stpath.model.nn_utils import MLPWrapper
from stpath.model.nn_utils.fa import FrameAveraging


class Attention(FrameAveraging):
    def __init__(
        self,
        d_model, 
        n_heads=1,
        proj_drop=0.,
        attn_drop=0.,
        max_n_tokens=5e3,
    ):
        super(Attention, self).__init__(dim=2)

        self.max_n_tokens = max_n_tokens
        self.d_head, self.n_heads = d_model // n_heads, n_heads
        self.scale = self.d_head ** -0.5

        self.layernorm_qkv = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 3),
        )
        self.W_output = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(proj_drop),
        )
        self.attn_dropout = nn.Dropout(attn_drop)
        self.edge_bias = nn.Sequential(
            nn.Linear(self.dim + 1, self.n_heads, bias=False),
        )

    def forward(self, x, coords, pad_mask: torch.Tensor=None):
        B, N, C = x.shape
        q, k, v = self.layernorm_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda x: rearrange(x, 'b n (h d) -> b h n d', h=self.n_heads), (q, k, v))

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        """build pairwise representation with FA"""
        radial_coords = coords.unsqueeze(dim=2) - coords.unsqueeze(dim=1)  # [B, N, N, 2]
        radial_coord_norm = radial_coords.norm(dim=-1).reshape(B*N, N, 1)  # [B*N, N, 1]

        radial_coords = rearrange(radial_coords, 'b n m d -> (b n) m d')
        neighbor_masks = ~rearrange(pad_mask, 'b n m -> (b n) m') if pad_mask is not None else None
        frame_feats, _, _ = self.create_frame(radial_coords, neighbor_masks)  # [B*N*4, N, 2]
        frame_feats = frame_feats.view(B * N, self.n_frames, N, -1)  # [N, 4, N, d_model]

        radial_coord_norm = radial_coord_norm.unsqueeze(dim=1).expand(B * N, self.n_frames, N, -1)

        # for efficiency
        spatial_bias = self.edge_bias(torch.cat([frame_feats, radial_coord_norm], dim=-1)).mean(dim=1)  # [B * N, N, n_heads]
        spatial_bias = rearrange(spatial_bias, '(b n) m h -> b h n m', b=B, n=N)

        """add spatial bias"""
        attn = attn + spatial_bias
        if pad_mask is not None:
            pad_mask = pad_mask.unsqueeze(1)
            attn.masked_fill_(pad_mask, -1e9)
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        return self.W_output(x)


class TransformerBlock(nn.Module):
    def __init__(            
        self,
        d_model,
        n_heads=1,
        activation="gelu",
        attn_drop=0.,
        proj_drop=0.,
        mlp_ratio=4.0,
    ):
        super(TransformerBlock, self).__init__()

        self.attn = Attention(
            d_model=d_model, n_heads=n_heads, proj_drop=proj_drop, attn_drop=attn_drop
        )

        self.mlp = MLPWrapper(
            in_features=d_model, hidden_features=int(d_model * mlp_ratio), out_features=d_model,
            activation=activation, drop=proj_drop, norm_layer=nn.LayerNorm
        )

    def forward(self, token_embs, coords, padding_mask=None):
        context_token_embs = self.attn(token_embs, coords, padding_mask)
        token_embs = token_embs + context_token_embs

        token_embs = token_embs + self.mlp(token_embs)
        return token_embs


class SpatialTransformer(nn.Module):
    def __init__(self, config):
        super(SpatialTransformer, self).__init__()

        self.blks = nn.ModuleList([
            TransformerBlock(config.d_model, n_heads=config.n_heads, activation=config.act, 
                              attn_drop=config.attn_dropout, proj_drop=config.dropout, mlp_ratio=config.mlp_ratio,
                            ) \
                for i in range(config.n_layers)
        ])

    def forward(self, features, coords, batch_idx, **kwargs):
        # apply the same mask to all cells in the same batch
        batch_mask = ~(batch_idx.unsqueeze(0) == batch_idx.unsqueeze(1))

        # forward pass
        features = features.unsqueeze(0)  # [1, N_cells, d_model]
        coords = coords.unsqueeze(0)  # [1, N_cells, 2]
        batch_mask = batch_mask.unsqueeze(0)  # [1, N_cells, N_cells]
        for blk in self.blks:
            features = blk(features, coords, padding_mask=batch_mask)
        return features.squeeze(0)  # [N_cells, N_genes]


if __name__ == "__main__":
    # Define the model
    from stflow.model.nn_utils.config import ModelConfig

    model = SpatialTransformer(
        ModelConfig(
            n_genes=100,
            d_model=12,
            n_layers=3,
            n_heads=4,
            dropout=0.1,
            attn_dropout=0.1,
            act="gelu",
            mlp_ratio=4.0,
        )
    )
    model.eval()

    # Test the model with dummy data
    features = torch.randn(10, 12)
    coords = torch.randn(10, 2)
    batch_idx = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
    output = model(features, coords, batch_idx)
    print(output[3:6])

    batch_idx = torch.tensor([0, 0, 0])
    output = model(features[3:6], coords[3:6], batch_idx)
    print(output)
