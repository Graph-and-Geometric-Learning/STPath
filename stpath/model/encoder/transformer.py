import torch
import torch.nn as nn

from stflow.model.nn_utils import MLPWrapper
from stflow.model.nn_utils.attention import Attention


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

    def forward(self, token_embs, padding_mask=None):
        context_token_embs = self.attn(token_embs, padding_mask)
        token_embs = token_embs + context_token_embs

        token_embs = token_embs + self.mlp(token_embs)
        return token_embs


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()

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
        batch_mask = batch_mask.unsqueeze(0)
        for blk in self.blks:
            features = blk(features, padding_mask=batch_mask)
        return features.squeeze(0)  # [N_cells, N_genes]


if __name__ == "__main__":
    # Define the model
    from stflow.model.nn_utils.config import ModelConfig

    model = Transformer(
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
    output = model(features[3:6], coords, batch_idx)
    print(output)


