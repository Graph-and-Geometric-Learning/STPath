import torch
import torch.nn as nn
from .utils import *


def RegressionHead(
    d_model: int,
    output_dim: int,
    hidden_dim: int | None = None,
) -> nn.Module:
    """Single-hidden layer MLP for supervised output.

    Args:
        d_model: input dimension
        output_dim: dimensionality of the output.
        hidden_dim: optional dimension of hidden layer, defaults to d_model.
    Returns:
        output MLP module.
    """
    hidden_dim = hidden_dim if hidden_dim is not None else d_model
    return nn.Sequential(
        nn.Linear(d_model, hidden_dim),
        nn.GELU(),
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, output_dim),
    )



def batched_index_select(values, indices, dim = 1):
    value_dims = values.shape[(dim + 1):]
    values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
    indices = indices[(..., *((None,) * len(value_dims)))]
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
    value_expand_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_len))
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    values = values.expand(*value_expand_shape)

    dim += value_expand_len
    return values.gather(dim, indices)


# scale the row vectors of coordinates such that their root-mean-square norm is one
class NormCoordLN(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super(NormCoordLN, self).__init__()
        self.eps = eps

    def _norm_no_nan(self, x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
        '''
        L2 norm of tensor clamped above a minimum value `eps`.
        
        :param sqrt: if `False`, returns the square of the L2 norm
        '''
        # clamp is slow
        # out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
        out = torch.sum(torch.square(x), axis, keepdims) + eps
        return torch.sqrt(out) if sqrt else out

    def forward(self, coords):
        # coords: [..., N, 3]
        vn = self._norm_no_nan(coords, axis=-1, keepdims=True, sqrt=False, eps=self.eps)
        nonzero_mask = (vn > 2 * self.eps)
        vn = torch.sum(vn * nonzero_mask, dim=-2, keepdim=True
            ) / (self.eps + torch.sum(nonzero_mask, dim=-2, keepdim=True))
        vn = torch.sqrt(vn + self.eps)
        return nonzero_mask * (coords / vn)


def get_activation(activation="gelu"):
    return {
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "relu": nn.ReLU,
    }[activation]


def MLPWrapper(in_features, hidden_features, out_features, activation="gelu", norm_layer=None, bias=True, drop=0., drop_last=True):
    if activation == "swiglu":
        return SwiGLUMLP(in_features, hidden_features, out_features, norm_layer, bias, drop, drop_last)
    else:
        return MLP(in_features, hidden_features, out_features, get_activation(activation), norm_layer, bias, drop, drop_last)


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.,
        drop_last=True,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop) if drop_last else nn.Identity()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SwiGLUMLP(nn.Module):
    """ MLP w/ GLU style gating
    See: https://arxiv.org/abs/1612.08083, https://arxiv.org/abs/2002.05202
    """
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        norm_layer=None,
        bias=True,
        drop=0.,
        drop_last=True,
    ):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        assert hidden_features % 2 == 0

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = nn.SiLU()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_features // 2) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features // 2, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop) if drop_last else nn.Identity()

    def init_weights(self):
        # override init of fc1 w/ gate portion set to weight near zero, bias=1
        fc1_mid = self.fc1.bias.shape[0] // 2
        nn.init.ones_(self.fc1.bias[fc1_mid:])
        nn.init.normal_(self.fc1.weight[fc1_mid:], std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x1, x2 = x.chunk(2, dim=-1)
        x = self.act(x1) * x2
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


if __name__ == "__main__":
    model = MLP(10, 20, 30)
    x = torch.randn(1, 10)
    print(model(x).shape)

    model = SwiGLUMLP(10, 20, 30)
    x = torch.randn(1, 10)
    print(model(x).shape)