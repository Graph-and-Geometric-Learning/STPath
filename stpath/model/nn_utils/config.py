

class ModelConfig():

    def __init__(
        self,
        dim=3,  # dimension of the coordinates
        d_input=64, 
        d_model=64, 
        n_layers=4,
        n_genes=50,
        dropout=0.1,
        attn_dropout=0.1,
        n_neighbors=16,
        valid_radius=1e6,
        embedding_grad_frac=1.0,
        n_heads=4,
        rbf_count=64,
        rbf_sigma=0.1,
        act="gelu",
        **kwargs,
    ):

        self.dim = dim
        self.d_input = d_input
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_genes = n_genes
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.n_neighbors = n_neighbors
        self.valid_radius = valid_radius
        self.embedding_grad_frac = embedding_grad_frac
        self.n_heads = n_heads
        self.act = act
        self.rbf_count = rbf_count
        self.rbf_sigma = rbf_sigma

        for key, value in kwargs.items():
            setattr(self, key, value)
        
        self._hidden_dim_check()

    def _hidden_dim_check(self):
        # check if d_model/d_edge_model can be divided by n_heads
        assert self.d_model % self.n_heads == 0, f"d_model should be divisible by n_heads"

    @staticmethod
    def get_default_config():
        return ModelConfig(
            n_genes=38984,
            d_input=1536,
            d_model=512,
            n_layers=4,
            n_heads=4,
            dropout=0.1,
            attn_dropout=0.1,
            act="gelu",
            mlp_ratio=2.0,
        )


if __name__ == "__main__":
    config = ModelConfig.get_default_config()
    print(config.n_genes)
