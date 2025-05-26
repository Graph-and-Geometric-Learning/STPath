import timm
import torch
from transformers import ViTModel


class TimmCNNEncoder(torch.nn.Module):
    def __init__(self, model_name: str = 'resnet50.tv_in1k', 
                 kwargs: dict = {'features_only': True, 'out_indices': (3,), 'pretrained': True, 'num_classes': 0}, 
                 pool: bool = True):
        super().__init__()
        
        if kwargs.get('pretrained', False) == False:
            # give a warning
            print(f"Warning: {model_name} is used to instantiate a CNN Encoder, but no pretrained weights are loaded. This is expected if you will be loading weights from a checkpoint.")
        self.model = timm.create_model(model_name, **kwargs)
        self.model_name = model_name
        if pool:
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
        else:
            self.pool = None
    
    def forward(self, x):
        out = self.forward_features(x)
        if self.pool:
            out = self.pool(out).squeeze(-1).squeeze(-1)
        return out
    
    def forward_features(self, x):
        out = self.model(x)
        if isinstance(out, list):
            assert len(out) == 1
            out = out[0]
        return out


class TimmViTEncoder(torch.nn.Module):
    def __init__(self, model_name: str = "vit_large_patch16_224", 
                 kwargs: dict = {'dynamic_img_size': True, 'pretrained': True, 'num_classes': 0}):
        super().__init__()
        
        if kwargs.get('pretrained', False):
            # give a warning
            print(f"Warning: {model_name} is used to instantiate a Timm ViT Encoder, but no pretrained weights are loaded. This is expected if you will be loading weights from a checkpoint.")
        self.model = timm.create_model(model_name, **kwargs)
        self.model_name = model_name
    
    def forward(self, x):
        out = self.model(x)
        return out
    
    def forward_features(self, x):
        out = self.model.forward_features(x)
        return out


class HFViTEncoder(torch.nn.Module):
    def __init__(self, model_name: str = "owkin/phikon", 
                 kwargs: dict = {'add_pooling_layer': False}):
        super().__init__()
        
        self.model = ViTModel.from_pretrained(model_name, **kwargs)
        self.model_name = model_name
    
    def forward(self, x):
        out = self.forward_features(x)
        out = out.last_hidden_state[:, 0, :]
        return out
    
    def forward_features(self, x):
        out = self.model(pixel_values=x)
        return out
    

class CLIPVisionModelPostProcessor(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out.pooler_output


class DenseNetBackbone(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.classifier = torch.nn.Identity()
        self.pool = torch.nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        x = self.model.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x


class GigapathSlide(torch.nn.Module):
    def __init__(self, checkpoint_path: str):
        super().__init__()

        # import gigapath.slide_encoder as slide_encoder
        from .gigapath_slide_encoder import create_model

        self.tile_encoder = timm.create_model(model_name='vit_giant_patch14_dinov2', 
                **{'img_size': 224, 'in_chans': 3, 
                'patch_size': 16, 'embed_dim': 1536, 
                'depth': 40, 'num_heads': 24, 'init_values': 1e-05, 
                'mlp_ratio': 5.33334, 'num_classes': 0})
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        self.tile_encoder.load_state_dict(state_dict, strict=True)

        self.slide_model = create_model("hf_hub:prov-gigapath/prov-gigapath", "gigapath_slide_enc12l768d", 1536)

        self.tile_encoder.eval()
        self.slide_model.eval()

    def forward(self, x, coords):
        tile_embed = self.tile_encoder(x)
        _, output = self.slide_model(tile_embed.unsqueeze(0), coords.unsqueeze(0))
        return output[0][0, 1:]
