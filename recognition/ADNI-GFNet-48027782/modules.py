import torch
from torch.fft import rfft2, irfft2
import torch.nn as nn
from timm.layers.drop import DropPath
from timm.layers.helpers import to_2tuple
from torch.nn.init import trunc_normal_
from functools import partial
import math

'''
Many of the classes within have been transferred from gfnet.py,
with some additional documentation.

Y. Rao, W. Zhao, Z. Zhu, Z. Zhou, and J. Lu
Global Filter Networks for Image Classification, 2021
arXiv: https://arxiv.org/abs/2107.00645
GitHub: https://github.com/raoyongming/GFNet
'''


class GlobalFilter(nn.Module):
    '''
    Filter layer for GFNet

    Takes an input, performs a discrete Fourier transform, multiplies it by
    a learnable weight, and then performs an inverse Fourier transform back to
    the original dimensions
    '''
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        # complex_weight is the learnable filter
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2,
                dtype=torch.float32) * 0.02) 
    
    def forward(self, x, spatial_size=None):
        # Pretty sure N is embedded dimension, C is number of input channels
        batch, N, C = x.shape
        
        # Determine size to use for Fourier transform
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size

        # Change input shape & type to prepare for 
        x = x.view(batch, a, b, C)
        x = x.to(torch.float32)

        # Transform to frequency space
        x = rfft2(x, dim=(1, 2), norm='ortho')
        # Perform element-wise transform of input through filters
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        # Move back to original feature space
        x = irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')

        # Return x back to original shape
        x = x.reshape(batch, N, C)

        return x
    
class MultiLayerPerceptron(nn.Module):
    '''
    Multilayer Perceptron
    
    Two fully connected layers, with an activation function between them
    Uses the same dropout on both layers.
    Defaults to Gaussian Error Linear Unit.
    '''
    def __init__(self, in_features, hidden_features = None, out_features = None,
                 act_layer = nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class GFNetBlock(nn.ModuleDict):
    '''
    Main processing block of GFNet
    Normalise --> global filter --> normalise --> MLP
    '''
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., 
            act_layer=nn.GELU, norm_layer=nn.LayerNorm, h=14, w=8):
        '''
        dim: Input dimension
        mlp_ratio: Size of MLP output compared to input dimension
        drop: Dropout amount for MLP
        drop_path: Dropout amount for TIMM DropPath 
        act_layer: Activation layer for MLP
        norm_layer: Normalisation layer used before and after filter
        h: h parameter passed to MLP 
        w: w parameter passed to MLP
        '''
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = GlobalFilter(dim, h=h, w=w)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MultiLayerPerceptron(in_features=dim,
                hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    
    def forward(self, x):
        y = self.norm1(x)
        y = self.filter(x)
        y = self.norm2(x)
        y = self.mlp(x)
        # Drop-path is a dropout variant which removes paths, not just
        # individual nodes
        y = self.drop_path(x)

        return x + y

class PatchEmbed(nn.Module):
    '''
    Image to patch embedding
    
    Uses convolution to get the patches and embedded dimension, then flattens
    '''
    def __init__(self, img_size = (240, 256), patch_size = (16, 16), in_chans=1, embedded_dim=768):
        '''
        img_size: size of the input image, in either 2tuple or single-side dimension
        patch_size: size of the input image, in either 2tuple or single-side dimension
        in_chans: Number of channels in the input image
        embedded_dim: Number of dimensions to embed the input image into
        '''

        super().__init__()

        # Ensure img_size and patch_size are 2D tuples
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        
        # Determine number of patches based on image and patch size
        self.num_patches = ((self.img_size[0] // self.patch_size[0])
                * (self.img_size[1] * self.img_size[1]))
        
        # Conv. layer used to extract patches & project onto embedded dimension
        self.proj = nn.Conv2d(in_chans, embedded_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        batch, channels, height, width = x.shape

        # Ensure correct dimensions
        assert (height == self.img_size[0] and width == self.img_size[1]), \
            "Input image doesn't match expected size"
        
        # Patch & embed the image
        x = self.proj(x)
        # Flatten into embedded space
        x = self.proj(x).flatten(2).transpose(1, 2)

class GFNet(nn.Module):
    '''
    Lightly modified version of GFNet from GFNet GitHub 
    Note that representation size & uniform_drop have been removed from this
    version
    '''
    def __init__(self, img_size=(240, 256), patch_size=(16, 16), in_chans=1,
                 num_classes=2, embedded_dim=768, depth=12, mlp_ratio=4.,
                 drop_rate=0, drop_path_rate=0., norm_layer=None):
        super().__init__()
        
        self.num_classes = num_classes
        self.embedded_dim = embedded_dim
        # Normalisation layer defaults 1e-6 added to denominator for stabliity
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        else:
            norm_layer = norm_layer
        
        # Patch embedding layer
        self.patch_embedd = PatchEmbed(img_size = img_size,
                patch_size = patch_size, in_chans = 1,
                embedded_dim = embedded_dim)
        
        # Position embedding parameters
        self.pos_embed = nn.Parameter(torch.zeros(1,
                self.patch_embedd.num_patches, embedded_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        h = img_size[0] // patch_size[0]
        w = img_size[1] // patch_size[1]

        # Not using uniform drop
        # Drop rate increases with depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList([
            GFNetBlock(
                dim=embedded_dim, mlp_ratio=mlp_ratio, drop=drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer, h=h, w=w
            ) for i in range(depth)])
        
        self.norm = norm_layer(embedded_dim)

        # Hold-over from representation_size from original version
        self.pre_logits = nn.Identity()

        # Classifier head - coverts embedded dimension to prediction
        self.head = nn.Linear(self.embedded_dim, num_classes)

        # Holdover from dropcls parameter
        self.final_dropout = nn.Identity()

        # Initialise embedded position init via truncated normal distribution
        # Original GFNet used TIMM preview version; now just a part of PyTorch
        trunc_normal_(self.pos_embed, std=0.2)

        # Initialise the linear & layernorm weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        '''
        Initialise linear weights to truncated normal distribution, or their bias
        Initialise layer norms to have 0 bias and weight of 1
        '''
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)