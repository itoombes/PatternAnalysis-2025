import torch
from torch import tensor
from torch.fft import rfft2, irfft2
import torch.nn as nn
from timm.layers.drop import DropPath
import math

'''
Many of the classes within have essentially been transferred from GFNet.py,
with some additional documentation.

Y. Rao, W. Zhau, Z. Zhu, Z. Zhou, and J. Lu
Global Filter Networks for Image Classification, 2021
arXiv: https://arxiv.org/abs/2107.00645
Github: https://github.com/raoyongming/GFNet
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
        if type(img_size) is int:
            self.img_size = (img_size, img_size)
        else:
            self.img_size = img_size
        
        if type(patch_size) is int:
            self.patch_size = (patch_size, patch_size)
        else:
            self.patch_size = patch_size
        
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