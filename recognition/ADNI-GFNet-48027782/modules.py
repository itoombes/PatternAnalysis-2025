import torch
from torch import tensor
from torch.fft import rfft2, irfft2
import torch.nn as nn
from timm.layers.drop import DropPath
import math

'''
REFERENCE: GFNet/gfnet.py

Y. Rao, W. Zhau, Z. Zhu, Z. Zhou, and J. Lu
Global Filter Networks for Image Classification, 2021
arXiv: https://arxiv.org/abs/2107.00645
Github:  https://github.com/raoyongming/GFNet
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
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02) 
    
    def forward(self, x, spatial_size=None):
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
    Directly copied from GFNet Git
    '''
    def __init__(self, in_features, hidden_features=None, out_features=None):
        pass