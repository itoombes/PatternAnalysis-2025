import torch
from torch import tensor
from torch.fft import rfft2, irfft2
import torch.nn as nn

'''
REFERENCES:

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

    Adapted from GFNet Github page
    '''
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02) 
    
    def forward(self, x):
        B, H, W, C = x.shape
        x = rfft2(x, dim=(1, 2), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = irfft2(x, s=(H, W), dim=(1, 2), norm='ortho')
        return x
    
class MultiLayerPerceptron(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        pass