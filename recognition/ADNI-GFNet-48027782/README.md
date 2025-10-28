# GFNet Classification of ADNI Data
COMP3710 Project\
Isaac Toombes, 48027782\
Semester 2 2025 

## Global Filter Networks

Global filter networks function by replacing the self-attention layer in vision transformers with Fourier transforms[1].
An example overview this layout is as follows [1]:

![Layout of GFNet](figures/GFNet_layout.png)

In essence, the model works by replacing the self-attention layer with a Fourier transform, learning filters for the transform-space, and then perfoming an inverse Fourier transform to return the data to its original domain [1].
This data is then fed through a multi-layer perceptron.


## Pre-processing
Each image in the ADNI dataset individually contains a large proportion of blank space around the area of interest, indicating that cropping may be useful.
However, by combining the images together (taking the maximum value across every image for every pixel), it becomes apparent that the data is spread in such a way to make this infeasible:

![Visualisation of image space used by ADNI data](figures/adni_imgspaceused.png)

## Requirements 

### Python & CUDA versions 
 - Python 3.13.9
 - CUDA 13.0

### Packages:
 - contourpy         1.3.3
 - cycler            0.12.1
 - filelock          3.19.1
 - fonttools         4.60.1
 - fsspec            2025.9.0
 - Jinja2            3.1.6
 - kiwisolver        1.4.9
 - MarkupSafe        2.1.5
 - matplotlib        3.10.7
 - mpmath            1.3.0
 - networkx          3.5
 - numpy             2.3.4
 - packaging         25.0
 - pandas            2.3.3
 - pillow            12.0.0
 - pip               25.3
 - pyparsing         3.2.5
 - python-dateutil   2.9.0.post0
 - pytz              2025.2
 - setuptools        70.2.0
 - six               1.17.0
 - sympy             1.14.0
 - torch             2.9.0+cu130
 - torchvision       0.24.0+cu130
 - typing_extensions 4.15.0
 - tzdata            2025.2

A `pip`-readable list is in the [`requirements.txt`](./requirements.txt) file.

### References
[1] Y. Rao, W. Zhau, Z. Zhu, J. Zhou, and J. Lu, "Global Filter Networks for Image Classification," 2021, arXiV: 2107.00645. [Online] [https://arxiv.org/abs/2107.00645](https://arxiv.org/abs/2107.00645)
