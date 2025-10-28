# GFNet Classification of ADNI Data
COMP3710 Project\
Isaac Toombes, 48027782\
Semester 2 2025 

## Alzheimer's Disease Neuroimaging Initiative (ADNI)
The ADNI dataset [1] was used as-provided on the UQ Rangpur cluster computer, in the `/home/groups/comp3710/ADNI` folder.
The data consists of 35,200 256×240 8-bit grayscale MRI scans, in `.jpeg` format.
The data is split into two groups - `test` and `train`, and then further partitioned into `AD` (Alzheimer's disease) and `NC` (cognitive normal) subsets.
Each file is labelled in the format `<patient id>_<image number>.jpeg`, where a patient can appear multiple times in the same dataset.

## Global Filter Network (GFNet) Architecture

Global filter networks function by replacing the self-attention layer in vision transformers with Fourier transforms[2].
More explicitly, the GFNet model defines a 'Global Filter', which takes its input data, transforms it into the frequency domain, multiplies it with learnable filters, inverse Fourier transforms the data back to the original space.
The outputs of the Global Filter layers are then then run through Multi-Layer Perceptrons (MLPs) [2]. 
A visual overview of this architecture is as follows [2]:

![Layout of GFNet](figures/GFNet_layout.png)

Note that in addition to the Global Filter and MLP blocks, there is a patch-embedding layer, several normalisation layers, an average pooling layer, and a linear layer.
The patch-embedding layer works as it does in a vision transformer; it breaks the input image into distinct 'patches', before flattening and projecting them onto the embedding dimension.

The purpose of this project is to adapt the GFNet architecture to classify observations in the ADNI dataset into either 'cognitive normal' or 'Alzheimer's detected' classes, with a desired minimum level of accuracy of $80\%$.

## Software Requirements 

### Python & CUDA versions 
 - Python 3.13.9
 - CUDA 13.0

### Packages:
A `pip`-readable list is in the [`requirements.txt`](./requirements.txt) file.
- anyio             4.11.0
- certifi           2025.10.5
- click             8.3.0
- colorama          0.4.6
- contourpy         1.3.3
- cycler            0.12.1
- filelock          3.19.1
- fonttools         4.60.1
- fsspec            2025.9.0
- h11               0.16.0
- hf-xet            1.2.0
- httpcore          1.0.9
- httpx             0.28.1
- huggingface-hub   1.0.0
- idna              3.11
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
- PyYAML            6.0.3
- safetensors       0.6.2
- setuptools        70.2.0
- sh ellingham       1.5.4
- six               1.17.0
- sniffio           1.3.1
- sympy             1.14.0
- timm              1.0.21
- torch             2.9.0+cu130
- torchvision       0.24.0+cu130
- tqdm              4.67.1
- typer-slim        0.20.0
- typing_extensions 4.15.0
- tzdata            2025.2
## Data Pre-processing
Each image in the ADNI dataset individually contains a large proportion of blank space around the area of interest, indicating that cropping may be useful.
However, by combining the images together (taking the maximum value across every image for every pixel), it becomes apparent that the data is spread in such a way to make this infeasible:

![Visualisation of image space used by ADNI data](figures/adni_imgspaceused.png)

It was decided against processing the 8-bit values in the images (for example, by dividing them by 255 to change them to the range of 0 and 1), because of the use of discrete Fourier transforms.

## Creation of a validation data set
It was decided to select a subset of the training set to create a validation dataset, and provide an indication of the model's performance over training iterations.
To avoid contamination of data, the following process was used:
 - The training `AD` and `NC` datasets were further partitioned based on patient ID
 - 20% of the patient ID's in the `AD` dataset, and 20% of patient ID's in the `NC` dataset, were randomly selected
 - The observations associated with these ID`s were used to create the validation dataset, and removed from the training dataset

## References
[1] Alzheimer's Disease Neuroimaging Initiative, "ADNI | Alzheimer's disease neuroimaging initiative," 2025. [Online] [https://adni.loni.usc.edu/](https://adni.loni.usc.edu/)

[2] Y. Rao, W. Zhau, Z. Zhu, J. Zhou, and J. Lu, "Global Filter Networks for Image Classification," 2021, arXiV: 2107.00645. [Online] [https://arxiv.org/abs/2107.00645](https://arxiv.org/abs/2107.00645)