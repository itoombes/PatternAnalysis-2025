import torch
from dataset import Adni
from modules import GFNet

# GFNet S, adapted for ADNI
model = GFNet(
    img_size=(240,256),
    patch_size=(16,16),
    in_chans = 1,
    embedded_dim=384,
    depth=19,
    mlp_ratio=4,
    drop_path_rate=0.15
)

if __name__ == "__main__":
    print(model.state_dict())