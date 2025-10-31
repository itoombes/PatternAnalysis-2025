import torch
from torchvision.io.image import decode_image
import torchvision.transforms.v2 as v2
from modules import GFNet
from train import EMBEDDED_DIM, DEPTH, MLP_RATIO, DROP_PATH_RATE, MODEL_SAVE_LOCATION
from dataset import TRANSFORM
from sys import argv

def predict(file):
    '''
    Predict Alzheimer's probability based on input image

    file: Image to MRI scan to use. Should be 256x240 grayscale.
    '''
    # Load CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {torch.cuda.get_device_name(device)}')

    # Load the image and apply same transform as dataset
    im = decode_image(file)
    if TRANSFORM:
        im = TRANSFORM(im)
    
    # Make it a 'batch' of 1 and send to device
    im = im.unsqueeze(0).to(device)
    
    # Load the model
    model = GFNet(
        img_size = (240, 256),
        patch_size = (16, 16),
        in_chans = 1,
        embedded_dim = EMBEDDED_DIM,
        depth = DEPTH,
        mlp_ratio = MLP_RATIO,
        drop_path_rate = DROP_PATH_RATE
    ).to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_LOCATION, weights_only = True))

    # Predict the model on the image
    model.eval()
    with torch.no_grad():
        prob_cn = float(model(im)[0].cpu()[0])
        prob_ad = float(model(im)[0].cpu()[1])
    
    out_msg = "Alzheimer's disease" if prob_ad >= 0.5 else "cognitive normal"
    out_val = prob_ad if prob_ad >= 0.5 else prob_cn

    print(f'Predicted probablity of {out_msg} is {out_val * 100:.2f}%')

if __name__ == '__main__':
    # argv[1] should be file location.
    if len(argv) != 2:
        print('Invalid arguments')
    predict(argv[1])
