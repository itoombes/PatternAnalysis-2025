import torch
import torch.nn as nn
from dataset import Adni
from modules import GFNet
import time
import pickle

# Root of ADNI data
ADNI_ROOT = 'C:/Users/itoom/COMP3710/ADNI/'
# Location to save validation data & model parameters
DATA_SAVE_ROOT = 'D:/University/PatternAnalysis-2025/'
# Name of model save file
MODEL_SAVE_NAME = 'gfnet_weights.pt2'
# Name of stat save file
STAT_SAVE_NAME = 'stats.pkl'

# Number of training epochs
N_EPOCHS = 30 
# Interval between evaluations
EVAL_INT = 3

# Model hyperparameters
# Embedded dimension
EMBEDDED_DIM = 384
# Ratio of embedded dimension to multi-layer perceptron
MLP_RATIO = 4
# Number of model blocks
DEPTH = 19
# Path dropout rate; increases by this increment after each block
DROP_PATH_RATE = 0.15


def train_model():
    # Use CUDA if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {torch.cuda.get_device_name(device)}')

    # Load a GFNet model with the specified hyperparameters
    model = GFNet(
        img_size=(240, 256),
        patch_size=(16, 16),
        in_chans = 1,
        embedded_dim=EMBEDDED_DIM,
        depth=DEPTH,
        mlp_ratio=MLP_RATIO,
        drop_path_rate=DROP_PATH_RATE
    ).to(device)

    # Using cross entropy as validation criterion
    criterion = nn.CrossEntropyLoss()
    
    # Using GFNet default optimiser - - see main_gfnet.py in GFNet GitHub
    # AdamW with epsilon 1e-8, weight decay 0.05
    weight_decay = [p for p in model.parameters() if p is not (model.pos_embed)]
    no_weight_decay = model.pos_embed

    params = [{'params': weight_decay, 'weight_decay': 0.05},
              {'params': no_weight_decay}]

    optimiser = torch.optim.AdamW(params, eps=1e-8)

    # Load training and validation datasets
    adni = Adni(ADNI_ROOT, validation_split=0.15, validation_split_base=42)
    train_loader, validate_loader = adni.get_training_and_validation_dataloaders(128, 128)

    # Train the model
    for e in range(N_EPOCHS):
        start_time = time.time()
        model.train()
        total_loss = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images.to(device)
            labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss = loss

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        end_time = time.time()
        avg_loss = total_loss / len(train_loader.dataset)

        print(f'Epoch {e}: {end_time - start_time:.2f}, {avg_loss}')


if __name__ == "__main__":
    train_model()