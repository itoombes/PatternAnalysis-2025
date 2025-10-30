import torch
import torch.nn as nn
import dataset
from modules import GFNet
import time
import pickle
from sys import argv

# File save locations
LOSS_OVER_TIME_SAVE = "loss.pkl"
VALIDATION_SCORES_SAVE = 'validation_scores.pkl'

# Number of training epochs
N_EPOCHS = 6 
# Interval between evaluations
EVAL_INT = 2

# Model hyperparameters
# Embedded dimension
EMBEDDED_DIM = 100 
# Ratio of embedded dimension to multi-layer perceptron
MLP_RATIO = 2
# Number of model blocks
DEPTH = 5
# Path dropout rate; increases by this increment after each block
DROP_PATH_RATE = 0.01


def train_model():
    # Use CUDA if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {torch.cuda.get_device_name(device)}')

    # Load a GFNet model with the specified hyperparameters
    model = GFNet(
        img_size = (240, 256),
        patch_size = (16, 16),
        in_chans = 1,
        embedded_dim = EMBEDDED_DIM,
        depth = DEPTH,
        mlp_ratio = MLP_RATIO,
        drop_path_rate = DROP_PATH_RATE
    ).to(device)

    # Using cross entropy as validation criterion
    criterion = nn.CrossEntropyLoss()
    
    # Using GFNet default optimiser - - see main_gfnet.py in GFNet GitHub
    # AdamW with epsilon 1e-8, weight decay 0.05, no weight decay on position 
    # embedding parameters
    weight_decay = [p for p in model.parameters() if p is not (model.pos_embed)]
    no_weight_decay = model.pos_embed

    params = [{'params': weight_decay, 'weight_decay': 0.05},
              {'params': no_weight_decay}]

    optimiser = torch.optim.AdamW(params, eps=1e-8)

    # Load training and validation datasets
    validation_loader, training_loader = dataset.get_validation_and_training_dataloaders(128, 128, 0.2, seed=42)

    # Used to save statistics
    loss_over_time = list()
    validation_accuracy = dict()

    # Train the model
    for e in range(N_EPOCHS):
        print(f'Epoch {e}: ', end='', flush=True)
        start_time = time.time()
        model.train()
        total_loss = 0

        for batch_idx, (images, labels) in enumerate(training_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Optimise
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        # Print stats, and compute average loss
        end_time = time.time()
        avg_loss = total_loss / len(training_loader)
        print(f'{end_time - start_time:.2f}s, {avg_loss}')
        # Save average loss
        loss_over_time.append(avg_loss)
    
    # Save the statistics
    pickle.dump(loss_over_time, open(LOSS_OVER_TIME_SAVE, 'wb'))

if __name__ == "__main__":
    if len(argv) == 1:
        print('Training')
        train_model()
        exit()
    
    if argv[1] == 'eval':
        print('Evaluation')
    
    if argv[1] == 'vis':
        print('Visualisation')
        print(pickle.load(open(LOSS_OVER_TIME_SAVE, "rb")))