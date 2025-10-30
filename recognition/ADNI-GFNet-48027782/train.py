import torch
import torch.nn as nn
import dataset
from modules import GFNet
import time
import pickle
from sys import argv
import numpy as np
import pandas as pd

# File save locations
MODEL_SAVE_LOCATION = 'model.pt'
LOSS_OVER_TIME_SAVE = 'loss.pkl'
VALIDATION_LOSS_SAVE = 'validation_scores.pkl'

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
    validation_loss_over_time = dict()

    # Track model performance across validations
    best_model_loss = -1

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

        # Print stats and compute average loss
        end_time = time.time()
        avg_loss = total_loss / len(training_loader)
        print(f'{end_time - start_time:.2f}s, {avg_loss}')
        # Save average loss
        loss_over_time.append(avg_loss)
    
        # Evaluate during every interval
        if ((e + 1) % EVAL_INT == 0):
            print('Validating... ', end='', flush=True)
            start_time = time.time()
            
            model.eval()
            with torch.no_grad():
                total_loss = 0
                for batch_idx, (images, labels) in enumerate(validation_loader):
                    images = images.to(device)
                    labels = labels.to(device)

                    # Forward pass
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    total_loss += loss.item()
                
            # Print stats and compute average loss
            end_time = time.time()
            avg_loss = total_loss / len(validation_loader)
            print(f'{end_time - start_time:.2f}s, {avg_loss}')
            # Save average loss
            validation_loss_over_time[e] = avg_loss

            # If have better model, save it
            if (avg_loss < best_model_loss or best_model_loss < 0):
                print('\tNew best model!')
                best_model_loss = avg_loss
                # Save the model
                torch.save(model.state_dict(), MODEL_SAVE_LOCATION)
                # Save statistics, in case program is distrupted
                pickle.dump(loss_over_time, open(LOSS_OVER_TIME_SAVE, 'wb'))
                pickle.dump(validation_loss_over_time,
                            open(VALIDATION_LOSS_SAVE, 'wb'))

    # Save the statistics
    pickle.dump(loss_over_time, open(LOSS_OVER_TIME_SAVE, 'wb'))
    pickle.dump(validation_loss_over_time, open(VALIDATION_LOSS_SAVE, 'wb'))

def evaluate():
    # Use CUDA if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {torch.cuda.get_device_name(device)}')

    # Load model with specified hyperparameters
    start_time = time.time()
    print('Loading model... ', end='', flush=True)
    model = GFNet(
        img_size = (240, 256),
        patch_size = (16, 16),
        in_chans = 1,
        embedded_dim = EMBEDDED_DIM,
        depth = DEPTH,
        mlp_ratio = MLP_RATIO,
        drop_path_rate = DROP_PATH_RATE
    ).to(device)

    # Load the model
    model.load_state_dict(torch.load(MODEL_SAVE_LOCATION, weights_only=True))
    end_time = time.time()
    print(f'Done! ({end_time - start_time:.2f}s)')

    # Load the data
    start_time = time.time()
    print('Loading data... ', end='', flush=True)
    dataloader = dataset.get_test_dataloader(batch_size=128)
    end_time = time.time()
    print(f'Done! ({end_time - start_time:.2f}s)')

    # Run over training data and save into a pd.DataFrame
    results = pd.DataFrame()
    start_time = time.time()
    print('Run over test set... ', end='', flush=True)
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            # Run over data
            images = images.to(device)
            labels = labels.numpy()
            output = model(images)

            # Get predictions from the output
            predicted = torch.max(output, 1)[1].cpu().numpy()
            
            # Append the results to the dataframe
            new_results = np.stack([predicted, labels], axis=1)
            results = pd.concat([results,
                                 pd.DataFrame(new_results,
                                              columns=['Pred', 'True'])], 
                                ignore_index = True)
    end_time = time.time()
    print(f'Done! ({end_time - start_time:.2f}s)')

    num_correct = len(results[results['Pred'] == results['True']])
    print(f'Accuracy: {(num_correct / len(results))*100:.2f}%')

if __name__ == "__main__":
    '''
    If no arguments, run model training
    If argument is 'eval', run model evaluation
    If argument is 'vis', visualise the training & validation loss over time
    '''
    if len(argv) == 1:
        print('Training')
        train_model()
        exit()
    
    if argv[1] == 'eval':
        print('Evaluation')
        evaluate()
        exit()
    
    if argv[1] == 'vis':
        print('Visualisation')
        print(pickle.load(open(LOSS_OVER_TIME_SAVE, "rb")))
        print(pickle.load(open(VALIDATION_LOSS_SAVE, 'rb')))