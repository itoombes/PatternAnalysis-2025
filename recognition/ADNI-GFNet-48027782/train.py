import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
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
N_EPOCHS = 150 
# Interval between evaluations
EVAL_INTERVAL = 5

# Model hyperparameters -- based on GFNet-ti
# Embedded dimension
EMBEDDED_DIM = 256 
# Ratio of embedded dimension to multi-layer perceptron
MLP_RATIO = 4
# Number of model blocks
DEPTH = 12
# Path dropout rate; increases by this increment after each block
DROP_PATH_RATE = 0.2

# Whether to use a LR scheduler
# Currently setup to be CosineAnnealingLr
USE_SCHEDULER = False

# Seed to use for validation split
VALIDATION_SEED = 42

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
    # Load into a scheduler
    if USE_SCHEDULER:
        scheduler = CosineAnnealingLR(optimizer = optimiser, T_max = 40,
                                      eta_min = 0.0005) 

    # Load training and validation data
    validation_loader, training_loader = (
        dataset.get_validation_and_training_dataloaders(128, 128, 0.2,
                                                        seed = VALIDATION_SEED))

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
        if ((e + 1) % EVAL_INTERVAL == 0):
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
        
        # Step the learning rate
        if USE_SCHEDULER:
            scheduler.step()

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
    model.load_state_dict(torch.load(MODEL_SAVE_LOCATION, weights_only = True))
    end_time = time.time()
    print(f'Done! ({end_time - start_time:.2f}s)')

    start_time = time.time()
    print('Loading data... ', end='', flush=True)
    # Load testing data
    test_loader = dataset.get_test_dataloader(batch_size = 128)
    # Load training and validation data
    validation_loader, training_loader = (
        dataset.get_validation_and_training_dataloaders(128, 128, 0.2,
                                                        seed = VALIDATION_SEED))
    end_time = time.time()
    print(f'Done! ({end_time - start_time:.2f}s)')

    # Run over the data
    model.eval()
    # Load the test results into a DataFrame 
    test_results = pd.DataFrame()
    start_time = time.time()
    print('Run over test set... ', end='', flush=True)
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            # Run over data
            images = images.to(device)
            labels = labels.numpy()
            output = model(images)

            # Get predictions from the output
            predicted = torch.max(output, 1)[1].cpu().numpy()
            
            # Append the results to the dataframe
            new_results = np.stack([predicted, labels], axis=1)
            test_results = pd.concat([test_results,
                    pd.DataFrame(new_results, columns=['Pred', 'True'])], 
                                 ignore_index = True)
    end_time = time.time()
    print(f'Done! ({end_time - start_time:.2f}s)')

    # Load the validation results into a DataFrame 
    validation_results = pd.DataFrame()
    start_time = time.time()
    print('Run over validation set... ', end='', flush=True)
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(validation_loader):
            # Run over data
            images = images.to(device)
            labels = labels.numpy()
            output = model(images)

            # Get predictions from the output
            predicted = torch.max(output, 1)[1].cpu().numpy()
            
            # Append the results to the dataframe
            new_results = np.stack([predicted, labels], axis=1)
            validation_results = pd.concat([validation_results,
                    pd.DataFrame(new_results, columns=['Pred', 'True'])], 
                                 ignore_index = True)
    end_time = time.time()
    print(f'Done! ({end_time - start_time:.2f}s)')  
    
    # Load the training results into a DataFrame 
    training_results = pd.DataFrame()
    start_time = time.time()
    print('Run over training set... ', end='', flush=True)
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(training_loader):
            # Run over data
            images = images.to(device)
            labels = labels.numpy()
            output = model(images)

            # Get predictions from the output
            predicted = torch.max(output, 1)[1].cpu().numpy()
            
            # Append the results to the dataframe
            new_results = np.stack([predicted, labels], axis=1)
            training_results = pd.concat([training_results,
                    pd.DataFrame(new_results, columns=['Pred', 'True'])], 
                                 ignore_index = True)
    end_time = time.time()
    print(f'Done! ({end_time - start_time:.2f}s)')

    # Print accuracy results

    validation_correct = len(
        validation_results[validation_results['Pred'] == validation_results['True']])
    print(f'Validation Accuracy: {(validation_correct / len(validation_results))*100:.2f}%')

    training_correct = len(
        training_results[training_results['Pred'] == training_results['True']])
    print(f'Training Accuracy: {(training_correct / len(training_results))*100:.2f}%')

    test_correct = len(
        test_results[test_results['Pred'] == test_results['True']])
    print(f'Test Accuracy: {(test_correct / len(test_results))*100:.2f}%')

def visualise():
    # Load the statistics
    loss_over_time = pickle.load(open(LOSS_OVER_TIME_SAVE, 'rb'))
    validation_loss = pickle.load(open(VALIDATION_LOSS_SAVE, 'rb'))

    import matplotlib.pyplot as plt
    plt.plot(range(0, len(loss_over_time)), loss_over_time, 'r.-', label='Training loss')
    plt.plot(validation_loss.keys(), validation_loss.values(), 'b.-', label='Validation loss')
    plt.title('Training and validation loss over epochs')
    plt.xlabel('Epoch number (0-indexed)')
    plt.ylabel('Cross-validation loss')
    plt.show()

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
        visualise()