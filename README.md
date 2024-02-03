# Helper Functions Space

This Repository contains functions that I frequently use in my projects. Hence, I decided to write them separately and re-use them as needed

It contains
1. 'train_test_loop.py'
    - this is a class that is used for training neural networks,
    - it tracks training/validation losses (accurancy is applicable) and saves the losses into a npy file (numpy file format) 
    - print training progresses and write training messages to a text file for later reference
    - saves best model state_dict (i.e., lowest validation loss) and last model state_dict (i.e., after each training run)
    - it prints and writes out the hypterparameters such as lr, model architecture, input dimension for future reference
    - in progress: print and save plot of training/validation losses

  2. 'loss_functions.py'
this class provides a simple way to access loss functions from torch.nn, it takes the type of problem (e.g., binary classification) as input and return the appropriate loss function from torch


