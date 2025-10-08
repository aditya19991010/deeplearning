#  Checklist for reading an article
1. Problem statement
2. Dataset
   3. add sample shape and feature
4. Input and output
5. Model architecture and hyperparams
6. Result

-----------
## What to look for in the article 
- Novelty 
- Result comparison with state of the art
- key contribution that advances the knowledge


## Data processing 
- Data splitting
  - Split the data
  - constructs a tensor using train-test-val data
- Data loading
  - Create Tensor Dataframe using TensorDataframe
  - Set batch size 
  - Load Dataframe using Dataloader

## Set up model architecture
 - Define class of the model
   - Define series of networks
   - Define forward pass

## Training set up
 - Set hyperparameters
   - learning rate
   - batch size
   - epoch
 - Define loss function
 - Define optimizer


## Train loop
   1. Set model to training
   2. iterate through each batch from the dataloader
   - pred the target 
   - calculate loss
   - initiate optimize with zero grad
   - backprop
   - optimize the weight
   - calculate total loss
   - get average loss

## Evaluation loop
 - Set model for evaluation
 - iterate though each batch of test data
 - pred
 - calculate loss
 - get average loss

## Iterate the train and eval loop for n epochs time
