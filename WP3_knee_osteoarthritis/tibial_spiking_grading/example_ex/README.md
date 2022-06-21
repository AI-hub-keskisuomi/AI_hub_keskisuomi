# example

params.json specifies parameter space to search in hyperparameter tuning.
- num_epochs: max epochs to train a model with each hyperparameter combination
- learning_rate: adjusts the magnitude for updating the model weights
- step_size: interval for controlling adjusting learning-rate
- model: used to specify and document which predefined model was used
- gamma: factor for decaying learning-rate (step-size defines the interval)
- img_size: used to specify and document the input size for the model
- batch_size: specifies number of samples to use in computing the loss and the gradient