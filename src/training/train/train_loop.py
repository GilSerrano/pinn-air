import tqdm
import torch
from torch.optim import AdamW

class TrainLoop(object):
    """
    A trainloop object that handles the training of the model
    """

    def __init__(self, args, model, train_dataloader, validation_dataloader=None):

        # Save the model and training data
        self.model = model
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader

        # Setup the number of epochs for the training
        self.epochs = args.num_steps

        # Setup the Adam optimizer
        self.optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def train(self):

        # Set the model to be in training mode
        self.model.train()

        # Enable gradient tracking
        with torch.enable_grad():

            for i, batch in tqdm(enumerate(self.data_iterator), desc="Iterating over batches"):

                # Get the input of the network and the expect output from the batch
                x, y = batch

                # Perform a forward pass
                y_hat = self.model(x)

                # Zero the gradients
                self.optimizer.zero_grad()

                # Forward pass
                loss = self.model.compute_loss(x, y, y_hat)

                # Backward pass
                loss.backward()

                # Update the parameters
                self.optimizer.step()

        # Set the model to be in evaluation mode
        self.model.eval()

    def test(self):
        pass
