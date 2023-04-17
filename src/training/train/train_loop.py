import torch
from tqdm import tqdm
from torch.optim import AdamW

from .losses import mse_loss

class TrainLoop(object):
    """
    A trainloop object that handles the training of the model
    """

    def __init__(self, args, model, train_dataloader, validation_dataloader=None, test_dataloader=None):

        # Save the model and training data, validation and test data
        self.model = model
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.test_dataloader = test_dataloader

        # Setup the number of epochs for the training
        self.epochs = torch.arange(1, args.num_steps + 1)

        # Setup the Adam optimizer
        self.optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def train(self):

        # Set the model to be in training mode
        self.model.train()

        # Enable gradient tracking
        with torch.enable_grad():
            
            # Train for the desired number of epochs
            for epoch in self.epochs:

                print('Training epoch {}'.format(epoch))

                for i, batch in tqdm(enumerate(self.train_dataloader), desc="Computing batch"):

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

    def evaluate(self, dataloader):
        """
        Evaluate the performance of the network on a given dataset

        Args:
            dataloader (Dataloader): A dataloader (i.e. training or validation dataloaders)

        Returns:
            float: Mean square error over the predictions and the expected output
        """
        
        # Set the model in the evaluation mode
        self.model.eval()

        y_pred = []
        y_true = []

        # Disable gradient tracking on this section
        with torch.no_grad():

            for x, y in dataloader:
                
                # Predict the output
                y_hat = self.model(x)
                
                # Add the prediction to the vector
                y_pred += [y_hat]
                y_true += [y]

        # Create the torch tensors from the lists (and make sure they are in the right device)
        y_pred = torch.tensor(y_pred).to(self.model.device)
        y_true = torch.tensor(y_true).to(self.model.device)

        # Compute the MSE and return its value
        return mse_loss(y_true, y_pred)

    def validate(self):
        """
        Compute the MSE over the validation set
        """
        self.evaluate(self.validation_dataloader)

    def test(self):
        """
        Compute the MSE over the test set
        """
        self.evaluate(self.test_dataloader)
