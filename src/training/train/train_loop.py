import torch
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

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

        # Setup the SummaryWriter to use with tensorboard
        self.writer = SummaryWriter()

    def train(self):
        
        # List to track the mean of the losses over each epoch
        train_mean_losses = []
        validation_mean_losses = []

        # List the minimum squared error on the validation set over the epochs
        validation_mse = []

        # Enable gradient tracking
        with torch.enable_grad():
            
            # Train for the desired number of epochs
            for epoch in self.epochs:

                # Train loss
                loss = []

                # Set the model to be in training mode
                self.model.train()

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

                    # Add the result to the tensorboard
                    self.writer.add_scalar("Loss/train", loss, epoch)

                    # Backward pass
                    loss.backward()

                    # Update the parameters
                    self.optimizer.step()

                # Set the model to be in evaluation mode
                self.model.eval()

                # Compute the MSE and validation loss on the validation set
                mse, loss_val = self.validate()

                # Save the loss and mse on the validation set for plotting later on
                validation_mean_losses.append(loss_val)
                validation_mse.append(mse)

                # Save the current model parameters
                

        # Flush the writer and close it
        self.writer.flush()
        self.writer.close()

    def evaluate(self, dataloader, compute_loss=False):
        """
        Evaluate the performance of the network on a given dataset

        Args:
            dataloader (Dataloader): A dataloader (i.e. training or validation dataloaders)
            compute_loss (bool): Whether to compute the loss or not
        Returns:
            float: Mean square error over the predictions and the expected output
        """
        
        # Set the model in the evaluation mode
        self.model.eval()

        y_pred = []
        y_true = []

        loss = []

        # Disable gradient tracking on this section
        with torch.no_grad():

            for x, y in dataloader:
                
                # Predict the output
                y_hat = self.model(x)

                # Compute the loss over the validation set
                if compute_loss:
                    loss.append(self.model.compute_loss(x, y, y_hat))
                
                # Add the prediction to the vector
                y_pred += [y_hat]
                y_true += [y]

        # Create the torch tensors from the lists (and make sure they are in the right device)
        y_pred = torch.tensor(y_pred).to(self.model.device)
        y_true = torch.tensor(y_true).to(self.model.device)

        # Compute the MSE and return its value, along with the loss (if requested)
        return mse_loss(y_true, y_pred), torch.tensor(loss).mean().item()

    def validate(self):
        """
        Compute the MSE over the validation set
        """
        return self.evaluate(self.validation_dataloader, compute_loss=True)

    def test(self):
        """
        Compute the MSE over the test set
        """
        return self.evaluate(self.test_dataloader, compute_loss=False)
