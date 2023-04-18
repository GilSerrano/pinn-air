import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import AdamW
from os.path import join as pjoin
from torch.utils.tensorboard import SummaryWriter

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
        self.writer = SummaryWriter(log_dir=pjoin(args.save_dir, "tensorboard"))

        # Metrics to save during training as a function of the epochs
        self.train_mean_losses = []
        self.validation_mean_losses = []
        self.validation_mse = []

        # Get the save directory to store the model over the epochs
        self.save_dir = args.save_dir

    def train(self):
        
        # Reset the list to track the mean of the losses over each epoch
        self.train_mean_losses = []
        self.validation_mean_losses = []

        # Reset the list the minimum squared error on the validation set over the epochs
        self.validation_mse = []

        # Enable gradient tracking
        with torch.enable_grad():
            
            # Train for the desired number of epochs
            for epoch in self.epochs:

                print('Training epoch {}'.format(epoch))

                # Train loss
                train_loss = []

                # Set the model to be in training mode
                self.model.train()

                for batch in tqdm(self.train_dataloader, desc="Computing batch"):

                    # Zero the gradients
                    self.optimizer.zero_grad()

                    # Get the input of the network and the expect output from the batch
                    x, y = batch

                    # Perform a forward pass
                    y_hat = self.model(x)

                    # Forward pass
                    loss = self.model.compute_loss(x, y, y_hat)

                    # Save the loss of the training on this batch
                    train_loss.append(loss)

                    # Backward pass
                    loss.backward()

                    # Update the parameters
                    self.optimizer.step()

                # Compute the training loss on the epoch
                loss_train = torch.tensor(train_loss).mean().item()

                # Compute the MSE and validation loss on the validation set
                mse, loss_val = self.validate()
                
                # Add the results to the tensorboard
                self.writer.add_scalar("Loss/train", loss_train, epoch)
                self.writer.add_scalar("Loss/validation", loss_val, epoch)
                self.writer.add_scalar("MSE/validation", mse, epoch)

                # Save the loss and mse on the validation set for plotting later on
                self.train_mean_losses.append(loss_train)
                self.validation_mean_losses.append(loss_val)
                self.validation_mse.append(mse)

                # Save the current model parameters
                self.save_model(epoch, loss_train, 0.0, 0.0) #loss_val, mse)

        # Flush the writer and close it
        self.writer.flush()
        self.writer.close()

        # Set the model to be in evaluation mode again
        self.model.eval()

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
        y_pred = torch.cat(y_pred, 0).to(self.model.device)
        y_true = torch.cat(y_true, 0).to(self.model.device)

        # Compute the MSE and return its value, along with the loss (if requested)
        return F.mse_loss(y_pred, y_true), torch.tensor(loss).mean().item()

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

    def save_model(self, epoch, train_loss, val_loss, val_mse):
        """
        Save the current model for later 

        Args:
            epoch (int): The current epoch number
            train_loss (float): The training loss on the epoch
            val_loss (float): The validation loss on the epoch
            val_mse (float): The validation loss on the epoch
        """

        checkpoint_path = pjoin(self.save_dir, 'checkpoint_{:04d}.pth.tar'.format(epoch))
        print('Saving checkpoint {}'.format(checkpoint_path))

        # Save the current model
        torch.save({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "validation_mse": val_mse,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }, checkpoint_path)