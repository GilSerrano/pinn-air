import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import AdamW
from os.path import join as pjoin
from torch.utils.tensorboard import SummaryWriter

class TrainLoop2(object):
    """
    A trainloop object that handles the training of the model
    """

    def __init__(self, args, model, train_dataloader, validation_dataloader):

        # Save the model and training data, validation and test data
        self.model = model
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader

        self.teacher_ratio = 0.5
        self.training_mode = "teacher"

        # Setup the number of epochs for the training
        self.epochs = torch.arange(1, args.num_steps + 1)

        # Setup the Adam optimizer
        self.optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # Setup the SummaryWriter to use with tensorboard
        self.writer = SummaryWriter(log_dir=pjoin(args.save_dir, "tensorboard"))

        # Metrics to save during training as a function of the epochs
        self.train_mean_losses = []
        self.train_mean_individual_losses = []
        self.validation_mean_losses = []

        # Save the index of the best model (the one that has the lowest loss in the validation set)
        self.best_model_idx = 0

        # Get the save directory to store the model over the epochs
        self.save_dir = args.save_dir

    def train(self):
        
        # Reset the list to track the mean of the losses over each epoch
        self.train_mean_losses = []

        # Enable gradient tracking
        with torch.enable_grad():
            
            # Train for the desired number of epochs
            for epoch in self.epochs:
 
                print('Training epoch {}'.format(epoch))

                # Train loss
                train_loss = []
                train_individual_losses = []

                # Set the model to be in training mode
                self.model.train()

                for batch in tqdm(self.train_dataloader, desc="Computing batch"):

                    # Zero the gradients
                    self.optimizer.zero_grad()

                    # Get the input of the network and the expect output from the batch
                    x, y, u = batch

                    # Perform a forward pass
                    y_hat = self.model(x, u, y, training_type=self.training_mode, teacher_ratio=self.teacher_ratio)

                    # Forward pass
                    loss, loss_terms = self.model.compute_loss(x, u, y, y_hat)

                    # Save the loss of the training on this batch
                    train_loss.append(loss)
                    train_individual_losses.append(loss_terms)

                    # Backward pass
                    loss.backward()

                    # Update the parameters
                    self.optimizer.step()

                # Compute the training loss on the epoch
                loss_train = torch.tensor(train_loss).mean().item()

                # Compute the individual loss terms on the epoch
                loss_terms = {}
                for key in train_individual_losses[0].keys():
                    if type(train_individual_losses[0][key]) == dict:
                        loss_terms[key] = {}
                        for subkey in train_individual_losses[0][key].keys():
                            loss_terms[key][subkey] = torch.tensor([train_individual_losses[i][key][subkey] for i in range(len(train_individual_losses))]).mean().item()
                    else:
                        loss_terms[key] = torch.tensor([train_individual_losses[i][key] for i in range(len(train_individual_losses))]).mean().item()

                # Compute the MSE and validation loss on the validation set
                _, loss_val = self.validate()

                # Add the results to the tensorboard
                self.writer.add_scalar("Loss/train", loss_train, epoch)
                self.writer.add_scalar("Loss/validation", loss_val, epoch)

                # Add the individual loss terms to the tensorboard
                for key, value in loss_terms.items():
                    if type(value) == dict:
                        for subkey, subvalue in value.items():
                            self.writer.add_scalar("Loss_terms/{}/{}".format(key, subkey), subvalue, epoch)
                    else:
                        self.writer.add_scalar("Loss_terms/{}".format(key), value, epoch)

                # Save the loss and mse on the validation set for plotting later on
                self.train_mean_losses.append(loss_train)
                self.train_mean_individual_losses.append(loss_terms)
                self.validation_mean_losses.append(loss_val)

                # Save the current model parameters
                self.save_model(epoch, loss_train, 0.0, 0.0)

                # Save the best model parameters
                if epoch == 1 or (epoch > 1 and loss_val < self.validation_mean_losses[self.best_model_idx-2]):
                    self.best_model_idx = epoch

        # Flush the writer and close it
        self.writer.flush()
        self.writer.close()
        
        # Save the training results
        self.save_training_statistics()

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

            for x, y, u in dataloader:
                
                # Predict the output
                y_hat = self.model(x, u, y, training_type=self.training_mode, teacher_ratio=self.teacher_ratio)

                # Compute the loss over the validation set
                if compute_loss:
                    total_loss, _ = self.model.compute_loss(x, u, y, y_hat)
                    loss.append(total_loss)
            
                # Save the predictions and the expected output
                # If the model is a classification model, we only want to save the last output
                y_pred += [torch.flatten(y_hat)]

                y_true += [torch.flatten(y)]

        # Create the torch tensors from the lists (and make sure they are in the right device)
        y_pred = torch.cat(y_pred, 0).to(x.device)
        y_true = torch.cat(y_true, 0).to(x.device)

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
    
    def load_best_model(self):
        """
        Load the best model parameters from the training
        """

        # Load the training statistics
        statiscs_path = pjoin(self.save_dir, "training_statistics.pth.tar")
        training_statistics = torch.load(statiscs_path)
        self.train_mean_losses = training_statistics["train_mean_losses"]
        self.train_mean_individual_losses = training_statistics["train_mean_individual_losses"]
        self.validation_mean_losses = training_statistics["validation_mean_losses"]
        self.validation_mse = training_statistics["validation_mse"]
        self.best_model_idx = training_statistics["best_model_idx"]

        checkpoint_path = pjoin(self.save_dir, 'checkpoint_{:04d}.pth.tar'.format(self.best_model_idx))
        print('Loading checkpoint {}'.format(checkpoint_path))

        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path)

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

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

    def save_training_statistics(self):
        """
        Save the MSE and other training statistics for plotting later on
        """
        
        # Save the training statistics
        torch.save({
            "train_mean_losses": self.train_mean_losses,
            "train_mean_individual_losses": self.train_mean_individual_losses,
#            "validation_mean_losses": self.validation_mean_losses,
#            "validation_mse": self.validation_mse,
            "best_model_idx": self.best_model_idx
        }, pjoin(self.save_dir, "training_statistics.pth.tar"))