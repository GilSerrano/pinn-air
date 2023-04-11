import tqdm
import torch
from torch.optim import AdamW

class TrainLoop(object):

    def __init__(self, args, model, data_iterator):

        # Save the model and training data
        self.model = model
        self.data_iterator = data_iterator

        # Setup the Adam optimizer
        self.optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # Setup the learning rate sched   

        # Setup the loss function

    def train(self):
        

        with torch.enable_grad():

            for i, batch in tqdm(enumerate(self.data_iterator), desc="Iterating over batches"):

                # Make sure we have everything in the right device
                batch = 

                # Zero the gradients
                self.optimizer.zero_grad()

                # Forward pass
                self.model.compute_loss()


    def test(self):
        pass
