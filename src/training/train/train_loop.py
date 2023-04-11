from torch.optim import AdamW

class TrainLoop(object):

    def __init__(self, args, model, data):

        # Save the model and training data
        self.model = model
        self.data = data

        # Setup the Adam optimizer
        self.optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # Setup the learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay)

        # Setup the loss function
