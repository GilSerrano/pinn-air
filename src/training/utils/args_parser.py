from argparse import ArgumentParser

def add_base_options(parser):
    group = parser.add_argument_group('base')
    group.add_argument("--cuda", default=True, type=bool, help="Use cuda device, otherwise use CPU.")
    group.add_argument("--device", default=0, type=int, help="Device id to use.")
    group.add_argument("--seed", default=10, type=int, help="For fixing random seed.")
    group.add_argument("--batch_size", default=4, type=int, help="Batch size during training.")


def add_model_options(parser):
    group = parser.add_argument_group('model')
    #group.add_argument("--arch", default='mlp', choices=['mlp'], type=str, help="Architecture types to try.") 
    #group.add_argument("--layers", default=8, type=int,help="Number of layers.")


def add_data_options(parser):
    group = parser.add_argument_group('dataset')
    group.add_argument("--dataset", default='sim_circles', choices=['sim_circles'], type=str, help="Dataset name (choose from list).")
    group.add_argument("--data_dir", default="", type=str, help="If empty, will use defaults according to the specified dataset.")


def add_training_options(parser):
    group = parser.add_argument_group('training')
    group.add_argument("--save_dir", required=True, type=str, help="Path to save checkpoints and results.")
    group.add_argument("--overwrite", action='store_true', help="If True, will enable to use an already existing save_dir.")
    group.add_argument("--lr", default=1e-4, type=float, help="Learning rate.")
    group.add_argument("--weight_decay", default=0.0, type=float, help="Optimizer weight decay.")
    group.add_argument("--eval_batch_size", default=4, type=int, help="Batch size during evaluation loop.")
    #group.add_argument("--eval_split", default='test', choices=['val', 'test'], type=str, help="Which split to evaluate on during training.")
    #group.add_argument("--eval_during_training", action='store_true', help="If True, will run evaluation during training.")
    #group.add_argument("--log_interval", default=1_000, type=int, help="Log losses each N steps")
    #group.add_argument("--save_interval", default=50_000, type=int, help="Save checkpoints and run evaluation each N steps")
    group.add_argument("--num_steps", default=600_000, type=int, help="Training will stop after the specified number of steps.")
    #group.add_argument("--resume_checkpoint", default="", type=str, help="If not empty, will start from the specified checkpoint (path to model###.pt file).")


def train_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_training_options(parser)
    return parser.parse_args()