"""
Train a model to learn the dynamics of the system
"""

import os
import json
from utils.fix_seed import fixseed
from utils.parser_util import train_args
from data_loaders.get_data import get_dataset_loader


FIX_SEED = True

def main():
    # parse arguments
    args = train_args()
    
    
    if FIX_SEED:
        fixseed(args.seed)

    # Create save directory
    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    print("creating data loader...")
    data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, datapath=args.data_dir)

    #TODO everything from here on out
    print("creating model...")
    model = create_model(args, data)
    if args.cuda and torch.cuda.is_available():
        model.to('cuda:' + args.device)

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))
    print("Training...")
    TrainLoop(args, train_platform, model, diffusion, data).run_loop()




if __name__ == "__main__":
    main()

