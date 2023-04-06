"""
Train a model to learn the dynamics of the system
"""

import os
import json
from utils.fix_seed import fixseed
from utils.parser_util import train_args
FIX_SEED = True


def main():
    # parse arguments
    args = train_args()
    if FIX_SEED:
        fixseed(args.seed)
    
    # logging - Honestly, never tested this, but Tensorboard is cool for logging
    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name='Args')

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
    data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size)


    #TODO everything from here on out
    print("creating model and diffusion...")
    model = create_model(args, data)
    if args.cuda and torch.cuda.is_available():
        model.to('cuda:' + args.device)

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))
    print("Training...")
    TrainLoop(args, train_platform, model, diffusion, data).run_loop()
    train_platform.close()




if __name__ == "__main__":
    main()

