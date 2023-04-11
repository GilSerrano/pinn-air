import os
import json

def check_save_directory(args):

    # Check if the save directory was even passed as an argument
    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    
    # Check if the save directory exists and if it should be overwritten
    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
    
    # Create the save directory if it doesn't exist
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Save the arguments to the save directory
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)