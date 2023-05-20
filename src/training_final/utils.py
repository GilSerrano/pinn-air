import os
import torch

def load_best_model(best_epoch, model, output_dir="./output", device="cpu"):

    checkpoint_path = os.path.join(output_dir, f"epoch_{best_epoch}_best_model.pt")
    print("Loading: {}".format(checkpoint_path))

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])

    return model.to(device)

def fetch_best_epoch(output_dir="output/", file="best_epoch.txt"):
    """
        fetch the model from the best epoch
        it corresponds to the last line in the file
    """

    with open(os.path.join(output_dir, file), "rb") as f:
        try:  # catch OSError in case of a one line file 
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b'\n':
                f.seek(-2, os.SEEK_CUR)
        except OSError:
            f.seek(0)
        best_epoch = f.readline().decode()

    return best_epoch