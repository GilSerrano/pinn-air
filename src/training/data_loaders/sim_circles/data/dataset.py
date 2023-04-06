import torch
from torch.utils import data
import numpy as np
import os
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import spacy

from torch.utils.data._utils.collate import default_collate
from data_loaders.sim_circles.utils.get_opt import get_opt

'''
Base class for the simulation dataset, other datasets based in simulation can inherit from this class,
or include it in their parameters
'''
class SimDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt

        data_dict = {}
        id_list = []

        # split file has the IDs of the sequences to be used for training or testing
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                #TODO - replace with our actual dataset files and directories
                # this currently loads the motion and text data from the HumanML dataset, which
                # has a file for each motion sequence, and each file contains the motion data
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                data_dict[name] = {'motion': motion,
                                    'length': len(motion)}
                
                new_name_list.append(name)
                length_list.append(len(motion))
            except:
                print("DEBUG - error in loading {}".format(name))

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list

    def __len__(self):
        return len(self.data_dict) 

    def __getitem__(self, item):
        idx =  item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length = data['motion'], data['length']


        return  motion, m_length



# A wrapper class for the sim_circles dataset
class SIM_CIRCLES(data.Dataset):
    def __init__(self, datapath='./dataset/sim_circles/sim_circles_opt.txt', split="train", **kwargs):
        
        self.dataset_name = 'sim_circles'
        self.dataname = 'sim_circles'

        # Configurations of sim_circles dataset 
        abs_base_path = f'.'
        dataset_opt_path = pjoin(abs_base_path, datapath)
        device = None  # torch.device('cuda:0') # This param is not in use in this context
        opt = get_opt(dataset_opt_path, device)
        opt.meta_dir = pjoin(abs_base_path, opt.meta_dir)
        opt.model_dir = pjoin(abs_base_path, opt.model_dir)
        opt.checkpoints_dir = pjoin(abs_base_path, opt.checkpoints_dir)
        opt.data_root = pjoin(abs_base_path, opt.data_root)
        opt.save_root = pjoin(abs_base_path, opt.save_root)
        opt.meta_dir = './dataset'
        self.opt = opt
        print('Loading dataset %s ...' % opt.dataset_name)

        # split file has the IDs of the sequences to be used for training or testing
        self.split_file = pjoin(opt.data_root, f'{split}.txt')
        self.sim_dataset = SimDataset(self.opt, self.split_file)

        assert len(self.sim_dataset) > 1, 'You loaded an empty dataset, '

    def __getitem__(self, item):
        return self.sim_dataset.__getitem__(item)

    def __len__(self):
        return self.sim_dataset.__len__()