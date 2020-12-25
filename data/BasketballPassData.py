import os
import cv2
import re
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import skimage as skm
import h5py

class BasketballData(Dataset):

    def __init__(self, datapath):
        self.datapath = datapath
        #self.frames = self.fhandle['seq_0'].shape[1]
        self.order = ['BasketballPass']
        fhandle = h5py.File(self.datapath, 'r')
        self.fkeys = list(fhandle.keys()) 
        fhandle.close()

    def __len__(self):
        return 1


    def __getitem__(self, index):
        fhandle = h5py.File(self.datapath, 'r')
        #print(self.order[index]) 
        match = re.compile(self.order[index])
        mfile = []
        for fna in self.fkeys:
            mout = match.findall(fna)
            if len(mout) > 0:
                mfile.append(fna)

        print(fhandle[mfile[0]])
        #print(len(mfile))
        assert len(mfile) == 2

        for i in mfile:
            if i.endswith('cmp'):
                #cmp_seq = fhandle[i]
                #print(fhandle[i])
                cmp_seq = fhandle[i][::]/255.0
            elif i.endswith('raw'):
                #raw_seq = fhandle[i]
                raw_seq = fhandle[i][::]/255.0

        # change shape to num * 1 * h * w
        sh = cmp_seq.shape
        #print(sh)
        cmp_seq = np.reshape(cmp_seq,(sh[0], 1, sh[1], sh[2]))
        raw_seq = np.reshape(raw_seq,(sh[0], 1, sh[1], sh[2]))

        return {'compressed_frames':torch.from_numpy(cmp_seq).float(),
                'raw_frames': torch.from_numpy(raw_seq).float()}


    
