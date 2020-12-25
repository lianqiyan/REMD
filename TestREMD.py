import numpy as np
import cv2
import h5py
import torch
import csv
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torch.nn.functional as F
from collections import OrderedDict
from lib.PAUnet import ArRNN
from data.BasketballPassData import BasketballData
import os
from tqdm import tqdm
import skimage as skm
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "6"


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        return loss


def parse_args():
    parser = argparse.ArgumentParser(description='EDSR and MDSR')
    parser.add_argument('--input_size', type=int, default=128, help='image size of training')
    parser.add_argument('--learning rate', type=float,default=1e-4, help='learning rate')
    parser.add_argument('--input_dim', type=int, default=5, help='Input dim')
    parser.add_argument('--hidden_dim', type=int, default=24, help='Hidden dim')
    parser.add_argument('--is_train', type=bool, default=True, help='If training')
    parser.add_argument('--kernel_size', type=int, default=3, help='Kernel size')
    parser.add_argument('--num_RBs', type=int, default=24, help='num of reidual blocks')
    parser.add_argument('--fmaps', type=int, default=16, help='fmaps')

    args = parser.parse_args()
    return args

class STAAR():
    def __init__(self, is_train, learning_rate=1e-4, model_path=None):
        self.is_train = is_train
        self.learning_rate = learning_rate
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args = parse_args()
        model = ArRNN(args)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        self.model = model.to(self.device)

        if self.is_train:
            self.model.train()
            #self.pixel_loss = nn.MSELoss(reduction='sum').to(self.device)
            self.cri_pix = CharbonnierLoss().to(self.device)

            self.optim_params = []
            for k, v in self.model.named_parameters():
                if v.requires_grad:
                    self.optim_params.append(v)
                else:
                    print('Params [{:s}] will not optimize.'.format(k))
            self.optimizer = torch.optim.Adam(self.optim_params, lr=self.learning_rate)
                                                #weight_decay=0.02)
        
    def feed_data(self, data):
        self.cmp_seq = data['compressed_frames'].to(self.device)
        self.label_seq = data['raw_frames'].to(self.device)

    def update_lr(self):
        self.learning_rate = self.learning_rate * 0.1
        self.optimizer = torch.optim.Adam(self.optim_params, lr=self.learning_rate)

    def get_parameter_number(self):
        total_num = sum(p.numel() for p in self.model.parameters())
        trainable_num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}
        

    def optimize_parameters(self, i):

        self.gen_seq  = self.model(self.cmp_seq, i)

        loss =  self.cri_pix(self.gen_seq, self.label_seq) 
        loss_value = loss.item()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss_value

    def load(self):
        if self.model_path is not None:
            print('Loading model for G [{:s}] ...'.format(self.model_path))
            self.load_network(self.model_path, self.model)

    def save(self, model_name):
        self.save_network(self.model, network_label=model_name)


    def test(self, test_seq,epoch, pre1=None, pre2=None):
        self.model.eval()
        with torch.no_grad():
            tout_seq = self.model(test_seq,epoch,pre1,pre2)
        self.model.train()
        return tout_seq
        
    def save_network(self,  network, network_label):
        save_filename = '{}.pth'.format(network_label)
        save_filename = os.path.join('./model', save_filename)
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_filename)


    def load_network(self, load_path, network, strict=True):
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        load_net = torch.load(load_path)
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        network.load_state_dict(load_net_clean, strict=strict)


    def resume_training(self, resume_state):
        '''Resume the optimizers and schedulers for training'''
        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        assert len(resume_optimizers) == len(self.optimizers), 'Wrong lengths of optimizers'
        assert len(resume_schedulers) == len(self.schedulers), 'Wrong lengths of schedulers'
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)


    
def get_test_psnr(train_seq, model_out, label, index, name):
    
    train_seq = train_seq.cpu().clone().numpy() * 255.0
    train_seq = train_seq.astype(np.uint8)

    if isinstance(model_out, torch.Tensor):
        model_out = model_out.cpu().clone().numpy() * 255.0
        model_out = model_out.astype(np.uint8)
    else:
        model_out = model_out* 255.0
        model_out = model_out.astype(np.uint8)


    label = label.cpu().clone().numpy() * 255.0
    label = label.astype(np.uint8)

    rpsnr = np.zeros((train_seq.shape[1], 1))
    epsnr = np.zeros((train_seq.shape[1], 1))
    
    #print(label.shape)
    ops = np.zeros(model_out.shape[0])
    tavg = np.zeros(model_out.shape[0])
    for n in range(model_out.shape[0]):
        for k in range(model_out.shape[1]):
            epsnr[k] =  skm.measure.compare_psnr(label[n,k,:,:,:], model_out[n,k,:,:,:])
            rpsnr[k] =  skm.measure.compare_psnr(label[n,k,:,:,:], train_seq[n,k,:,:,:])
        ops[n] = np.mean(rpsnr)
        tavg[n] = np.mean(epsnr)

    
    file_name = name+ '__psnr_reccord.csv'
    with open(file_name, 'a+') as f:
        csv_write = csv.writer(f)
        data_row = ['seq_'+str(index+1)+'_original:',np.mean(ops), 'enhanced:', np.mean(tavg),  'Delta:', np.mean(tavg-ops)]
        csv_write.writerow(data_row)


def crop_test(vali_data, ArModel):
        #ArModel.model = ArModel.model.cuda()
        vali_data = vali_data.cuda()
        sh = list(vali_data.size()) # the shape is N*1*W*H
        if sh[2] == 1080: #  1920*1080
            ws = sh[2]
            hs = sh[3]//8
            #if sh[0]>510: # smaller
                #ws = ws//2
        elif sh[2] == 1600: # 2560*1600
            ws = sh[2]//2
            hs = sh[3]//8
        elif sh[2] == 720:  # 1280*720
            ws = sh[2]//2
            hs = sh[3]//2
        elif sh[2] == 480:  # 832*480
            ws = sh[2]
            hs = sh[3]
        else:   # 416*240
            ws = sh[2]
            hs = sh[3]
        
        c_patchs = []
        for x in range(0, sh[2], ws): #divide into patchs
            for y in range(0, sh[3], hs):
                x_end = x + ws
                y_end = y + hs
                c_temp = vali_data[:,:,x:x_end, y:y_end]
                c_patchs.append(c_temp)

        vali_list = [torch.stack(c_patchs, dim=0)]
            
        out = []
        ts = 1

        out = []
        pre_gen1 = None
        pre_gen2 = None

        for clip in vali_list:
            batch_out = []
            print(clip.shape)
            for k in range(0, clip.shape[0], ts):
                k_end = k + ts
                temp_out = ArModel.test(clip[k:k_end,::],10, pre_gen1, pre_gen2)
                batch_out.append(temp_out)

                temp_size = list(temp_out.size())
                pre_gen1 = temp_out[:,-1,::]
                if temp_size[1]>1:
                    pre_gen2 = temp_out[:,-2,::]
            out.append(torch.cat(batch_out, dim=0))


        if len(out)>1:
            print(out[0].shape)
            temp_out = torch.cat(out, dim=1)
        else:
            temp_out = out[0]

        num = 0
        vali_out = torch.zeros(sh)
        for x in range(0, sh[2], ws): #divide into patchs
            for y in range(0, sh[3], hs):
                x_end = x + ws
                y_end = y + hs
                vali_out[:,:,x:x_end, y:y_end] = temp_out[num,::]
                num = num + 1

        vali_out = vali_out.view(1,sh[0],sh[1],sh[2],sh[3])

        return vali_out


def test_consective_frame_clip(head=''):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = './model/REMD.pth'
    test_path = './data/BasketballPass_QP42.h'
    print("Testing Lowdelay P mode")

    test_dat = BasketballData(test_path)

    is_train = False
    learning_rate=1e-5

    ArModel = STAAR(is_train, learning_rate, model_path)
    ArModel.load()

    index = 0;
    sh = list(test_dat[index]['raw_frames'].size()) # the shape is N*1*W*H
    vali_data = test_dat[index]['compressed_frames']
    vali_label = test_dat[index]['raw_frames']
    print("crop test")
    vali_out = crop_test(vali_data,ArModel)


    vali_data = vali_data.view(1,sh[0],sh[1],sh[2],sh[3])
    vali_label = vali_label.view(1,sh[0],sh[1],sh[2],sh[3])

    print(vali_out.shape)

    get_test_psnr(vali_data, vali_out, vali_label, index, head+'_REMD_result_')

if  __name__ == '__main__':
    test_consective_frame_clip(head='BasketballPass')



