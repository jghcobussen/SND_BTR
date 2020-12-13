import torch
import pandas as pd
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from bisect import bisect_left
from utils import CM_TO_MUM
from coord_conv import CoordConv
from statistics import mean

class Flatten(nn.Module):
    def forward(self, input):
        #print("Input of Flatten() looks like:")
        #print(input.size())
        print("Output of Flatten looks like: ")
        a = (input.view(input.size(0), -1))
        #print(a.size())
        return input.view(input.size(0), -1)


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, pool=False):
        super().__init__()
        self.block = nn.Sequential()
        self.block.add_module("conv", CoordConv(in_channels, out_channels,
                                                kernel_size=(k_size, k_size), stride=1, with_r=True))

        if pool:
            self.block.add_module("Pool", nn.MaxPool2d(2))
        self.block.add_module("BN", nn.BatchNorm2d(out_channels))
        self.block.add_module("Act", nn.ReLU())
        # self.block.add_module("dropout", nn.Dropout(p=0.5))

    def forward(self, x):
        #print("x from Block() looks like:")
        #print(x.size())
        return self.block(x)


class SNDNet(nn.Module):
    def __init__(self, n_input_filters):
        super().__init__()
        self.model = nn.Sequential(
            Block(n_input_filters, 32, pool=True),
            Block(32, 32, pool=True),
            Block(32, 64, pool=True),
            Block(64, 64, pool=True), ### was True
            #Block(64, 64, pool=False),
            #Block(64, 64, pool=True),
            #Block(128, 128, pool=False),
            Flatten(),
            nn.Linear(256, 1), ### was 64, 1
            #nn.Tanh() #ADDED
            #nn.Sigmoid() #ADDED
            
            
            # nn.ReLU(),
            # nn.Dropout(p=0.5),
            # nn.Linear(512, 512),            
            # nn.ReLU(),
            #nn.Linear(1280,2)
        )

    def compute_loss(self, X_batch, y_batch):
        X_batch = X_batch.to(self.device)
        print("X_batch has size: " + str(X_batch.size()))
        y_batch = y_batch.to(self.device)
        print(len(y_batch))
        logits = self.model(X_batch)
        loss_tensor = F.mse_loss(logits, y_batch, reduction = 'none')
        #loss_tensor = F.smooth_l1_loss(logits, y_batch, reduction = 'none')
        #loss_tensor = F.binary_cross_entropy(logits, y_batch, reduction = 'none')
        #loss_tensor = F.cross_entropy(logits, y_batch, reduction = 'none') 
        #loss_tensor = F.binary_cross_entropy_with_logits(logits, y_batch, reduction = 'none')
        normalized_loss = loss_tensor
        return normalized_loss.mean()

       # returnvalue = F.smooth_l1_loss(logits, y_batch).mean()
       # print(returnvalue)
       # print("returnvalue type: " + str(type(returnvalue)))
       # return returnvalue

    def predict(self, X_batch):
        self.model.eval()
        return self.model(X_batch.to(self.device))

    @property
    def device(self):
        return next(self.model.parameters()).device


class MyDataset(Dataset):
    """
    Class defines how to preprocess data before feeding it into net.
    """
    def __init__(self, TT_df, y, parameters, data_frame_indices, n_filters):
        """
        :param TT_df: Pandas DataFrame of events
        :param y: Pandas DataFrame of true electron energy and distance
        :param parameters: Detector configuration
        :param data_frame_indices: Indices to train/test on
        :param n_filters: Number of TargetTrackers in the simulation
        """
        self.indices = data_frame_indices
        self.n_filters = n_filters
        self.X = TT_df
        self.y = y
        self.params = parameters

    def __getitem__(self, index):
        return torch.FloatTensor(digitize_signal(self.X.iloc[self.indices[index]],
                                                 self.params,
                                                 filters=self.n_filters)),\
               torch.FloatTensor(self.y.iloc[self.indices[index]])

    def __len__(self):
        return len(self.indices)

def digitize_signal(event, params, filters=1):
    """
    Convert pandas DataFrame to image
    :param event: Pandas DataFrame line
    :param params: Detector configuration
    :param filters: Number of TargetTrackers in the simulation
    :return: numpy tensor of shape (n_filters, H, W).
    """
    shape = (filters,
             int(np.ceil(params.snd_params["Y_HALF_SIZE"] * 2 * CM_TO_MUM /
                         params.snd_params["RESOLUTION"])),
             int(np.ceil(params.snd_params["X_HALF_SIZE"] * 2 * CM_TO_MUM /
                         params.snd_params["RESOLUTION"])))
    response = np.zeros(shape)
    #print("Y_HALF is: " + str(params.snd_params["Y_HALF_SIZE"]))
    #print("X_HALF is: " + str(params.snd_params["X_HALF_SIZE"]))
    #print("RESOLUTION is: " + str(params.snd_params["RESOLUTION"]))
    #print("Shape looks like: " + str(shape))
    #print("Response looks like: " + str(response))
    
    #print("X looks like: ")
    #print("Z: " + str(len(event['Z'])))
    #print(event['Z'])
    #print("X: " + str(len(event['X'])))
    #print(event['X'])
    #print("Y: " + str(len(event['Y'])))
    #print(event['Y'])
    #print("Shape: " + str(shape))

    for x_index, y_index, z_pos in zip(np.floor((event['X'] + params.snd_params["X_HALF_SIZE"]) * CM_TO_MUM /
                                                params.snd_params["RESOLUTION"]).astype(int),
                                       np.floor((event['Y'] + params.snd_params["Y_HALF_SIZE"]) * CM_TO_MUM /
                                                params.snd_params["RESOLUTION"]).astype(int),
                                       event['Z']):
        #if x_index > 207:
        #    continue
        #if y_index > 171: 
        #    continue
        #print("event['X']: " + str(event['X']))
        #print("Indeces look like: ")
        #print(x_index, y_index, z_pos)

        response[params.tt_map[bisect_left(params.tt_positions_ravel, z_pos)],
                 shape[1] - y_index - 1,
                 x_index] += 1
    #print("Final response looks like: ")
    #print(response)
    return response
