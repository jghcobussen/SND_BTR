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
        return self.block(x)


class SNDNet(nn.Module):
    def __init__(self, n_input_filters):
        super().__init__()
        self.model = nn.Sequential(
            Block(n_input_filters, 32, pool=True),
            Block(32, 32, pool=True),
            Block(32, 64, pool=True),
            Block(64, 64, pool=True),
            #Block(32, 32, pool=True),
            #Block(128, 128, pool=False),
            Flatten(),
            nn.Linear(256, 1),
            # nn.ReLU(),
            # nn.Dropout(p=0.5),
            # nn.Linear(512, 512),
            # nn.ReLU(),
            #nn.Linear(1280,2)
        )

    def compute_loss(self, X_batch, y_batch):
        X_batch = X_batch.to(self.device)
        y_batch = y_batch.to(self.device)
        logits = self.model(X_batch)
        loss_tensor = F.smooth_l1_loss(logits, y_batch, reduction = 'none') 
        normalized_loss = loss_tensor/(y_batch*y_batch)
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

def reconstruct_ghosts(event):
    array = event.to_numpy()
    x_hits, y_hits = [], []
    for i in range (0, len(array)):
        x = array[i][3]
        x_hits.append(x)
        y = array[i][4]
        y_hits.append(y)

    ghost_coord = []

    for i in range(0, len(x_hits)):
        x = x_hits[i]
        for j in range (0, len(y_hits)):
            y = y_hits[j]
            coord = [x,y]
            ghost_coord.append(coord)
    df_ghost_coord = pd.DataFrame(np.array(ghost_coord), columns=['X', 'Y'])
    return df_ghost_coord

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
    
    #event = reconstruct_ghosts(old_event)

    for x_index, y_index, z_pos in zip(np.floor((event['X'] + params.snd_params["X_HALF_SIZE"]) * CM_TO_MUM /
                                                params.snd_params["RESOLUTION"]).astype(int),
                                       np.floor((event['Y'] + params.snd_params["Y_HALF_SIZE"]) * CM_TO_MUM /
                                                params.snd_params["RESOLUTION"]).astype(int),
                                       event['Z']):
        response[params.tt_map[bisect_left(params.tt_positions_ravel, z_pos)],
                 shape[1] - y_index - 1,
                 x_index] += 1
    return response
