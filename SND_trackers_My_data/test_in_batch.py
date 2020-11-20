# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 23:50:36 2020

@author: joyce
"""

from utils import DataPreprocess, Parameters, Parameters_reduced
from net import SNDNet, MyDataset, digitize_signal
# usful module 
import torch
from matplotlib import pylab as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time
from tqdm import tqdm
from IPython import display
import os
import gc # Gabage collector interface (to debug stuff)

#-------------------------------------------------------------

try:
    assert(torch.cuda.is_available())
except:
    raise Exception("CUDA is not available")
n_devices = torch.cuda.device_count()
print("\nWelcome!\n\nCUDA devices available:\n")
for i in range(n_devices):
    print("\t{}\twith CUDA capability {}".format(torch.cuda.get_device_name(device=i), torch.cuda.get_device_capability(device=i)))
print("\n")
device = torch.device("cuda", 0)

#---------------------------------------------------------

# Turn off interactive plotting: for long run it makes screwing up everything
plt.ioff()

# Here we choose the geometry with 9 time the radiation length
params = Parameters_reduced("4X0")  #!!!!!!!!!!!!!!!!!!!!!CHANGE THE DIMENTION !!!!!!!!!!!!!!!!
processed_file_path = os.path.expandvars("/dcache/bfys/jcobus/ship_tt_processed_data") #!!!!!!!!!!!!!!!!!!!!!CHANGE THE PATH !!!!!!!!!!!!!!!!
processed_file_path_0 = os.path.expandvars("/dcache/bfys/jcobus/nue_CCDIS_0to200k/new")
processed_file_path_1 = os.path.expandvars("/dcache/bfys/jcobus/nue_NuEElastic_0to200k")
step_size = 5000    # size of a chunk
file_size = 100000  # size of the BigFile.root file
n_steps = int(file_size / step_size) # number of chunks

chunklist_TT_df = []  # list of the TT_df file of each chunk
chunklist_y_full = [] # list of the y_full file of each chunk

# It is reading and analysing data by chunk instead of all at the time (memory leak problem)
print("\nReading the tt_cleared_reduced.pkl & y_cleared.pkl files by chunk")
#First 2 
outpath = processed_file_path + "/{}".format(0)
chunklist_TT_df.append(pd.read_pickle(os.path.join(outpath, "tt_cleared_reduced.pkl")))
chunklist_y_full.append(pd.read_pickle(os.path.join(outpath, "y_cleared.pkl")))
outpath = processed_file_path + "/{}".format(1)
chunklist_TT_df.append(pd.read_pickle(os.path.join(outpath, "tt_cleared_reduced.pkl")))
chunklist_y_full.append(pd.read_pickle(os.path.join(outpath, "y_cleared.pkl")))

reindex_TT_df = pd.concat([chunklist_TT_df[0],chunklist_TT_df[1]],ignore_index=True)
reindex_y_full = pd.concat([chunklist_y_full[0],chunklist_y_full[1]], ignore_index=True)

for i in tqdm(range(n_steps-2)):  # tqdm: make your loops show a progress bar in terminal
    outpath = processed_file_path + "/{}".format(i+2)
    chunklist_TT_df.append(pd.read_pickle(os.path.join(outpath, "tt_cleared_reduced.pkl"))) # add all the tt_cleared.pkl files read_pickle and add to the chunklist_TT_df list
    chunklist_y_full.append(pd.read_pickle(os.path.join(outpath, "y_cleared.pkl"))) # add all the y_cleared.pkl files read_pickle and add to the chunklist_y_full list
    reindex_TT_df = pd.concat([reindex_TT_df,chunklist_TT_df[i+2]], ignore_index=True)
    reindex_y_full = pd.concat([reindex_y_full,chunklist_y_full[i+2]], ignore_index=True)

for i in tqdm(range(n_steps)):
    j = i + 10
    outpath = processed_file_path + "/{}".format(i)
    chunklist_TT_df.append(pd.read_pickle(os.path.join(outpath, "t_cleared_reduced.pkl")))
    chunklist_y_full.append(pd.read_pickle(os.path.join(outpath, "y_cleared.pkl")))
    reindex_TT_df = pd.concat([reindex_TT_df, chunklist_TT_df[j]], ignore_index=True)
    reindex_y_full = pd.concat([reindex_y_full, chunklist_y_full[j]], ignore_index=True)

# reset to empty space
chunklist_TT_df = []
chunklist_y_full = []
nb_of_plane = len(params.snd_params[params.configuration]["TT_POSITIONS"])

#---------------------------------------------------------

# True value of NRJ/dist for each true electron event

y = reindex_y_full[["L"]]
#NORM = 1. / 400
#y["E"] *= NORM

#y = reindex_y_full[["E"]]

y_test = reindex_y_full[["E"]]

# reset to empty space
reindex_y_full = []

# Spliting
print("\nSplitting the data into a training and a testing sample")

indeces = np.arange(len(reindex_TT_df))
train_indeces, test_indeces_raw, _, _ = train_test_split(indeces, indeces, train_size=0.9, random_state=1543)

# Break test data into parts
test_indeces_1, test_indeces_2, test_indeces_3, test_indeces_4 = [], [], [], []

for i in range (0, len(test_indeces_raw)):
    index = test_indeces_raw[i]
    row = y_test.loc[index]
    if 200 <= row[0] and row[0] < 250:
        test_indeces_1.append(index)
    elif 250 <= row[0] and row[0] < 300:
        test_indeces_2.append(index)
    elif 300 <= row[0] and row[0] < 350:
        test_indeces_3.append(index)
    elif 350 <= row[0] and row[0] <= 400:
        test_indeces_4.append(index)

print(len(test_indeces_1))
print(len(test_indeces_2))
print(len(test_indeces_3))
print(len(test_indeces_4))

#TrueE_test_all = y["E"][test_indeces_raw]
#np.save("TrueE_test_norm_smoothl1_all.npy", TrueE_test_all)
TrueE_test_1 = y["Label"][test_indeces_1]
np.save("TrueE_mse_norm_0.npy", TrueE_test_1)
TrueE_test_2 = y["E"][test_indeces_2]
np.save("TrueE_mse_norm_1.npy", TrueE_test_2)
TrueE_test_3 = y["E"][test_indeces_3]
np.save("TrueE_mse_norm_2.npy", TrueE_test_3)
TrueE_test_4 = y["E"][test_indeces_4]
np.save("TrueE_mse_norm_3.npy", TrueE_test_4)

#batch_size = 512
batch_size = 150

#test_dataset_all = MyDataset(reindex_TT_df, y, params, test_indeces_raw, n_filters=nb_of_plane)
#test_batch_gen_all = torch.utils.data.DataLoader(test_dataset_all, batch_size=batch_size, shuffle=False, num_workers=0)

test_dataset_1 = MyDataset(reindex_TT_df, y, params, test_indeces_1, n_filters=nb_of_plane)
test_batch_gen_1 = torch.utils.data.DataLoader(test_dataset_1, batch_size=batch_size, shuffle=False, num_workers=0)

test_dataset_2 = MyDataset(reindex_TT_df, y, params, test_indeces_2, n_filters=nb_of_plane)
test_batch_gen_2 = torch.utils.data.DataLoader(test_dataset_2, batch_size=batch_size, shuffle=False, num_workers=0)

test_dataset_3 = MyDataset(reindex_TT_df, y, params, test_indeces_3, n_filters=nb_of_plane)
test_batch_gen_3 = torch.utils.data.DataLoader(test_dataset_3, batch_size=batch_size, shuffle=False, num_workers=0)

test_dataset_4 = MyDataset(reindex_TT_df, y, params, test_indeces_4, n_filters=nb_of_plane)
test_batch_gen_4 = torch.utils.data.DataLoader(test_dataset_4, batch_size=batch_size, shuffle=False, num_workers=0)

#----------------------------------------- Testing the dataset with trained network

val_accuracy_1 = []
y_score = []

test_batch_gen_array = [test_batch_gen_1, test_batch_gen_2, test_batch_gen_3, test_batch_gen_4]

net = torch.load("9X0_file/" + str(29) + "_9X0_coordconv_DS5_mse_norm.pt") 

preds_0, preds_1, preds_2, preds_3 = [], [], [], []
preds_list = [preds_0, preds_1, preds_2, preds_3]

for j in range (0, 4):
    with torch.no_grad():
        for (X_batch, y_batch) in test_batch_gen_array[j]:
            preds_list[j].append(net.predict(X_batch))
        ans = np.concatenate([p.detach().cpu().numpy() for p in preds_list[j]])
        np.save("PredE_mse_norm_" + str(j) + ".npy",ans[:, 0])
        print("Save Prediction for batch "+ str(j))
