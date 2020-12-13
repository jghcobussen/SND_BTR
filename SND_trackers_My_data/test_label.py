
#!/usr/bin/env python3
# Import Class from utils.py & net.py file
from utils_full_unreduced import DataPreprocess, Parameters, Parameters_reduced
from net_unreduced import SNDNet, MyDataset, digitize_signal
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
import sys

plt.ioff()

# Test to see if cuda is available or not + listed the CUDA devices that are available
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
# Turn off interactive plotting: for long run it makes screwing up everything
plt.ioff()

# Here we choose the geometry with 9 time the radiation length
params = Parameters_reduced("SNDatLHC2")  #!!!!!!!!!!!!!!!!!!!!!CHANGE THE DIMENTION !!!!!!!!!!!!!!!!
#processed_file_path_2 = os.path.expandvars("/dcache/bfys/jcobus/DS5")
processed_file_path_1 = os.path.expandvars("/dcache/bfys/jcobus/nue_NuEElastic/0to200k_new") #!!!!!!!!!!!!!!!!!!!!!CHANGE THE PATH !!!!!!!!!!!!!!!!
processed_file_path_0 = os.path.expandvars("/dcache/bfys/jcobus/nue_CCDIS/0to200k_new")
step_size = 5000    # size of a chunk
file_size = 150000  # size of the BigFile.root file
n_steps = int(file_size / step_size) # number of chunks

# ------------------------------------------ LOAD THE reindex_TT_df & reindex_y_full PD.DATAFRAME --------------------------------------------------------------------------
chunklist_TT_df = []  # list of the TT_df file of each chunk
chunklist_y_full = [] # list of the y_full file of each chunk
chunklist_TT_test = []

# It is reading and analysing data by chunk instead of all at the time (memory leak problem)
print("\nReading the tt_cleared_reduced.pkl & y_cleared.pkl files by chunk")

#First 2
outpath_0 = processed_file_path_0 + "/{}".format(0)
chunklist_TT_df.append(pd.read_pickle(os.path.join(outpath_0, "tt_cleared.pkl"))) ### was tt_cleared_reduced
chunklist_y_full.append(pd.read_pickle(os.path.join(outpath_0, "y_cleared.pkl")))
outpath_0 = processed_file_path_0 + "/{}".format(1)
chunklist_TT_df.append(pd.read_pickle(os.path.join(outpath_0, "tt_cleared.pkl"))) ### was tt_cleared_reduced
chunklist_y_full.append(pd.read_pickle(os.path.join(outpath_0, "y_cleared.pkl")))

reindex_TT_df = pd.concat([chunklist_TT_df[0], chunklist_TT_df[1]],ignore_index=True)
reindex_y_full = pd.concat([chunklist_y_full[0], chunklist_y_full[1]], ignore_index=True)

for i in tqdm(range(n_steps-2)):  # tqdm: make your loops show a progress bar in terminal
    outpath_0 = processed_file_path_0 + "/{}".format(i+2)
    chunklist_TT_df.append(pd.read_pickle(os.path.join(outpath_0, "tt_cleared.pkl"))) ### was tt_cleared_reduced # add all the tt_cleared.pkl files read_pickle and add to $    chunklist_y_full.append(pd.read_pickle(os.path.join(outpath_0, "y_cleared.pkl"))) # add all the y_cleared.pkl files read_pickle and add to the chunklist_y_full list
    chunklist_y_full.append(pd.read_pickle(os.path.join(outpath_0, "y_cleared.pkl")))
    reindex_TT_df = pd.concat([reindex_TT_df,chunklist_TT_df[i+2]], ignore_index=True)
    reindex_y_full = pd.concat([reindex_y_full,chunklist_y_full[i+2]], ignore_index=True)

print("TT_df inelastic: " + str(len(reindex_TT_df)))
print("y_full inelastic: " + str(len(reindex_y_full)))

event_limit = 23450
remove = int(len(reindex_TT_df)-event_limit)
reindex_TT_df = reindex_TT_df[:-remove]
reindex_y_full = reindex_y_full[:-remove]

print("TT_df inelastic after cut: " + str(len(reindex_TT_df)))
print("y_full inelastic after cut: " + str(len(reindex_y_full)))
for i in tqdm(range(n_steps+10)):
    j = n_steps + i
    outpath_1 = processed_file_path_1 + "/{}".format(i)
    chunklist_TT_df.append(pd.read_pickle(os.path.join(outpath_1, "tt_cleared.pkl"))) ### was tt_cleared_reduced
    chunklist_y_full.append(pd.read_pickle(os.path.join(outpath_1, "y_cleared.pkl")))
    reindex_TT_df = pd.concat([reindex_TT_df, chunklist_TT_df[j]], ignore_index=True)
    reindex_y_full = pd.concat([reindex_y_full, chunklist_y_full[j]], ignore_index=True)

print("TT_df both:" + str(len(reindex_TT_df)))
print("y_full both: " + str(len(reindex_y_full)))

event_limit_full = 2*event_limit
remove = int(len(reindex_TT_df)-event_limit_full)
reindex_TT_df = reindex_TT_df[:-remove]
reindex_y_full = reindex_y_full[:-remove]

print("TT_df both after cut: " + str(len(reindex_TT_df)))
print("y_full both after cut: " + str(len(reindex_y_full)))

elastic_array, inelastic_array = [], []
for i in range (0, len(reindex_y_full)):
    if reindex_y_full.iloc[i]['Label'] == 1:
        elastic_array.append(i)
    elif reindex_y_full.iloc[i]['Label'] == -1:
        inelastic_array.append(i)

print("Number of inelastic events: " + str(len(inelastic_array)))
print("Number of elastic events: " + str(len(elastic_array)))

# reset to empty space
chunklist_TT_df = []
chunklist_y_full = []
#chunklist_TT_test = []

nb_of_plane = len(params.snd_params[params.configuration]["TT_POSITIONS"])
#sys.exit()

#sys.exit()

# True value of NRJ/dist for each true electron event
#y = reindex_y_full[["E", "Z","THETA"]]
y = reindex_y_full[["Label"]]
f = open("y_full_array.txt", "w+")
f.write(reindex_y_full.to_string())
f.close()
### NORM = 1. / 400
### y["E"] *= NORM
#y["Z"] *= -1
#y["THETA"] *= (180/np.pi)

y_test = reindex_y_full[["E"]]

# reset to empty space
# reindex_y_full = []

# Spliting
print("\nSplitting the data into a training and a testing sample")

indeces = np.arange(len(reindex_TT_df))
train_indeces, test_indeces, _, _ = train_test_split(indeces, indeces, train_size=0.9, random_state=1543) ### train_size = 0.9

#break test data into parts
test_indeces_1, test_indeces_2, test_indeces_3, test_indeces_4 = [], [], [], []

for i in range(0,len(test_indeces)):
    index = test_indeces[i]
    row = y_test.loc[index]
    if 0 <= row[0] and row[0] < 1250:
        test_indeces_1.append(index)
    if 1250 <= row[0] and row[0] < 2500:
        test_indeces_2.append(index)
    if 2500 <= row[0] and row[0] < 3750:
        test_indeces_3.append(index)
    if 3750 <= row[0] and row[0] < 5100:
        test_indeces_4.append(index)

print(len(test_indeces_1))
print(len(test_indeces_2))
print(len(test_indeces_3))
print(len(test_indeces_4))

#Save which train indeces are which
train_indeces_elastic, train_indeces_inelastic = [], []
for i in range(0, len(train_indeces)):
    index = train_indeces[i]
    if reindex_y_full.iloc[index]['Label'] == 1:
        train_indeces_elastic.append(index)
    elif reindex_y_full.iloc[index]['Label'] == -1:
        train_indeces_inelastic.append(index)

test_indeces_elastic, test_indeces_inelastic = [], []
for i in range (0, len(test_indeces)):
    index = test_indeces[i]
    if reindex_y_full.iloc[index]['Label'] == 1:
        test_indeces_elastic.append(index)
    elif reindex_y_full.iloc[index]['Label'] == -1:
        test_indeces_inelastic.append(index)

'''

f = open("train_test_indeces_Vh5.txt", "a")
f.write("train_indeces_elastic: ")
f.write("\t" + str(train_indeces_elastic))
f.write("\ttrain_indeces_inelastic: ")
f.write("\t" + str(train_indeces_inelastic))
f.write("\ttest_indeces_elastic: ")
f.write("\t" + str(test_indeces_elastic))
f.write("\ttest_indeces_inelastic: ")
f.write("\t" + str(test_indeces_inelastic))
f.close

print("Length of train_indeces_elastic: " + str(len(train_indeces_elastic)))
print("Length of train_indeces_inelastic: " + str(len(train_indeces_inelastic)))
print("Length of test_indeces_elastic: " + str(len(test_indeces_elastic)))
print("Length of test_indeces_inelastic: " + str(len(test_indeces_inelastic)))

'''
# Save some useful information about the events
elastic_train_E, inelastic_train_E, elastic_test_E, inelastic_test_E = [], [], [], []
elastic_train_nb, inelastic_train_nb, elastic_test_nb, inelastic_test_nb = [], [], [], []
elastic_train_index, inelastic_train_index, elastic_test_index, inelastic_test_index = [], [], [], []


for i in range(0, len(train_indeces_elastic)):
    elastic_train_E.append(reindex_y_full.iloc[train_indeces_elastic[i]]['E'])
    energy = reindex_TT_df.iloc[train_indeces_elastic[i]]['X']
    elastic_train_nb.append(len(energy))
    elastic_train_index.append(train_indeces_elastic[i])
for i in range(0, len(train_indeces_inelastic)):
    inelastic_train_E.append(reindex_y_full.iloc[train_indeces_inelastic[i]]['E'])
    energy = reindex_TT_df.iloc[train_indeces_inelastic[i]]['X']
    inelastic_train_nb.append(len(energy))
    inelastic_train_index.append(train_indeces_inelastic[i])
for i in range(0, len(test_indeces_elastic)):
    elastic_test_E.append(reindex_y_full.iloc[test_indeces_elastic[i]]['E'])
    energy = reindex_TT_df.iloc[test_indeces_elastic[i]]['X']
    elastic_test_nb.append(len(energy))
    elastic_test_index.append(test_indeces_elastic[i])
for i in range (0, len(test_indeces_inelastic)):
    inelastic_test_E.append(reindex_y_full.iloc[test_indeces_inelastic[i]]['E'])
    energy = reindex_TT_df.iloc[test_indeces_inelastic[i]]['X']
    inelastic_test_nb.append(len(energy))
    inelastic_test_index.append(test_indeces_inelastic[i])

a0, b0, c0, d0 = [], [], [], []
a0.append(elastic_train_index)
a0.append(elastic_train_E)
a0.append(elastic_train_nb)
b0.append(inelastic_train_index)
b0.append(inelastic_train_E)
b0.append(inelastic_train_nb)
c0.append(elastic_test_index)
c0.append(elastic_test_E)
c0.append(elastic_test_nb)
d0.append(inelastic_test_index)
d0.append(inelastic_test_E)
d0.append(inelastic_test_nb)

np.save("elastic_train_Vf_range.npy", a0)
np.save("inelastic_train_Vf_range.npy", b0)
np.save("elastic_test_Vf_range.npy", c0)
np.save("inelastic_test_Vf_range.npy", d0)

#sys.exit()

TrueE_test_1 = y["Label"][test_indeces_1]
np.save("True_Label_Vf_0.npy", TrueE_test_1)
TrueE_test_2 = y["Label"][test_indeces_2]
np.save("True_Label_Vf_1.npy", TrueE_test_2)
TrueE_test_3 = y["Label"][test_indeces_3]
np.save("True_Label_Vf_2.npy", TrueE_test_3)
TrueE_test_4 = y["Label"][test_indeces_4]
np.save("True_Label_Vf_3.npy", TrueE_test_4) 

#batch_size = 512
batch_size = 150

#test_dataset_all = MyDataset(reindex_TT_df, y, params, test_indeces, n_filters=nb_of_plane)
#test_batch_gen_all = torch.utils.data.DataLoader(test_dataset_all, batch_size=batch_size, shuffle=False, num_workers=0)

test_dataset_1 = MyDataset(reindex_TT_df, y, params, test_indeces_1, n_filters=nb_of_plane)
test_batch_gen_1 = torch.utils.data.DataLoader(test_dataset_1, batch_size=batch_size, shuffle=False, num_workers=0)

test_dataset_2 = MyDataset(reindex_TT_df, y, params, test_indeces_2, n_filters=nb_of_plane)
test_batch_gen_2 = torch.utils.data.DataLoader(test_dataset_2, batch_size=batch_size, shuffle=False, num_workers=0)

test_dataset_3 = MyDataset(reindex_TT_df, y, params, test_indeces_3, n_filters=nb_of_plane)
test_batch_gen_3 = torch.utils.data.DataLoader(test_dataset_3, batch_size=batch_size, shuffle=False, num_workers=0)

test_dataset_4 = MyDataset(reindex_TT_df, y, params, test_indeces_4, n_filters=nb_of_plane)
test_batch_gen_4 = torch.utils.data.DataLoader(test_dataset_4, batch_size=batch_size, shuffle=False, num_workers=0)


val_accuracy_1 = []
y_score = []

test_batch_gen_array = [test_batch_gen_1, test_batch_gen_2, test_batch_gen_3, test_batch_gen_4]

net = torch.load("9X0_file/" + str(19) + "_9X0_coordconv_Label_Vf.pt")

preds_0, preds_1, preds_2, preds_3 = [], [], [], []
preds_list = [preds_0, preds_1, preds_2, preds_3]

for j in range (0, 4):
    with torch.no_grad():
        for (X_batch, y_batch) in test_batch_gen_array[j]:
            preds_list[j].append(net.predict(X_batch))
        ans = np.concatenate([p.detach().cpu().numpy() for p in preds_list[j]])
        np.save("PredE_Vf_" + str(j) + ".npy",ans[:, 0])
        print("Save Prediction for batch "+ str(j))

