from codeop import compile_command
from torch.utils import data
from gil.data_processing.h5df_dataset_loader import HDF5Dataset
from gil.policy.model.MotionNet import SetupTableMoveNet
from gil.policy.model.utils import initialize_weights
import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from datetime import datetime
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

NUM_EPOCHS = 100
BATCH_SIZE = 100
DATA_FOLDER = os.path.dirname(os.path.realpath(__file__))+"/../../../data/policy/"
MODEL_NAME = "motionmove"
SAVED_MODELS_FOLDER = DATA_FOLDER+"saved_models/"
LOGDIR = DATA_FOLDER+"training_log/"+MODEL_NAME+"_"+str(datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))+"/"
DATASET_FOLDER = "/home/nkquynh/gil_ws/gil_hierachial_hmp/hierarchical-hmp/gil/datasets/gil_dataset/set_table/without_human_30_04_2022_20_40_21/motion_data/move"
TRAIN_TEST_RATE = 0.8
CHECKPOINT_AFTER = 50
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter(LOGDIR)
#Init the tasknet
motionnet = SetupTableMoveNet(observation_dim=14,action_dim=2)
motionnet.to(device)
motionnet.apply(initialize_weights)

#Load the Task dataset
loader_params = {'batch_size': BATCH_SIZE, 'shuffle': True, 'num_workers': 2}
dataset = HDF5Dataset(DATASET_FOLDER, data_cache_size=1000,recursive=True, load_data=False)

#Split the dataset into two set
trainset_size = int(TRAIN_TEST_RATE*len(dataset))
testset_size = len(dataset)-trainset_size
train_set, test_set = data.random_split(dataset, (trainset_size, testset_size))


data_loader = data.DataLoader(train_set, **loader_params)
test_loader = data.DataLoader(test_set, **loader_params)

print("data size ", len(data_loader))
print("test size", len(test_loader))
#Define the criterion and optimizer
criterion = nn.MSELoss()

#Using SGD optimizer, learning rate 10^-3, and L2 regulization 4*10^-4
optimizer = optim.SGD(motionnet.parameters(), lr=2*10e-4, weight_decay=1e-4)

start_time = time.time()
start_loop_time = time.time()
for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    for i, return_value in enumerate(data_loader):
        observation, command = return_value
        observation = observation.to(device)
        command = command.to(device).float()
        
        #print(observation.device)
        #Zero gradient
        optimizer.zero_grad()
        #Forward
        command_pred = motionnet.forward(observation.float())
        #Loss is the summation of the Cross Entropies of Multihead output
        loss = criterion(command_pred, command)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % CHECKPOINT_AFTER == 0:    # print every 5 mini-batches
            period = time.time() - start_loop_time
            start_loop_time = time.time()
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / CHECKPOINT_AFTER:.3f}. Time per {CHECKPOINT_AFTER} batches: {period}')
            writer.add_scalar("motion_move/train_lost", running_loss / CHECKPOINT_AFTER, epoch*len(data_loader)+i)
            torch.save(motionnet.state_dict(), SAVED_MODELS_FOLDER+MODEL_NAME+".pth")
            running_loss = 0.0       

    with torch.no_grad():
        for i, return_value in enumerate(test_loader):
            observation, command = return_value
            observation = observation.to(device)
            command = command.to(device).float()
            #print(observation.device)
            #Zero gradient
            optimizer.zero_grad()
            #Forward
            command_pred = motionnet.forward(observation.float())
            #Loss is the summation of the Cross Entropies of Multihead output
            loss = criterion(command_pred, command)
            # print statistics
            running_loss += loss.item()
            # if i % CHECKPOINT_AFTER == 0:    # print every 5 mini-batches
            #     print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / CHECKPOINT_AFTER:.3f}')
                
            #     running_loss = 0.0         
        writer.add_scalar("motion_move/test_lost", running_loss / i, (epoch+1)*len(data_loader))

torch.save(motionnet.state_dict(), SAVED_MODELS_FOLDER+MODEL_NAME+".pth")
print('Finished Training')      
total_time = time.time()-start_time


#print("Example", motionnet.forward(dataset.__getitem__(10)[0].float().to(device)))
print("Total time:",total_time)

