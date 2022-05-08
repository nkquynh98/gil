from torch.utils import data
from gil.data_processing.h5df_dataset_loader import HDF5Dataset
from gil.policy.model.TaskNet import SetupTableTasknet
from gil.policy.model.utils import initialize_weights
from datetime import datetime
#For visulizing training process:
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
NUM_EPOCHS = 100
BATCH_SIZE = 100
DATA_FOLDER = os.path.dirname(os.path.realpath(__file__))+"/../../../data/policy/"
SAVED_MODELS_FOLDER = DATA_FOLDER+"saved_models/with_holding_encoded/"
MODEL_NAME = "tasknet"
LOGDIR = DATA_FOLDER+"training_log/"+MODEL_NAME+"_"+str(datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))+"/"
DATASET_ROOT = os.path.dirname(os.path.realpath(__file__))+"/../../../datasets/gil_dataset/"
DATASET_NAME = "set_table/with_holding_encoded_01_05_2022_18_04_35/"
DATASET_FOLDER = DATASET_ROOT+DATASET_NAME+"task_data"
CHECKPOINT_AFTER = 50
TRAIN_TEST_RATE = 0.8
writer = SummaryWriter(log_dir=LOGDIR)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#Init the tasknet
tasknet = SetupTableTasknet(goal_encoded_dim = 10, observation_dim=38, action_dim = 3, object_dim = 10, location_dim=3)
tasknet.to(device)
tasknet.apply(initialize_weights)

#Load the Task dataset
loader_params = {'batch_size': BATCH_SIZE, 'shuffle': True, 'num_workers': 2}
dataset = HDF5Dataset(DATASET_FOLDER, data_cache_size=100000,recursive=True, load_data=False)

#Split the dataset into two set
trainset_size = int(TRAIN_TEST_RATE*len(dataset))
testset_size = len(dataset)-trainset_size
train_set, test_set = data.random_split(dataset, (trainset_size, testset_size))


data_loader = data.DataLoader(train_set, **loader_params)
test_loader = data.DataLoader(test_set, **loader_params)

print("data size ", len(data_loader))
print("test size", len(test_loader))
#Define the criterion and optimizer
criterion = nn.CrossEntropyLoss()


#Using SGD optimizer, learning rate 10^-3, and L2 regulization 4*10^-4
optimizer = optim.SGD(tasknet.parameters(), lr=10e-3, weight_decay=4e-4)

start = time.time()
for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    for i, return_value in enumerate(data_loader):
        observation, action, locations, objects = return_value
        observation = observation.to(device)
        action = action.to(device)
        objects = objects.to(device)
        locations= locations.to(device)
        #print(observation.device)
        #Zero gradient
        optimizer.zero_grad()

        #Forward
        action_pred, location_pred, object_pred = tasknet.forward(observation.float())
        #Loss is the summation of the Cross Entropies of Multihead output
        loss = criterion(action_pred, action)+criterion(object_pred,objects)+criterion(location_pred, locations)
        

        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % CHECKPOINT_AFTER == 0:    # print every 5 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / CHECKPOINT_AFTER:.3f}')
            writer.add_scalar("Task/train_lost", running_loss / CHECKPOINT_AFTER, epoch*len(data_loader)+i)
            torch.save(tasknet.state_dict(), SAVED_MODELS_FOLDER+MODEL_NAME+".pth")
            running_loss = 0.0        


    #Test the current model
    with torch.no_grad():
        for i, return_value in enumerate(test_loader):
            observation, action, objects, targets = return_value
            observation = observation.to(device)
            action = action.to(device)
            objects = objects.to(device)
            targets= targets.to(device)
            #print(observation.device)

            #Forward
            action_pred, object_pred, target_pred = tasknet.forward(observation.float())
            #Loss is the summation of the Cross Entropies of Multihead output
            loss = criterion(action_pred, action)+criterion(object_pred,objects)+criterion(target_pred, targets)

            # print statistics
            running_loss += loss.item()
            if i % CHECKPOINT_AFTER == 0:    # print every 5 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / CHECKPOINT_AFTER:.3f}')
                writer.add_scalar("Task/test_lost", running_loss / CHECKPOINT_AFTER, epoch*len(data_loader)+i)
                running_loss = 0.0        
torch.save(tasknet.state_dict(), SAVED_MODELS_FOLDER+MODEL_NAME+".pth")
print('Finished Training')      
total_time = time.time()-start
writer.flush()
print("Example", tasknet.forward(dataset.__getitem__(10)[0].float().to(device)))
print("Total time:",total_time)

