from torch.utils import data
from gil.data_processing.h5df_dataset_loader import HDF5Dataset
import time
num_epochs = 1
loader_params = {'batch_size': 100, 'shuffle': True, 'num_workers': 1}
DATASET_FOLDER = "/home/nkquynh/gil_ws/gil_hierachial_hmp/hierarchical-hmp/gil/datasets/gil_dataset/set_table/without_human30_04_2022_17_59_40/task_data"
dataset = HDF5Dataset(DATASET_FOLDER, recursive=True, load_data=False, 
   data_cache_size=1000, transform=None)

print("dataset length", dataset.__len__())

data_loader = data.DataLoader(dataset, **loader_params)

for i in range(num_epochs):
   for data_value in data_loader:
      print(data_value[0].shape)
      print(data_value[0][0])
      break