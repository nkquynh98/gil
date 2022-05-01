import h5py
#import helpers
import numpy as np
from pathlib import Path
import torch
from torch.utils import data

import h5py
#import helpers
import numpy as np
from pathlib import Path
import random
import torch
from torch.utils import data

class HDF5Dataset(data.Dataset):
    """Represents an abstract HDF5 dataset.
    
    Input params:
        file_path: Path to the folder containing the dataset (one or multiple HDF5 files).
        recursive: If True, searches for h5 files in subdirectories.
        load_data: If True, loads all the data immediately into RAM. Use this if
            the dataset is fits into memory. Otherwise, leave this at false and 
            the data will load lazily.
        data_cache_size: Number of HDF5 files that can be cached in the cache (default=3).
        transform: PyTorch transform to apply to every data instance (default=None).
    """
    def __init__(self, file_path, recursive, load_data, data_cache_size=3, transform=None):
        super().__init__()
        self.data_info = []
        self.data_infos = {}
        self.data_cache = {}
        self.data_cache_size = data_cache_size
        self.transform = transform
        self.data_length = 0
        self.recursive = recursive
        self.load_data = load_data
        
        self.fetch_dataset(file_path=file_path)
    def fetch_dataset(self, file_path):
        # Search for all h5 files
        self.data_infos = {}
        p = Path(file_path)
        assert(p.is_dir())
        if self.recursive:
            files = sorted(p.glob('**/*.h5'))
        else:
            files = sorted(p.glob('*.h5'))
        if len(files) < 1:
            raise RuntimeError('No hdf5 datasets found')
        current_index = 0
        for h5dataset_fp in files:
            new_index = self._add_data_infos(str(h5dataset_fp.resolve()), self.load_data, current_index)
            
            current_index=new_index
        self.data_length = current_index
        #print("data info len" ,self.data_infos)
    def __getitem__(self, index):

        file_name = self._get_file_name_from_idx(index)
        assert file_name is not None, "No valid file can be found for this index"
        if file_name not in self.data_cache.keys():
            #If the file is not loaded to the cache, load it to the current cache
            self._load_data(file_name)
        index_in_dataset = index - self.data_infos[file_name]["start_idx"]
        return_values = []
        for name, dataset in self.data_cache[file_name].items():
            return_values.append(dataset[index_in_dataset])
        # # get data
        # x = self.get_data("data", index)
        # if self.transform:
        #     x = self.transform(x)
        # else:
        #     x = torch.from_numpy(x)

        # # get label
        # y = self.get_data("label", index)
        # y = torch.from_numpy(y)
        return return_values

    def __len__(self):
        return self.data_length

    def _add_data_infos(self, file_path, load_data, start_idx=0):
        with h5py.File(file_path) as h5_file:
            # Walk through all groups, extracting datasets
            assert h5_file.attrs["data_length"]
            ds_len = h5_file.attrs["data_length"]
            end_idx = start_idx+ds_len
            self.data_infos[file_path] = {"start_idx": start_idx, "end_idx": end_idx-1}
            if load_data:
                self._load_data(file_path)
            # for dname, ds in h5_file.items():
            #     print(dname)
            #     idx = -1
            #     ds_len = ds.shape[0]
            #     #print(ds_len)
            #     if load_data:
            #         # add data to the data cache
            #         idx = self._add_to_cache(ds[:], file_path)
            #     end_idx = start_idx+ds_len
            
            #     self.data_info.append({'file_path': file_path, 'type': dname, 'shape': ds.shape, "start_idx": start_idx, "end_idx": end_idx-1,'cache_idx': idx})

        return end_idx
    
    def _get_file_name_from_idx(self, idx):
        file_name = None
        for key,value in self.data_infos.items():
            if idx<=value["end_idx"] and idx>=value["start_idx"]:
                file_name = key
        return file_name

    def _load_data(self, file_path):
        """Load data to the cache given the file
        path and update the cache index in the
        data_info structure.
        """
        while len(self.data_cache) >= self.data_cache_size:
            print("overload")
            #remove one item from the cache at random
            removal_keys = list(self.data_cache.keys())
            removal_key = random.choice(removal_keys)
            self.data_cache.pop(removal_key)

        with h5py.File(file_path, "r") as h5_file:
            self.data_cache[file_path] = {}
            for dataset_name, dataset_value in h5_file.items():
                #print(dataset_name)
                self.data_cache[file_path][dataset_name] = torch.from_numpy(dataset_value[:])
            # for gname, group in h5_file.items():
            #     for dname, ds in group.items():
            #         # add data to the data cache and retrieve
            #         # the cache index
            #         idx = self._add_to_cache(ds[:], file_path)
            #         # find the beginning index of the hdf5 file we are looking for
            #         file_idx = next(i for i,v in enumerate(self.data_info) if v['file_path'] == file_path)
            #         print("file_idx", file_idx)
            #         # the data info should have the same index since we loaded it in the same way
            #         self.data_info[file_idx + idx]['cache_idx'] = idx
        # remove an element from data cache if size was exceeded
        #print(len(self.data_cache))

            