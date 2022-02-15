# Adapted from GrabNet

import numpy as np
import torch
from torch.utils import data
from torch.utils.data._utils.collate import default_collate
import os

import time
import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
to_cpu = lambda tensor: tensor.detach().cpu().numpy()

class LoadData(data.Dataset):
    def __init__(self,
                 dataset_dir,
                 ds_name='train',
                 dtype=torch.float32,
                 only_params = False,
                 load_on_ram = False):

        super(LoadData, self).__init__()

        self.only_params = only_params
        self.ds_path = os.path.join(dataset_dir, ds_name)
        self.ds = self._np2torch(os.path.join(self.ds_path, 'filename'))

        def _np2torch(self, ds_path):
            data = np.load(ds_path, allow_pickle=True)
            data_torch = {k:torch.tensor(data[k]) for k in data.files}
            return data_torch

        def __len__(self):
            k = list(self.ds.keys())[0]
            return self.ds[k].shape[0]

        def __getitem__(self, idx):

            return None