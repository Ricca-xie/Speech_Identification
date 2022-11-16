import numpy as np
import torch
import torch.nn as nn
from torch.utils import data


class SpeechDataset(data.Dataset):

    def __init__(
                self,
                data_path,
                label_path,
                transform = None,
                data_length = 125995,
                filters = 161):
        # test 13755
        # dev 13766
        self.transform = transform
        self.data = np.memmap(data_path, dtype=np.float64, mode='r', shape=(data_length, 3, 1, 300, filters))
        self.label = np.memmap(label_path, dtype=np.int32, mode='r', shape=(data_length, 1251))

        # self.data2 = np.memmap(data_path2, dtype=np.float32, mode='r', shape=(13766, 3, 1, 300, filters))
        # self.label2 = np.memmap(label_path2, dtype=np.int32, mode='r', shape=(13766, 1251))
        # import pdb;pdb.set_trace()
        # self.data = np.concatenate((self.data1,self.data2), axis=0)
        # self.label = np.concatenate((self.label1, self.label2), axis=0)
        # self.data = np.vstack((self.data1,self.data2))
        # self.label = np.vstack((self.label1,self.label2))
        #self.data_path = data_path_list
        #self.label_path = label_path_list
        #self.data_length = data_length
        # self.filters = filters

    def __getitem__(self,index):
        data = torch.FloatTensor(self.data[index])
        label = torch.LongTensor(self.label[index])
        #data_path = self.data_path[index]
        #label_path = self.label_path[index]

        #data = np.memmap(data_path, dtype=np.float64, mode='r', shape=(self.data_length, 3, 1, 300, self.filters))
        return data, label

    def __len__(self):
        return len(self.data)

def create_loader(
        dataset,
        batch_size,
        is_training=False,
):
    collate_fn = torch.utils.data.dataloader.default_collate
    loader_class = torch.utils.data.DataLoader
    loader = loader_class(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=0,
        pin_memory= False,
        drop_last=False,
        collate_fn = collate_fn,
    )

    return loader
