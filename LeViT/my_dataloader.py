import numpy as np
import os
import torch
import random

def data_loader(root_path, data_file_name, label_file_name, X_shape, Y_shape, batch_size=12, type = 'spec-n', is_append=False,is_training = False):
    data_path = os.path.join(root_path, data_file_name)
    label_path = os.path.join(root_path, label_file_name)

    # print(data_path, label_path)
    # import pdb;pdb.set_trace()
    data = np.memmap(data_path, dtype=np.float64, mode='r', shape=X_shape)
    label = np.memmap(label_path, dtype=np.int32, mode='r', shape=Y_shape)

    
    # data = torch.FloatTensor(np.memmap(data_path, dtype=np.float64, mode='r', shape=(13766, 3, 1, 300, 161))).to(device)
    # label = torch.LongTensor(np.memmap(label_path, dtype=np.int32, mode='r', shape=Y_shape)).to(device)
    # print('finish loading data...')
    dataset = []
    for X, Y in zip(data, label):
        dataset.append((X, Y))
    # print('finish creating dataset...')
    if is_append:
        print('is_training...')
        data_path2 = os.path.join(root_path, 'X_dev_'+type+'.myarray')
        label_path2 = os.path.join(root_path, 'Y_dev_'+type+'.myarray')
        data2 = np.memmap(data_path2, dtype=np.float64, mode='r', shape=(13766, 3, 1, 300, 161))
        label2 = np.memmap(label_path2, dtype=np.int32, mode='r', shape=(13766, 1251))
        for X, Y in zip(data2, label2):
            dataset.append((X, Y))

    if is_training:
        print('is training...')
        random.shuffle(dataset)
    print('finish shuffle...')
    for idx, value in enumerate(dataset):
        if idx == 0:
            train_data, label = torch.FloatTensor(value[0]).unsqueeze(0), torch.LongTensor(value[1]).unsqueeze(0)
            # print(train_data.shape)
        elif idx % batch_size == 0:
            yield (train_data, label)
            train_data, label = torch.FloatTensor(value[0]).unsqueeze(0), torch.LongTensor(value[1]).unsqueeze(0)
        else:
            train_data = torch.cat((train_data, torch.FloatTensor(value[0]).unsqueeze(0)), 0)
            label = torch.cat((label, torch.LongTensor(value[1]).unsqueeze(0)), 0)