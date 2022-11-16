import threading as td
import numpy as np
import os 
import time
import config as c
import pandas as pd
from prepare_data import extract_hpss_features_sg_pthread,files_list,extract_hpss_features_pthread,files_list
from sklearn.preprocessing import OneHotEncoder

CSV_ROOT = '../dataset/all_wav/DSS_1'
FILEPATH = 'concate-I/'
CLASSES = 1251
filters = 64
# filters = 161
layers = 3

THREAD_NUM = 20
DATANAME = 'mel-n'
#DATANAME = 'spec-n'
PATH_TRAIN = "all_train.csv"
# PATH_DEV = "dev.csv"
PATH_TEST = "test.csv"


def path_generator(label, state):
    if os.path.isdir(FILEPATH + "/" + DATANAME + "/") == False:
        os.makedirs(FILEPATH + "/" + DATANAME + "/")
    return FILEPATH + "/" + DATANAME + "/" + label + state + '_' + DATANAME + '.myarray'


def file_len(path):
    file = pd.read_csv(path)
    return len(file)

class myThread(td.Thread):
    def __init__(self,start_point,end_point,wav_dir, mode, X, Y, i):
        td.Thread.__init__(self)
        self.start_point = int(start_point)
        self.end_point = int(end_point)
        self.wav_dir = wav_dir
        self.mode = mode
        self.X = X
        self.Y = Y
        self.i = i
    
    def run(self):
        # extract_hpss_features_sg_pthread(self.wav_dir,self.start_point,self.end_point,self.mode, self.X, self.Y, self.i)
        extract_hpss_features_pthread(self.wav_dir, self.start_point, self.end_point, self.mode, self.X, self.Y, self.i)


#Optimize here, the termination has problem
def pthread_prepare(filepath, mode, X, Y):
    ls,cl = files_list(filepath)
    length = len(ls)
    part_len = int(length / THREAD_NUM)
    for i in range(THREAD_NUM):
        if i == THREAD_NUM-1:
            # length - part_len * (THREAD_NUM-1)
            thread = myThread(i*part_len, i*part_len + (length - part_len * (THREAD_NUM-1)), filepath, mode, X, Y, i)
        else:
            thread = myThread(i*part_len, (i+1)*part_len, filepath, mode, X, Y, i)
        thread.start()
        print("initialization succeed")
    thread.join()
    return

def main():
    start_time = time.time()
    X_train = np.memmap(path_generator('X_','train'), dtype=np.float64, mode='w+', shape=(file_len(os.path.join(CSV_ROOT,PATH_TRAIN)),layers,300,filters,1))
    Y_train = np.memmap(path_generator('Y_','train'), dtype=np.int32, mode='w+', shape=(file_len(os.path.join(CSV_ROOT,PATH_TRAIN)),CLASSES))
    # X_dev = np.memmap(path_generator('X_','dev'), dtype=np.float64, mode='w+', shape=(file_len(PATH_DEV),layers,300,filters,1))
    # Y_dev = np.memmap(path_generator('Y_','dev'), dtype=np.int32, mode='w+', shape=(file_len(PATH_DEV),CLASSES))
    X_test = np.memmap(path_generator('X_','test'), dtype=np.float64, mode='w+', shape=(file_len(os.path.join(CSV_ROOT,PATH_TEST)),layers,300,filters,1))
    Y_test = np.memmap(path_generator('Y_','test'), dtype=np.int32, mode='w+', shape=(file_len(os.path.join(CSV_ROOT,PATH_TEST)),CLASSES))
    pthread_prepare(os.path.join(CSV_ROOT,PATH_TRAIN), 3, X_train, Y_train)
    pthread_prepare(os.path.join(CSV_ROOT,PATH_TEST), 3, X_test, Y_test)
    # pthread_prepare(PATH_DEV, 3, X_dev, Y_dev)
    end_time = time.time()
    print(start_time-end_time)

if __name__ == "__main__":
    main()
    # print(file_len(PATH_TRAIN), file_len(PATH_TEST), file_len(PATH_DEV))
