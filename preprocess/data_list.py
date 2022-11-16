import os
import numpy as np
import pandas as pd
import config as c
import random

# path = "I:"+os.sep+"data_thchs30"+os.sep+"data_thchs30"+os.sep+"27class"
# path = "G:"+os.sep+"921people"
# path = "I:"+os.sep+"voxceleb1"+os.sep+"vox1_dev_wav"+os.sep+"wav"
# path = "/media/yn/419/voxceleb1/vox1_dev_wav/wav"
# PATH = "/media/yn/419/voxceleb1/vox1_dev_wav/wav"
# dev_path = "../dataset/wav_dev"
# test_path = "../dataset/wav_test"
path =  "../dataset/all_wav/wav"
PATH_I = "../wav"


train_path = "../dataset/all_wav/DSS_1/train.csv"
test_path = "../dataset/all_wav/DSS_1/test.csv"
dev_path = "../dataset/all_wav/DSS_1/dev.csv"

# train_path2 = "../dataset/all_wav/DSS_2_82/train.csv"
# test_path2 = "../dataset/all_wav/DSS_2_82/test.csv"
train_path2 = "../dataset/all_wav/DSS_2_8515/train.csv"
test_path2 = "../dataset/all_wav/DSS_2_8515/test.csv"
# dev_path2 = "../dataset/all_wav/DSS_2_8515/dev.csv"

def generate_valid_data_list_2(path): # with audios from different videos
    filelist = os.listdir(path)

    # import pdb; pdb.set_trace()
    count = list()

    for i in filelist:
        utlist = os.listdir(path+os.sep+i)
        n = 0
        for j in utlist:
            n += len(os.listdir(path+os.sep+i+os.sep+j))
        count.append([i,n])
        print(i,n)
    count = pd.DataFrame(np.array(count),columns=["id","num"])
    # 每个id对应多少个wav文件
    # num = np.array([int(i) for i in count['num']])
    num = count['num'].astype(int).values
    # print(num)
    print("mean:",np.mean(num))
    print("max: ",max(num))
    print("min: ",min(num))
    # import pdb; pdb.set_trace()
    valid = count.loc[count['num'].astype(int) >= c.total_len]['id'].tolist()
    print("valid:" ,len(set(valid)))
    print('filelist:', len(filelist))
    # import pdb;pdb.set_trace()
    return valid,filelist

def statistic(filepath):

	filelist = os.listdir(filepath)
	speaker_num = len(filelist)
	video_num = 0
	utterance_num = 0
	for file in filelist:
		video = os.listdir(os.path.join(filepath,file))
		video_num += len(video)
		for v in video:
			utterance = os.listdir(os.path.join(filepath,file,v))
			utterance_num += len(utterance)

	print('speaker_num', speaker_num)
	print('video_num', video_num)
	print('utterance_num',utterance_num)





def generate_data_list_2_bynumber(path,valid,filelist):
    test = pd.DataFrame(columns=["filepath","id","name"])
    train = pd.DataFrame(columns=["filepath","id","name"])
    m = 0
    n = 0
    for i in filelist: #id
        utlist=os.listdir(path+os.sep+i)
        if i not in valid:
           continue
        k = 0
        count_train = 0
        count_test = 0
        for j in utlist: #video
            aulist = os.listdir(path+os.sep+i+os.sep+j)
            for z in aulist: #utterance
                # import pdb; pdb.set_trace()
                if k >= c.train_len and k < c.total_len:
                    test.loc[m]=np.array([path+os.sep+i+os.sep+j+os.sep+z,i,i+'_'+str(count_test)+'_test'])
                    m += 1; count_test += 1; k += 1
                elif k >= c.total_len:
                    continue
                else:
                    train.loc[n]=np.array([path+os.sep+i+os.sep+j+os.sep+z,i,i+'_'+str(count_train)+'_train'])
                    k += 1; n += 1; count_train += 1
        print(i,str(filelist.index(i))+"/"+str(len(filelist)))

    # train.to_csv(train_path2)
    # test.to_csv(test_path2)
    print("train:" + str(train.shape))
    print("test:" + str(test.shape))

def generate_data_list_2_byvideo(path,filelist):
    test = pd.DataFrame(columns=["filepath","id","name"])
    train = pd.DataFrame(columns=["filepath","id","name"])
    m = 0
    n = 0
    # import pdb;pdb.set_trace()
    for i in filelist: #id
        utlist=os.listdir(path+os.sep+i)
        random.shuffle(utlist)
        add_to_test = True
        count_train = 0
        count_test = 0
        for j in utlist: #video
            aulist = os.listdir(path+os.sep+i+os.sep+j) #utterance
            if len(aulist) >= 5 and add_to_test: 
                for z in aulist:
                    test.loc[m]=np.array([path+os.sep+i+os.sep+j+os.sep+z,i,i+'_'+str(count_test)+'_test'])
                    m += 1; count_test += 1
                add_to_test = False
                print("success add to test:"+str(i))
                print(len(aulist))
            else:
                for z in aulist: #utterance
                    train.loc[n]=np.array([path+os.sep+i+os.sep+j+os.sep+z,i,i+'_'+str(count_train)+'_train'])
                    n += 1; count_train += 1
        print(i,str(filelist.index(i))+"/"+str(len(filelist)))
    # train.to_csv(train_path)
    # test.to_csv(test_path)
    print("train:" + str(train.shape))
    print("test:" + str(test.shape))

def generate_data_list_2_byratio(path,filelist):
    train = pd.DataFrame(columns=["filepath","id","name"])
    test = pd.DataFrame(columns=["filepath","id","name"])
    m = 0
    n = 0
    for i in filelist: #id
        utlist=os.listdir(path+os.sep+i)
        count_train = 0
        count_test = 0
        for j in utlist: #video
            aulist = os.listdir(path+os.sep+i+os.sep+j)
            k = 0
            for z in aulist: #utterance
                # import pdb; pdb.set_trace()
                if k >= len(aulist)*0.85:
                    test.loc[m]=np.array([path+os.sep+i+os.sep+j+os.sep+z,i,i+'_'+str(count_test)+'_test'])
                    m += 1; count_test += 1; k += 1
                else:
                    train.loc[n]=np.array([path+os.sep+i+os.sep+j+os.sep+z,i,i+'_'+str(count_train)+'_train'])
                    k += 1; n += 1; count_train += 1
        print(i,str(filelist.index(i))+"/"+str(len(filelist)))
    # train.to_csv(train_path2)
    # test.to_csv(test_path2)
    print("train:" + str(train.shape))
    print("test:" + str(test.shape))

def generate_data_list_3(filepath):
    data = pd.read_csv(filepath,sep=" ",header=None, names = ["type","path"])
    # import pdb; pdb.set_trace()
    data['id'] = pd.Series(data.apply(lambda x: x["path"].split("/")[0], axis=1), index=data.index)
    data['class'] = pd.Series(data.apply(lambda x: int(x['id'].split("d")[1])-10001, axis=1), index=data.index)
    data['name'] = pd.Series(data.apply(lambda x: x["path"].split(".")[0].replace("/","_"),axis=1), index=data.index)
    data['path'] = pd.Series(data.apply(lambda x: PATH + "/" + x["path"], axis=1), index=data.index)
    
    data = data.rename(columns={'path':'filepath'})
    test = data[data["type"] == 3]
    train = data[data["type"] == 1]
    dev = data[data["type"] == 2]

    # train.to_csv(train_path)
    # test.to_csv(test_path)
    # dev.to_csv(dev_path)

    print("train:" + str(train.shape))
    print("test:" + str(test.shape))
    print("dev:" + str(dev.shape))
    # import pdb; pdb.set_trace()

def main():
    # statistic(dev_path)
    # statistic(test_path)
    valid, filelist = generate_valid_data_list_2(path)

    # import pdb; pdb.set_trace()

    # generate_data_list_2_byratio(path,filelist)
    # generate_data_list_3("iden_split.txt")

    generate_data_list_2_byvideo(path, filelist)

if __name__ == '__main__':
    main()