import os
import numpy as np
from score_2021 import RefInfo
from scipy.signal import filtfilt,butter
import wfdb
import random
from model import CNN
import torch
import pandas as pd
from scipy.io import loadmat
from scipy.interpolate import interp1d
from tools.ssqueezepy.synsq_cwt import synsq_cwt_fwd

def norm(x):
    min_ = np.min(x)
    max_ = np.max(x)
    x = (x - min_)/(max_ - min_)
    return x

def load_data(sample_path):
    sig, fields = wfdb.rdsamp(sample_path)
    length = len(sig)
    fs = fields['fs']

    return sig, length, fs

def get_signal(path,channel = 1):
    '''
    Description:
        读取信号并获取采样频率，信号长度，心拍位置，房颤开始位置，房颤结束位置，类别等
    Params:
        path 读取数据的路径
    Return:
        res 该结构是一个列表，每个元素代表一个样本，样本的结构为一个字典，字典中包含名称，
            信号，采样频率，信号长度，心拍位置，房颤开始位置，房颤结束位置，类别等信息
    '''
    res = []

    for file in os.listdir(path):
        sample = {'name':'','sig':0,'fs':0,'len_sig':0,'beat_loc':0,'af_starts':0,'af_ends':0,'class_true':0}
        if file.endswith('.hea'):
            name = file.split('.')[0]
            sample['name'] = name
            sig,_,_ = load_data(os.path.join(path,name))
            if sig.shape[1] < sig.shape[0]:
                sig = np.transpose(sig)
            sample['sig'] = sig[channel,:]
            ref = RefInfo(os.path.join(path,name))
            sample['fs'] = ref.fs
            sample['len_sig'] = ref.len_sig
            sample['beat_loc'] = ref.beat_loc
            sample['af_starts'] = ref.af_starts
            sample['af_ends'] = ref.af_ends
            sample['class_true'] = ref.class_true
            res.append(sample)
    print('文件读取完成')
    return res

def select_data(res,record_path):
    tmp_set = open(record_path, 'r').read().splitlines()
    dataset = []
    for samp in res:
        name = samp['name']
        if name in tmp_set:
            dataset.append(samp)
    return dataset

def gen_sample(res,seed = 0,train_rate = 0.7,valid_rate = 0.1):
    random.seed(seed)
    index = list(range(105))
    random.shuffle(index)
    train_size = int(len(index) * train_rate)
    valid_size = int(len(index) * valid_rate)
    train_index = index[:train_size]
    valid_index = index[train_size:train_size + valid_size]
    test_index = index[train_size + valid_size:]
    train_samp = []
    valid_samp = []
    test_samp = []
    for samp in res:
        name = int(samp['name'].split('_')[1])
        if name in train_index:
            train_samp.append(samp)
        elif name in valid_index:
            valid_samp.append(samp)
        else:
            test_samp.append(samp)
    print('训练集样本：',len(train_index))
    print('验证集样本：',len(valid_index))
    print('测试集样本：',len(test_index))
    return train_samp,valid_samp,test_samp

def gen_ensemble_samp(res,n_part = 5,fold = 0,seed = 0, test_rate = 0.2):
    random.seed(seed)
    index = list(range(105))
    random.shuffle(index)
    train_size = int(len(index) * (1 - test_rate))
    test_size = train_size // n_part
    train_index = index[:train_size]
    test_index = index[train_size:]
    valid_index = train_index[fold * test_size:(fold + 1) * test_size]
    train_index = list(set(train_index) - set(valid_index))
    train_samp = []
    valid_samp = []
    test_samp = []
    for samp in res:
        name = int(samp['name'].split('_')[1])
        if name in train_index:
            train_samp.append(samp)
        elif name in valid_index:
            valid_samp.append(samp)
        else:
            test_samp.append(samp)
    return train_samp,valid_samp,test_samp

def data_enhance(sig,label,win_len,step):
    sig_len = len(sig)
    res_sig = []
    res_label = []
    for ii in range(0,sig_len-win_len,step):
        tmp = label[ii:ii+win_len]
        if np.sum(tmp) > len(tmp)//2:
            res_label.append(1)
            res_sig.append(sig[ii:ii+win_len])
        else:
            res_label.append(0)
            res_sig.append(sig[ii:ii+win_len])
        
    return res_sig,res_label

def get_cnn_featrue(X):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cnn0 = CNN()
    cnn0.load_state_dict(torch.load(r'.\model\CNN_best_model0.pt',map_location='cuda:0'))
    cnn0.eval()
    cnn0.to(device)

    cnn1 = CNN()
    cnn1.load_state_dict(torch.load(r'.\model\CNN_best_model1.pt',map_location='cuda:0'))
    cnn1.eval()
    cnn1.to(device)

    res_X = []
    for x in X:
        with torch.no_grad():
            x = torch.FloatTensor(x).to(device)
            _,y0 = cnn0(x)
            _,y1 = cnn1(x)
            y = torch.cat((y0,y1),-1)
            res_X.append(y.cpu().numpy())
    return res_X


def gen_cnn_X_Y(res,n_samp = 10,n_rate = 1,af_rate = 1):
    res_X = []
    res_Y = []
    [b,a] = butter(3,[0.5/100,40/100],'bandpass')
    for samp in res:
        class_true = int(samp['class_true'])
        sig = norm(filtfilt(b,a,samp['sig']))
        fs = samp['fs']
        sig_len = len(sig)
        if sig_len < 2*n_samp*fs:
            continue
        
        if class_true == 0:
            square_wave = np.zeros(sig_len)
            nEN = sig_len//int(n_rate * n_samp)
            tmp_X,tmp_Y = data_enhance(sig,square_wave,5*fs,nEN)
            res_X.extend(tmp_X)
            res_Y.extend(tmp_Y)
        elif class_true == 1:
            square_wave = np.ones(sig_len)
            nEN = sig_len//int(af_rate * n_samp)
            tmp_X,tmp_Y = data_enhance(sig,square_wave,5*fs,nEN)
            res_X.extend(tmp_X)
            res_Y.extend(tmp_Y)
        else:
            af_start = samp['af_starts']
            af_end = samp['af_ends']
            beat_loc = samp['beat_loc']
            square_wave = np.zeros(sig_len)
            for j in range(len(af_start)):
                square_wave[int(beat_loc[int(af_start[j])]):int(beat_loc[int(af_end[j])])] = 1
            tmp_X,tmp_Y = data_enhance(sig,square_wave,5*fs,fs)
            AF_index = np.where(np.array(tmp_Y) == 1,True,False)
            Normal_index = np.where(np.array(tmp_Y) == 0,True,False)
            AF_X = np.array(tmp_X)[AF_index,:]
            AF_Y = np.array(tmp_Y)[AF_index]
            N_X = np.array(tmp_X)[Normal_index,:]
            N_Y = np.array(tmp_Y)[Normal_index]
            nAF = len(AF_Y)//n_samp
            nN = len(N_Y)//n_samp
            if nAF == 0 or nN ==0:
                continue
            for n in range(0,len(AF_Y),nAF):
                res_X.append(AF_X[n,:])
                res_Y.append(AF_Y[n])
            for n in range(0,len(N_Y),nN):
                res_X.append(N_X[n,:])
                res_Y.append(N_Y[n])

    return res_X,res_Y

def gen_extra_cnn_X_Y(res,n_samp = 10,n_rate = 1,af_rate = 1):
    res_X = []
    res_Y = []
    [b,a] = butter(3,[0.5/100,40/100],'bandpass')
    for samp in res:
        class_true = int(samp['class_true'])
        sig = norm(filtfilt(b,a,samp['sig']))
        fs = samp['fs']
        sig_len = len(sig)
        if sig_len < 2*n_samp*fs:
            continue
        
        if class_true == 2:
            af_start = samp['af_starts']
            af_end = samp['af_ends']
            beat_loc = samp['beat_loc']
            square_wave = np.zeros(sig_len)
            for j in range(len(af_start)):
                square_wave[int(beat_loc[int(af_start[j])]):int(beat_loc[int(af_end[j])])] = 1
            tmp_X,tmp_Y = data_enhance(sig,square_wave,5*fs,fs)
            AF_index = np.where(np.array(tmp_Y) == 1,True,False)
            Normal_index = np.where(np.array(tmp_Y) == 0,True,False)
            AF_X = np.array(tmp_X)[AF_index,:]
            AF_Y = np.array(tmp_Y)[AF_index]
            N_X = np.array(tmp_X)[Normal_index,:]
            N_Y = np.array(tmp_Y)[Normal_index]
            nAF = len(AF_Y)*5//n_samp
            nN = len(N_Y)*2//n_samp
            if nAF == 0 :
                nAF = 1
            if nN == 0 :
                nN = 1 
            for n in range(0,len(AF_Y),nAF):
                res_X.append(AF_X[n,:])
                res_Y.append(AF_Y[n])
            for n in range(0,len(N_Y),nN):
                res_X.append(N_X[n,:])
                res_Y.append(N_Y[n])

    return res_X,res_Y



def read_physionet_header(file_path):
    with open(file_path) as f:
        context = f.readlines()
    label = -1
    for idx in range(len(context)):
        line = context[idx].strip().split(' ')
        if idx == 0:
            num_leads = line[1]
            fs = line[2]
            sig_len = line[3]
        elif idx == 1:
            data_name = line[0]
            adc = line[2].split('/')[0]
            baseline = line[4]
        elif idx == 15:
            tag = line[-1].split(',')
            if '164889003' in tag:
                label = 1
            elif '426783006' in tag:
                label = 0
            else:
                label = -1
    return data_name,float(fs),float(adc),int(sig_len),label,num_leads,float(baseline)

def load_wavelet_data(idx,save_path):
    try:
        tmp = np.load(os.path.join(save_path,idx))
        data = tmp['data']
        label = tmp['label']
    except:
        print('文件{}不存在'.format(idx))
    return data,label


if __name__ == '__main__':
    data_path = r'C:\Users\yurui\Desktop\item\cpsc\data\all_data'
    res = get_signal(data_path,0)
    X,Y = gen_cnn_X_Y(res,50,af_rate = 2)
    print(np.array(X).shape)
    print('1:',np.sum(np.where(np.array(Y)==1,1,0)))
    print('0:',np.sum(np.where(np.array(Y)==0,1,0)))
