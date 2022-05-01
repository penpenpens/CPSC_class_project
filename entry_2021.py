#!/usr/bin/env python3

import numpy as np
import os
import sys

import wfdb
from utils import qrs_detect, comp_cosEn, save_dict
from DataAdapter import DataAdapter2
import torch.utils.data as Data
from model import CNN
import torch
from read_code import load_data,norm
from scipy.signal import butter,filtfilt
from score_2021 import RefInfo
import matplotlib.pyplot as plt
from scipy import interpolate
from utils import p_t_qrs

def move_windows(X,fs,win_n = 5):
    step = int(fs // 2)
    windows_length = win_n * fs
    res_X = []
    res_std = []
    if len(X) % windows_length > windows_length // 2:
        X = X.reshape(len(X),1)
        X = np.vstack((X,np.zeros((len(X) % windows_length,1))))
        X = np.squeeze(X)
    else:
        X = X[:-(len(X) % windows_length)]

    for i in range(0,len(X) - windows_length,step):
        tmp = X[i:i+windows_length]
        res_X.append(tmp)
        res_std.append(np.std(tmp))
        
    return res_X,res_std

def cal_cross_th(X):
    cross_th = 0
    for i in range(len(X)-1):
        if X[i] == 0 and X[i+1] == 1:
            cross_th += 1
        elif X[i] == 1 and X[i+1] == 0:
            cross_th += 1
    return cross_th

def plus_inhibition(X):
    patience = 10
    flag = 0
    count = 0
    for i in range(len(X)-1):
        if X[i] == 1 and X[i+1] == 0:
            count = 0
            flag = 1
        if flag == 1:
            count += 1
        if X[i] == 0 and X[i+1] == 1:
            if count < patience:
                X[i-count:i] = 1
            flag = 0

    return X

def forward_backward_search(X,end_points,qrs_pos):
    segment = []
    res_end_points = []
    th = 10
    for i in range(len(qrs_pos)-1):
        segment.append(X[int(qrs_pos[i]):int(qrs_pos[i+1])])
    for (starts,ends) in end_points:
        tmp = []
        loc = find_segment_loc(starts, qrs_pos)
        corr_list = []
        for i in range(-th,th + 1):
            st = min(max(0,loc + i - 1),len(segment)-1)
            en = max(0,min(len(segment)-1,loc + i))
            corr_list.append(cal_corr(segment[en],segment[st]))
        idx = check_index(corr_list)
        if idx == -1:
            continue
        tmp.append(qrs_pos[loc + idx])

        loc = find_segment_loc(ends, qrs_pos)
        corr_list = []
        for i in range(-th,th + 1):
            st = min(max(0,loc + i - 1),len(segment)-1)
            en = max(0,min(len(segment)-1,loc + i))
            corr_list.append(cal_corr(segment[en],segment[st]))
        idx = check_index(corr_list)
        if idx != -1:
            tmp.append(qrs_pos[loc + idx])
            res_end_points.append(tmp)

    return res_end_points

def check_index(corr_index):
    corr_th = 0.5
    middle = len(corr_index)//2
    if corr_index[middle] < corr_th:
        return 0
    for i in range(1,len(corr_index)//2 + 1):
        if corr_index[middle - i] < corr_th:
            return -i
        if corr_index[middle + i] < corr_th:
            return i
    return -1

def cal_corr(seg,lseg):
    if len(seg) != len(lseg):
        if len(seg) > len(lseg):
            lseg = seg # 保证lseg的长度大于seg的长度
        xline = np.linspace(0,len(seg)-1,len(seg))
        new_xline = np.linspace(0,len(seg)-1,len(lseg))
        f = interpolate.interp1d(xline, seg, kind = 'cubic')
        y_new = f(new_xline) # 插值之后的seg
        if len(y_new) > len(lseg):
            y_new = y_new[:len(lseg)]
        elif len(y_new) < len(lseg):
            lseg = lseg[:len(y_new)]
        corr = np.corrcoef(y_new,lseg)[0,1]
    else:
        corr = np.corrcoef(seg,lseg)[0,1]
    return corr

def find_segment_loc(l,qrs_pos):
    for i in range(len(qrs_pos)):
        if l < qrs_pos[i]:
            return i - 1
    return 0

def find_start_end(X,fs,sig_len):
    res = []
    tmp = []
    for i in range(len(X)-1):
        if X[i] == 0 and X[i+1] == 1:
            if (i/2+2.5)*fs >= sig_len:
                break
            else:
                tmp.append((i/2+2.5)*fs)
        elif X[i] == 1 and X[i+1] == 0:
            if (i/2+2.5)*fs >= sig_len:
                tmp.append(sig_len-1)
                res.append(tmp)
                break
            else:
                tmp.append((i/2+2.5)*fs)
            if len(tmp) == 2:
                res.append(tmp)
            tmp = []
    return res

def challenge_entry(sample_path):
    """
    This is a baseline method.
    """
    debug = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    folds = 5

    cnn = []
    for fold in range(folds):
        cnn_path = r'./model/CNN_best_model0_'+ str(fold) +'.pt'
        m = CNN()
        m.load_state_dict(torch.load(cnn_path,map_location='cuda:0'))
        m.eval()
        m.to(device)
        cnn.append(m)

    cnn2 = []
    for fold in range(folds):
        cnn_path = r'./model/CNN_best_model1_'+ str(fold) +'.pt'
        m = CNN()
        m.load_state_dict(torch.load(cnn_path,map_location='cuda:0'))
        m.eval()
        m.to(device)
        cnn2.append(m)
        
    cnn_extra = []
    for fold in range(folds):
        cnn_path = r'./extra_model/CNN_best_model0_'+ str(fold) +'.pt'
        m = CNN()
        m.load_state_dict(torch.load(cnn_path,map_location='cuda:0'))
        m.eval()
        m.to(device)
        cnn_extra.append(m)

    cnn2_extra = []
    for fold in range(folds):
        cnn_path = r'./extra_model/CNN_best_model1_'+ str(fold) +'.pt'
        m = CNN()
        m.load_state_dict(torch.load(cnn_path,map_location='cuda:0'))
        m.eval()
        m.to(device)
        cnn2_extra.append(m)

    [b,a] = butter(3,[0.5/100,40/100],'bandpass')
    sig, _, fs = load_data(sample_path)
    qrs_pos = p_t_qrs(sig[:,1],fs)
    sig1 = norm(filtfilt(b,a,sig[:, 0]))
    sig2 = norm(filtfilt(b,a,sig[:, 1]))
    end_points = []

    batch_size = 64
    res_X,_ = move_windows(sig1,fs)
    res_X2,_ = move_windows(sig2,fs)
    test_set = DataAdapter2(res_X, res_X2)
    test_loader = Data.DataLoader(test_set,batch_size = batch_size,shuffle = False,num_workers = 0)

    res = np.zeros(len(res_X))
    idx = 0
    for i,data in enumerate(test_loader,0):
        inputs,inputs2 = data[0].to(device),data[1].to(device)
        preds = []
        for fold in range(folds):
            cnn_outputs = cnn[fold](inputs)
            cnn2_outputs = cnn2[fold](inputs2)

            max_num1,pred1 = cnn_outputs.max(1)
            max_num2,pred2 = cnn2_outputs.max(1)
        
        
            pred = np.zeros(len(pred1))
            for j in range(len(pred1)):
                if pred1[j] == 0 and pred2[j] == 0:
                    pred[j] = 0
                elif pred1[j] == 1 and pred2[j] == 1:
                    pred[j] = 1
                else:
                    if max_num1[j] > max_num2[j]:
                        pred[j] = pred1[j].cpu().numpy()
                    else:
                        pred[j] = pred2[j].cpu().numpy()

            preds.append(pred)
            
        preds = np.stack(preds)
        max_pred = np.where(np.sum(preds,0) > folds//2,1,0)

        with torch.no_grad():
            res[idx:idx + batch_size] = max_pred
        idx += batch_size
    
    res = np.squeeze(res)
    
    if np.sum(np.where(res == 0,1,0)) > len(res) * 0.99:
        end_points = []
        predict_label = 0
    elif np.sum(np.where(res == 1,1,0)) > len(res) * 0.9:
        tmp = []
        tmp.append(0)
        tmp.append(len(sig)-1)
        end_points.append(tmp)
        predict_label = 1
    else:
        end_points = find_start_end(res,fs,len(sig))
        predict_label = 2
    
    cross_th =  cal_cross_th(res)
    if cross_th > 30:
        n_count = np.sum(np.where(res == 0,1,0))
        af_count = len(res) - n_count
        if n_count > af_count:
            end_points = []
            predict_label = 0
        else:
            tmp = []
            end_points = []
            tmp.append(0)
            tmp.append(len(sig)-1)
            end_points.append(tmp)
            predict_label = 1
            

    if end_points != []:
        res_points = []
        for (starts,ends) in end_points:
            if ends - starts > 5 * fs:
                res_points.append([starts,ends])
        end_points = res_points
        if end_points == []:
            n_count = np.sum(np.where(res == 0,1,0))
            af_count = len(res) - n_count
            if n_count > af_count:
                end_points = []
                predict_label = 0
            else:
                tmp = []
                end_points = []
                tmp.append(0)
                tmp.append(len(sig)-1)
                end_points.append(tmp)
                predict_label = 1
    
    
    
    predict_label_extra = 0
    if predict_label == 2:
        new_end = []
        res = np.zeros(len(res_X))
        idx = 0
        for i,data in enumerate(test_loader,0):
            inputs,inputs2 = data[0].to(device),data[1].to(device)
            preds = []
            for fold in range(folds):
                cnn_outputs = cnn_extra[fold](inputs)
                cnn2_outputs = cnn2_extra[fold](inputs2)

                max_num1,pred1 = cnn_outputs.max(1)
                max_num2,pred2 = cnn2_outputs.max(1)


                pred = np.zeros(len(pred1))
                for j in range(len(pred1)):
                    if pred1[j] == 0 and pred2[j] == 0:
                        pred[j] = 0
                    elif pred1[j] == 1 and pred2[j] == 1:
                        pred[j] = 1
                    else:
                        if max_num1[j] > max_num2[j]:
                            pred[j] = pred1[j].cpu().numpy()
                        else:
                            pred[j] = pred2[j].cpu().numpy()

                preds.append(pred)

            preds = np.stack(preds)
            max_pred = np.where(np.sum(preds,0) > folds//2,1,0)

            with torch.no_grad():
                res[idx:idx + batch_size] = max_pred
            idx += batch_size

        res = np.squeeze(res)

        if np.sum(np.where(res == 0,1,0)) > len(res) * 0.99:
            new_end = []
            predict_label_extra = 0
        elif np.sum(np.where(res == 1,1,0)) > len(res) * 0.9:
            tmp = []
            tmp.append(0)
            tmp.append(len(sig)-1)
            new_end.append(tmp)
            predict_label_extra = 1
        else:
            new_end = find_start_end(res,fs,len(sig))
            predict_label_extra = 2

        cross_th = cal_cross_th(res)
        if cross_th > 30:
            n_count = np.sum(np.where(res == 0,1,0))
            af_count = len(res) - n_count
            if n_count > af_count:
                new_end = []
                predict_label_extra = 0
            else:
                tmp = []
                new_end = []
                tmp.append(0)
                tmp.append(len(sig)-1)
                new_end.append(tmp)
                predict_label_extra = 1


        if new_end != []:
            res_points = []
            for (starts,ends) in new_end:
                if ends - starts > 5 * fs:
                    res_points.append([starts,ends])
            new_end = res_points
            if new_end == []:
                n_count = np.sum(np.where(res == 0,1,0))
                af_count = len(res) - n_count
                if n_count > af_count:
                    new_end = []
                    predict_label_extra = 0
                else:
                    tmp = []
                    new_end = []
                    tmp.append(0)
                    tmp.append(len(sig)-1)
                    new_end.append(tmp)
                    predict_label_extra = 1
    
    
    
    if predict_label_extra == 2:
        if new_end != []:
            end_points = new_end
    

    pred_dcit = {'predict_endpoints': end_points}

    return pred_dcit


if __name__ == '__main__':
    DATA_PATH = sys.argv[1]
    RESULT_PATH = sys.argv[2]
    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)
    
    test_set = os.listdir(DATA_PATH)
    # test_set = open(os.path.join(RECORDS_PATH, 'RECORDS'), 'r').read().splitlines()
    for i, sample in enumerate(test_set):
        if ".dat" in sample:
            sample = os.path.splitext(sample)[0]
            print(sample)
            sample_path = os.path.join(DATA_PATH, sample)
            pred_dict = challenge_entry(sample_path)

            save_dict(os.path.join(RESULT_PATH, sample+'.json'), pred_dict)
