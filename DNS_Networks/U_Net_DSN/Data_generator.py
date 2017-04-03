# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 20:41:09 2017

@author: yongjun.chen
"""
import numpy as np
import h5py
import random
import os.path

class Data_generator:
    def __init__(self,patch_shape,data_format):
        self.patch_shape = patch_shape
        self.data_format = data_format
        self.data_dir = '../data/'
        #self.data_dir = '/tempspace/tzeng/snmes3d/data/'
        
    def train_generator(self):
        #data_dir = '/Users/lei.cai/'
        data_file = 'snemi3d_train_full_stacks_v1.h5'
        data_files = [ 'snemi3d_train_full_stacks_v'+str(i)+'.h5' for i in range(1,2) ]
        d_files = []
        h5fs = []
        self.data_all = []
        self.label_all = []
        for i in range(len(data_files)):
            d_files.append(self.data_dir+data_files[i])
        for i in range(len(d_files)):
            h5fs.append(h5py.File(d_files[i],'r'))
            self.data_all.append(h5fs[i]['data'][:,:,0:60])
            self.label_all.append(h5fs[i]['label'][:,:,0:60])
    
        d_file = self.data_dir + data_file
        h5f = h5py.File(d_file,'r')
        
        self.data = h5f['data'][:,:,0:60]
        self.label = h5f['label'][:,:,0:60]
        
        s_patch = self.patch_shape[0]
        h_patch = self.patch_shape[1]
        w_patch = self.patch_shape[2]        
        
        h_input = self.data.shape[0]
        w_input = self.data.shape[1]
        s_input = self.data.shape[2]
        
        n_files = len(self.data_all)
        while True:
            w_start = random.randrange(0,w_input-w_patch)
            w_end = w_start+w_patch
            h_start = random.randrange(0,h_input-h_patch)
            h_end = h_start+h_patch
            s_start = random.randrange(0,s_input-s_patch)
            s_end = s_start+s_patch
            file_id = random.randrange(0,n_files)
            
            X = self.data_all[file_id][h_start:h_end,w_start:w_end,s_start:s_end]
            Y = self.label_all[file_id][h_start:h_end,w_start:w_end,s_start:s_end]
            X = np.transpose(X,(2,0,1))
            Y = np.transpose(Y,(2,0,1))
            if self.data_format  == 'NCHW':
                X = X.reshape(s_patch,1,h_patch,w_patch)
                Y = Y.reshape(s_patch,1,h_patch,w_patch)
            else:
                X = X.reshape(s_patch,h_patch,w_patch,1)
                Y = Y.reshape(s_patch,h_patch,w_patch,1)
            Y0 = Y
            Y1 = 1-Y0
            if self.data_format == 'NCHW':
                Y = np.concatenate((Y0,Y1),axis=1)
            else:
                Y = np.concatenate((Y0,Y1),axis=3)
            yield X,Y

    def valid_generator(self,Train,test_num):
        data_files = ['snemi3d_train_full_stacks_v1_cut/snemi3d_test_v16_w'+str(i)+'_h'+str(j)+'.h5' for i in range(1,6) for j in range(1,6)]
        d_files = []
        h5fs = []
        data_all = []
        label_all = []
        for i in range(len(data_files)):
            d_files.append(self.data_dir+data_files[i])
        for i in range(len(d_files)):
            h5fs.append(h5py.File(d_files[i],'r'))
            if Train == True:
                data_all.append(h5fs[i]['data'][:,:,60:80])
                label_all.append(h5fs[i]['label'][:,:,60:80])
            else:
                data_all.append(h5fs[i]['data'][:,:,80:100])
                label_all.append(h5fs[i]['label'][:,:,80:100])
        n_files = len(data_all)
        row = 224
        col = 224
        file_id = 0
        h_input = data_all[file_id].shape[0]
        w_input = data_all[file_id].shape[1]
        s_input = data_all[file_id].shape[2]

        while True:
            w_start = 0
            h_start = 0
            s_start = 0
            X = data_all[file_id][h_start:col,w_start:row,s_start:s_start+test_num]
            Y = label_all[file_id][h_start:col,w_start:row,s_start:s_start+test_num]
            X = np.transpose(X,(2,0,1))
            Y = np.transpose(Y,(2,0,1))
            if self.data_format  == 'NCHW':
                X = X.reshape(test_num,1,row,col)
                Y = Y.reshape(test_num,1,row,col)
            else:
                X = X.reshape(test_num,row,col,1)
                Y = Y.reshape(test_num,row,col,1)
            Y0 = Y
            Y1 = 1-Y0
            if self.data_format == 'NCHW':
                Y = np.concatenate((Y0,Y1),axis=1)
            else:
                Y = np.concatenate((Y0,Y1),axis=3)
            yield X,Y
            file_id+=1
            file_id = file_id % 25




            
            
        
        
            
