# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 10:41:09 2017

@author: lei.cai
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
            self.data_all.append(h5fs[i]['data'][:,:,0:80])
            self.label_all.append(h5fs[i]['label'][:,:,0:80])
    
        d_file = self.data_dir + data_file
        h5f = h5py.File(d_file,'r')
        
        self.data = h5f['data'][:,:,0:80]
        self.label = h5f['label'][:,:,0:80]
        
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

    def valid_generator(self,test_num):
        data_file = 'snemi3d_train_full_stacks_v1.h5'
        d_file = self.data_dir + data_file
        h5f = h5py.File(d_file,'r')
        data = h5f['data'][:,:,80:100]
        label = h5f['label'][:,:,80:100]
        row = 1024
        col = 1024

        h_input = data.shape[0]
        w_input = data.shape[1]
        s_input = data.shape[2]

        while True:
            w_start = 0
            h_start = 0
            s_start = random.randrange(0,s_input-test_num)

            X = data[h_start:col,w_start:row,s_start:s_start+test_num]
            Y = label[h_start:col,w_start:row,s_start:s_start+test_num]
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




            
            
        
        
            
