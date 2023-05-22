
import os
import sys
import esc_data_utils as utils
import numpy as np 
import pandas as pd
from torch.utils.data import Dataset, DataLoader

################
# DATA OBJECT #
################

path = '.data/ESC-50-master/meta/esc50.csv'

class ESC50data(Dataset):

    '''
    
    Dataset Object:
    1. receives dataframe 
    2. organizes categories based on passed data and label cols
    3. generates categorical look up tables
    4. retains number of data
    5. retains callable data/label iterator based on passed index
    
    '''

    def __init__(self, base, df, in_col, out_col):

        self.df = df
        self.data = []
        self.labels = []
        self.cat2idx = {}
        self.idx2cat = {}
        self.categories = sorted(df[out_col].unique())
        # iterates through categories attribute to build look up tables
        x = 0
        for i, category in enumerate(self.categories):
            x += 1
            z = (f'[+] Generating Data lookup {round((x/len(self.categories))*100, 2)}%')
            self.cat2idx[category] = i
            self.idx2cat[i] = category
            sys.stdout.write('\r'+z)
        print('\n')
        # iterates through the number of audio data and transforms to spectrograph
        # and assigns respective class number
        x = 0    
        for idx in range(len(df)):
            x += 1
            z = (f'[+] Generating Label lookup {round((x/len(df))*100, 2)}%')
            row = df.iloc[idx]
            fpath = os.path.join(base,row[in_col])
            self.data.append(utils.spec_to_img(utils.melspectrogram_db(fpath))[np.newaxis,...])
            # self.data.append(utils.spec_to_img(utils.mfcc_db(fpath))[np.newaxis,...])
            
            self.labels.append(self.cat2idx[row['category']])
            sys.stdout.write('\r'+z)
        print('\n')

    def __len__(self):

        '''
        retrieves number of data
        '''
        return len(self.data)

    def __getitem__(self, idx):

        '''
        retrieves single data and label based on index
        '''
        return self.data[idx], self.labels[idx]

#####################
# DATA OBJECT UTILS #
#####################

def read_data(path):

    '''
    reads in dataframe generates 80% TRAIN | 20% VALID split
    '''
    df = pd.read_csv(path)
    # in the ESC50 data set there are five folds
    # training data will use folds 1 - 4
    # valid data will use fold 5
    train = df[df['fold'] != 5]
    valid = df[df['fold'] == 5]
    # generate data objects with train and valid data
    train_data = ESC50data('.data/ESC-50-master/audio', train, 'filename', 'category')
    valid_data = ESC50data('.data/ESC-50-master/audio', valid, 'filename', 'category')
    
    return train_data, valid_data


def gen_data_loader(train_data, valid_data):
    # build dataloader for training
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=16, shuffle=True)

    return train_loader, valid_loader