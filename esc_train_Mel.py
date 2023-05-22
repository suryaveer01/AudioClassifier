
import torch
import esc_dataset_Mel as data
import model_resnet_34 as m
import esc_data_utils as utils

#########################
# TRAINING ABSTRACTIONS #
#########################

def prepare_data():

    '''
    Prepare Data:
    1. read in audio meta data
    2. set data loaders for training
    3. save categorical look up tabels
    '''

    train_data, valid_data = data.read_data(data.path)
    train_loader, valid_loader = data.gen_data_loader(train_data, valid_data)
    utils.save_cat_idx(train_data, 'models/idx2cat.pkl')

    return train_loader, valid_loader

def train_resNet(train_loader, valid_loader):

    '''
    Train ResNet34:
    1. instantiate modified pretrained ResNet34
    2. train model
    3. save model
    '''

    model = m.RES().gen_resnet()

    trained_model = utils.train(model, train_loader, valid_loader,
                         epochs=50,
                         learning_rate=2e-4
                         )
    utils.save_model(trained_model, 'models/resNet.pth')

    return trained_model



                    
if __name__ == '__main__':

    ####################
    # RUN ABSTRACTIONS #
    ####################
    
    train_loader, valid_loader = prepare_data()
    # CNN_model = train_CNN(train_loader, valid_loader)
    resNet_model = train_resNet(train_loader, valid_loader)