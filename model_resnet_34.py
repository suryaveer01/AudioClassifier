
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet34


##############
# RESNET_34  #
##############   

class RES:

    '''
    Modified Pretrained ResNet34
    '''

    def __init__(self):

        self.model = resnet34(pretrained=True)

    def gen_resnet(self):

        if torch.cuda.is_available(): 
            device=torch.device('cuda:0')
        else:
            device=torch.device('cpu')

        model = self.model
        model.fc = nn.Linear(512, 50)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model = model.to(device)

        return model

    

        