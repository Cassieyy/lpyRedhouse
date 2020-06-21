import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
from model.MunetModel import Munet
from model.SeUnetModel import SeUnet
from model.Scmodel import SCModel
from dataset.dataset import Redhouse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import metric 
import torchvision

def test_model(model,device, dataload,thr):
    
    batch_size = dataload.batch_size # 需要设置
    dt_size = len(dataload.dataset)
    m_ious = []
    np = []
    for i,data in enumerate(dataload):
        input = data[0]
        inputs = input.repeat((1,3,1,1))

        target = data[1]
        # print(input.shape,target.shape)
        inputs = inputs.float().to(device)
        target = target.float().to(device)
        outputs = model(inputs)
        # torchvision.utils.save_image(outputs*100,'temp.png')
        # torchvision.utils.save_image(inputs*200,'temp1.png')

        m_iou,p=metric.get_miou(target.cpu().detach(),outputs.cpu().detach(),thr)
        m_ious.append(m_iou)
        np.append(p)
        print('m_ious:',m_iou)
        print('np',sum(np))
    
    return sum(m_ious)/len(m_ious),sum(np)


def test(model,device,dataloader,thr):
    return test_model(model,device,dataloader,thr)

def load_model_checkpoints(model,checkpoint_path='./checkpoints/2020_04_24_19_46/epoch_19.pth'):
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict)
    
if __name__ == "__main__":

    batch_size = 8
    thr = 0.5
    test_datapath = './data/img_mask_test'
    load_checkpoint_path = './checkpoints/2020_06_21_11_22/epoch_5.pth'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SCModel(3,1).to(device)
    model.eval()
    dataset = Redhouse(test_datapath)
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=False)
    print('*' * 15,'batch_sizes = {}'.format(batch_size),'*' * 15)
    load_model_checkpoints(model,load_checkpoint_path)
    m_ious,pn = test(model,device,dataloader,thr)
    print('*' * 15,m_ious,'*' * 15)
    print('*' * 15,pn)