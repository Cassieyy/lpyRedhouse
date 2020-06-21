import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import nibabel as nib
import gzip
import numpy as np  
import pickle

'''
该文件本来作为数据类，但是总体占用内存资源过大，现将每一帧
存在 ../data/img_mask下 {}_img_mask.pkl
'''
class Redhouse(Dataset):
    def __init__(self,root='./data/train.pkl',img_size=256,img_transform=None,mask_transform=None):
        self.imgs = []
        self.img_size = img_size
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        file = open(root,'rb+')
        data = pickle.load(file)
        print('len(data)',len(data))
        for item in data:
            # print(type(item),len(item),len(item[0]),item[0][1].shape)
            self.imgs.extend(item)
        print('img:',len(self.imgs))

    def __len__(self):
        return len(self.imgs)


    def __getitem__(self,index):
        # 这个地方还可以进一步调整想要的数据形式
        # [img,mask,label,img_path,mask_path,index]
        # if index ==4018:
        #     print(self.imgs[index])
        file = open('../data/img_mask/{}_img_mask.pkl'.format(index),'wb+')
        pickle.dump(self.imgs[index],file)
        file.close()
        return index

if __name__ == "__main__":
    batch_size = 1
    dataset = Redhouse(root = '../data/test.pkl')
    dataloader = DataLoader(dataset,batch_size=batch_size)
    print(dataloader.batch_size,len(dataloader.dataset))
    for i,batch in enumerate(dataloader):
        print(i,type(batch))
        break
        
