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


class Redhouse(Dataset):
    def __init__(self,root='./data/img_mask',img_size=256,img_transform=None,mask_transform=None):
        self.imgs = []
        self.img_size = img_size
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        for filename in os.listdir(root):
            self.imgs.append(os.path.join(root,filename))


    def __len__(self):
        return len(self.imgs)

    #[C,H,W]
    def crop_center(self,img,cropx,cropy):
        c,x,y = img.shape
        startx = x//2 - cropx//2
        starty = y//2 - cropy//2    
        return img[:,starty:starty+cropy, startx:startx+cropx]

    def __getitem__(self,index):
        # 这个地方还可以进一步调整想要的数据形式
        # [img,mask,label,img_path,mask_path,index]
        file = open(self.imgs[index],'rb+')
        data = pickle.load(file)
        file.close()
        img = data[0]
        mask = data[1]

        img = self.crop_center(img,self.img_size,self.img_size)
        mask = self.crop_center(mask,self.img_size,self.img_size)
        return (img,mask,data[3],data[4])

if __name__ == "__main__":
    batch_size = 1
    dataset = Redhouse(root = '../data/rubbish')
    dataloader = DataLoader(dataset,batch_size=batch_size)
    print(dataloader.batch_size,len(dataloader.dataset))
    for i,(img,mask,mask_path,img_path) in enumerate(dataloader):
        print(img.shape,mask_path,img_path)
        break
        
