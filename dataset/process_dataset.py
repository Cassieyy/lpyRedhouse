import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import os
import nibabel as nib
import gzip
import numpy as np  
import pickle
def un_gz(file_name):
    """ungz zip file"""
    f_name = file_name.replace(".gz", "")
    g_file = gzip.GzipFile(file_name)
    open(f_name, "wb+").write(g_file.read())
    g_file.close()
    os.remove(file_name)


def gunzip(root='/home/share/RedHouse/train'):
    print('start unzip...')
    result = []
    for fir_dir in os.listdir(root):   #['DONE0705', 'MRI_2', 'DONE0712']
        print('fir_dir',fir_dir)
        for sec_dir in os.listdir(os.path.join(root,fir_dir)): #['钱慧君-79', '张娣-26', '凌蕾-74', '陆沈明-28', '贺蓓仪-23', '王晶-45', '谢洁林-50', '钱岚-51', '闵祺尔-48']
            print('sec_dir',sec_dir)
            # for thi_dir in os.path.join(root,fir_dir,sec_dir): # ['main', 'segmentation']
            # for thi_path in os.path.join(root,fir_dir,sec_dir,'main'):
            #     for img_path in os.path.join(root,fir_dir,sec_dir,'main',thi_path):
            #         self.imgs.append(os.path.join(root,fir_dir,sec_dir,'main',thi_path,img_path))
            for thi_dir in os.listdir(os.path.join(root,fir_dir,sec_dir,'segmentation')):
                for mask_path in os.listdir(os.path.join(root,fir_dir,sec_dir,'segmentation',thi_dir)):
                    if '.gz' in mask_path:
                        # 解压并删除
                        file_name = os.path.join(root,fir_dir,sec_dir,'segmentation',thi_dir,mask_path)
                        un_gz(file_name)
                        print('need to gunzip...',file_name)
                    else:
                        file_name = os.path.join(root,fir_dir,sec_dir,'segmentation',thi_dir,mask_path)
                        print('seg file...',file_name)
                        result.append(file_name)
    print('the total data can be used in {} are {}'.format(root,len(result)))




class Redhouse(Dataset):
    def __init__(self,root='/home/share/RedHouse/train',which_indexs=['1','2'],threshold=0,img_transform=None,mask_transform=None):
        self.imgs = []
        self.masks = []
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.threshold = threshold
        for fir_dir in os.listdir(root):   #['DONE0705', 'MRI_2', 'DONE0712']
            if fir_dir in ['MRI_2','DONE0712']:continue
            for sec_dir in os.listdir(os.path.join(root,fir_dir)): #['钱慧君-79', '张娣-26', '凌蕾-74', '陆沈明-28', '贺蓓仪-23', '王晶-45', '谢洁林-50', '钱岚-51', '闵祺尔-48']
                for thi_dir in os.listdir(os.path.join(root,fir_dir,sec_dir,'segmentation')):
                    for mask_path in os.listdir(os.path.join(root,fir_dir,sec_dir,'segmentation',thi_dir)):
                        # print('mask_path is: ',mask_path)
                        items = mask_path.split('.')
                        if items[-2] in which_indexs:
                            self.masks.append(os.path.join(root,fir_dir,sec_dir,'segmentation',thi_dir,mask_path))
                            img_path = '.'.join(items[:-2]+[items[-1]])
                            self.imgs.append(os.path.join(root,fir_dir,sec_dir,'main',thi_dir,img_path))
    
        for img,mask in zip(self.imgs,self.masks):
            print('img:',img)
            print('mask',mask)
        print('img:',len(self.imgs),'mask:',len(self.masks))

    def __len__(self):
        return len(self.imgs)



    def __getitem__(self,index):
        
        img_path = self.imgs[index]
        img = nib.load(img_path)
        img = np.array(img.get_fdata())
        mask_path = self.masks[index]
        mask = nib.load(mask_path)
        mask = np.array(mask.get_fdata())# 读进来是float64
        data = []

        pic_num = 0 # 表示有肿瘤的数量
        mask_area = []
        nums = mask.shape[2]
        for i in range(nums):
            temp = mask[:,:,i]
            if np.sum(temp)>self.threshold:
                pic_num +=1
            mask_area.append(-np.sum(temp))
        index = np.argsort(mask_area)
        print(nums,pic_num,mask_area)
        total_nums = min(2*pic_num,nums) # 正负样本参与训练的数量
        for i in range(total_nums):
            one = [] # [img,mask,label,img_path,mask_path,index]
            one.append(img[:,:,index[i]])
            one.append(mask[:,:,index[i]])
            if i<pic_num:
                one.append(1)
            else:
                one.append(0)
            one.append(img_path)
            one.append(mask_path)
            one.append(index[i])
            data.append(one)
        return data



if __name__ == "__main__":

    dataset = Redhouse()
    dataloader = DataLoader(dataset)
    data = []

    for i,batch in enumerate(dataloader):
        print('batch len:',len(batch))
        data.append(batch)
        if i>10:
            break
    file = open('../data/test.pkl','wb+')
    pickle.dump(data,file)
    file.close()
        



