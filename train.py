import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
from model.MunetModel import Munet
from model.SeUnetModel import SeUnet
from model.Scmodel import SCModel as Scnet
from model.loss import FocalLoss
from dataset.dataset import Redhouse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter

def train_model(model, criterion, optimizer, device, dataload, num_epochs=22,
                                save_epoch = 5, log_name = './logs', checkpoint_name='./checkpoints',
                                loss_fre=2,img_fre=20,record = False):
    

    if record:
        log_dirname = os.path.join(log_name,time.strftime("%Y_%m_%d_%H_%M",time.localtime()))
        if not os.path.exists(log_dirname):
            os.makedirs(log_dirname)
        save_dirname =  os.path.join(checkpoint_name,time.strftime("%Y_%m_%d_%H_%M",time.localtime()))
        if not os.path.exists(save_dirname):
            os.makedirs(save_dirname)
        writer = SummaryWriter(log_dirname)
    print('starting training ---------')
    print(num_epochs)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        batch_size = dataload.batch_size # 需要设置
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0
        for i,data in enumerate(dataload):
            step += 1
            input = data[0]
            # 这边是针对
            inputs = input.repeat((1,3,1,1))
            target = data[1]
            # print(input.shape,target.shape)
            inputs = inputs.float().to(device)
            target = target.float().to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print("{}/{},train_loss:{:.2}".format(step, (dt_size - 1) // batch_size + 1, loss.item()))
            if record:
                if i % loss_fre == 0:
                    # writer.add_scalars('train/loss:',{'discriminator':d_loss.item(),
                    #                             'generator':g_loss.item()},epoch*dt_size+batch_size*i)
                    writer.add_scalar('train/loss:',loss.item(),epoch*dt_size+batch_size*i)
                if i % img_fre==0:
                    print('#####')
                    input = np.array(data[0][0],dtype=np.uint8)
                    target = np.array(data[1][0]*255,dtype=np.uint8)
                    output = np.array(outputs[0].detach().cpu()*255,dtype=np.uint8)
                    writer.add_images('Image/train/2020._cancer_epoch_{}'.format(epoch),np.concatenate([input,target,output],axis=2),epoch,dataformats='CHW')
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss))
        if epoch %save_epoch==0:
            torch.save(model.state_dict(), os.path.join(save_dirname,'epoch_{}.pth'.format(epoch)))


def train(model,criterion,optimizer,device,dataloader,record,num_epochs):
    train_model(model, criterion, optimizer, device,dataloader,record=record,num_epochs=num_epochs)

def load_model_checkpoints(model,checkpoint_path='./checkpoints/2020_05_23_10_17_se/latest.pth'):
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict)

if __name__ == "__main__":

    batch_size = 16
    continue_train = True
    focal_loss = True
    record = True

    lr = 0.000001
    num_epochs = 10
    load_checkpoint_path = './checkpoints/2020_06_21_11_22/epoch_5.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = Unet(1,1).to(device)
    # model = Munet(1,1).to(device)
    # model = SeUnet(1,1).to(device)
    model = Scnet(3,1).to(device)
    optimizer = optim.Adam(model.parameters(),lr=lr)
    dataset = Redhouse(root='./data/img_mask')
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True)

    print('*' * 15,'continue_train = {}'.format(continue_train),'*' * 15)
    print('*' * 15,'batch_sizes = {}'.format(batch_size),'*' * 15)
    print('*' * 15,'lr = {}'.format(lr),'*' * 15)
    print('*' * 15,'criterion?focal_loss = {}'.format(focal_loss),'*' * 15)
    print('*' * 15,'device {}'.format(device),'*' * 15)

    if continue_train:
        print('*' * 15,'continue training...','*' * 15)
        load_model_checkpoints(model,load_checkpoint_path)
    if focal_loss:
        criterion = FocalLoss()
    else:
        criterion = nn.BCELoss()

    train(model,criterion,optimizer,device,dataloader,record,num_epochs)

