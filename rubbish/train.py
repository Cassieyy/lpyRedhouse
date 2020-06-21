import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from model import DeformNet as DCT
import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, criterion, optimizer, device, train_dataload,test_dataload, num_epochs=20,
                                save_epoch = 5,checkpoint_dir='./checkpoints'):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(train_dataload)
        batch_size = train_dataload.batch_size # 需要设置
        dt_size = len(train_dataload.dataset)
        epoch_loss = 0
        step = 0
        for index,data in enumerate(train_dataload):
            step += 1
            input = data[0]
            target = data[1]
        #     print(input.shape,target.shape)
            inputs = input.float().to(device)
            target = target.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            # 这个criterion 需要的是long
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print("{}/{},train_loss:{:.2}".format(step, (dt_size - 1) // batch_size + 1, loss.item()))
            if index %1000==0:
                caculate_acc(model,test_loader,criterion)
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss))
        # if save_epoch %5==0:
        #     torch.save(model.state_dict(), os.path.join(checkpoint_dir,'epoch_{}.pth'.format(epoch)))
        

def train(model,criterion,optimizer,device,train_dataloader,test_loader):
    train_model(model, criterion, optimizer, device,train_dataloader,test_loader)

def caculate_acc(model,test_loader,criterion):
    epoch_loss = 0
    testing_correct=0
    for data in test_loader:
        input,target = data
        inputs = input.float().to(device)
        target = target.to(device)
        outputs = model(inputs)
        _,pred = torch.max(outputs,dim=1)
        testing_correct += torch.sum(pred == target.data)
        loss = criterion(outputs, target)
        epoch_loss += loss.item()
    print("Loss is {:.4f}, Test Accuracy is:{:.4f}"
                .format(epoch_loss,100*testing_correct/len(test_loader)))


if __name__ == "__main__":
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DCT().to(device)
    optimizer = optim.Adam(model.parameters(),lr=0.0001)# lr=0.0001
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5],std=[0.5])])
    train_dataset = torchvision.datasets.MNIST('./', train=True, 
            transform=transform, target_transform=None, download=False)
    test_datatest = torchvision.datasets.MNIST(root="./",
                        transform = transform,train = False)
    train_dataloader = DataLoader(train_dataset)
    test_loader = DataLoader(test_datatest)

    # caculate_acc(model,test_loader,criterion)
    train(model,criterion,optimizer,device,train_dataloader,test_loader)