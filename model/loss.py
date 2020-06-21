import os
import numpy as np
import torch
import torch.nn as nn

# FL(pt) = −αt(1 − pt)γ log(pt)
class FocalLoss(nn.Module):
    # def __init__(self):

    def forward(self, classifications, targets):
        alpha = 0.5 # 控制样本均衡
        gamma = 2.0
        classifications =  classifications.squeeze(1)
        targets = targets.squeeze(1)
        batch_size = classifications.shape[0]
        classification_losses = []
        for j in range(batch_size):
            classification = classifications[j, :, :]
            target = targets[j, :, :]
            # 这边是否有必要做clamp
            # classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)
            alpha_factor = torch.ones(target.shape).cuda() * alpha
            # 用两个where 来完成
            alpha_factor = torch.where(
                torch.eq(target, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(
                torch.eq(target, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(target * torch.log(classification) +
                    (1.0 - target) * torch.log(1.0 - classification))
            cls_loss = focal_weight * bce

            # 下面这个torch.where 是用来控制多少的阈值需要处理
            # cls_loss = torch.where(
            #     torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())

            classification_losses.append(
                cls_loss.mean())
        # return torch.stack(classification_losses).mean(dim=0, keepdim=True)
        return torch.stack(classification_losses).mean(dim=0)

if __name__ == "__main__":
    focaloss = FocalLoss()
    classfication = torch.rand((2,1,225,225)).cuda()
    target = torch.rand((2,1,225,225)).cuda()
    loss = nn.BCELoss()
    # print(classfication)
    # print(target)
    a = focaloss(classfication,target)
    print(a)
    b = loss(classfication,target)
    print(b)
