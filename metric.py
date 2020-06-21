import cv2
import numpy as np
import torch


SMOOTH = 1e-6

def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    return thresholded.mean().item()  # Or thresholded.mean() if you are interested in average across the batch



class IOUMetric:
    """
    Class to calculate mean-iou using fast_hist method
    """
 
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))
 
    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist
 
    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())
 
    def evaluate(self):
        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.diag(self.hist) / self.hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        mean_iu = np.nanmean(iu)
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum() 
        return acc, acc_cls, iu, mean_iu, fwavacc
 
 
 # mask [channel,h,w]
def get_iou(mask,predict,thr):
    mask = mask.squeeze(0)
    predict = predict.squeeze(0)
    height = predict.shape[0]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
    weight = predict.shape[1]

    predict[predict<thr] = 0
    predict[predict>=thr] = 1
    mask[mask<thr] = 0
    mask[mask>=thr] = 1
    # mask[:8,:] = 1
    predict = predict.numpy().astype(np.int16)
    mask = mask.numpy().astype(np.int16)

    Iou = IOUMetric(2)  
    Iou.add_batch(predict, mask)
    a, b, c, d, e= Iou.evaluate()
    pn = 0
    if Iou.hist.sum(axis=1)[1]>1:
        pn += 1
    return d,pn


# mask [batchsize,channel,h,w]
def get_miou(mask,predict,thr):
    batchsize = mask.shape[0]
    m_iou = []
    pn = []
    for i in range(batchsize):
        d,p = get_iou(mask[i],predict[i],thr)
        m_iou.append(d)
        pn.append(p)

    return sum(m_iou)/batchsize,sum(pn)