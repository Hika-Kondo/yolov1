import torch
from torch.autograd import Variable
import torch.nn as nn
from torchvision.transforms import ToPILImage
from torchvision.utils import save_image
import cv2
from PIL import Image
from PIL import ImageDraw
import numpy as np

import os

# from net import resnet50
import torchvision.transforms as transforms

VOC_CLASSES = (    # always index 0
    'pred', 'ans', '', '',
    '', '', '', '', '',
    '', '', '', '',
    '', '', '',
'', '', '', '')


Color = [[0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [192, 0, 0],
        [64, 128, 0],
        [192, 128, 0],
        [64, 0, 128],
        [192, 0, 128],
        [64, 128, 128],
        [192, 128, 128],
        [0, 64, 0],
        [128, 64, 0],
        [0, 192, 0],
        [128, 192, 0],
        [0, 64, 128]]


def decoder(pred):
    '''
    pred (tensor) 1x7x7x30
    return (tensor) box[[x1,y1,x2,y2]] label[...]
    '''
    grid_num = 14
    boxes=[]
    cls_indexs=[]
    probs = []
    cell_size = 1./grid_num
    pred = pred.data
    pred = pred.squeeze(0) #7x7x30
    contain1 = pred[:,:,4].unsqueeze(2)
    contain2 = pred[:,:,9].unsqueeze(2)
    contain = torch.cat((contain1,contain2),2)
    mask1 = contain > 0.1 #大于阈值
    mask2 = (contain==contain.max()) #we always select the best contain_prob what ever it>0.9
    mask = (mask1+mask2).gt(0)
    # min_score,min_index = torch.min(contain,2) #每个cell只选最大概率的那个预测框
    for i in range(grid_num):
        for j in range(grid_num):
            for b in range(2):
                # index = min_index[i,j]
                # mask[i,j,index] = 0
                if mask[i,j,b] == 1:
                    #print(i,j,b)
                    box = pred[i,j,b*5:b*5+4]
                    contain_prob = torch.FloatTensor([pred[i,j,b*5+4]])
                    xy = torch.FloatTensor([j,i])*cell_size #cell左上角  up left of cell
                    box[:2] = box[:2]*cell_size + xy # return cxcy relative to image
                    box_xy = torch.FloatTensor(box.size())#转换成xy形式    convert[cx,cy,w,h] to [x1,xy1,x2,y2]
                    box_xy[:2] = box[:2] - 0.5*box[2:]
                    box_xy[2:] = box[:2] + 0.5*box[2:]
                    max_prob,cls_index = torch.max(pred[i,j,10:],0)
                    if float((contain_prob*max_prob)[0]) > 0.1:
                        boxes.append(box_xy.view(1,4))
                        cls_indexs.append(cls_index)
                        probs.append(contain_prob*max_prob)
    _cls_indexs = torch.Tensor(len(cls_indexs))
    for i in range(len(cls_indexs)):
        _cls_indexs[i] = cls_indexs[i]
    cls_indexs = _cls_indexs
    if len(boxes) ==0:
        boxes = torch.zeros((1,4))
        probs = torch.zeros(1)
        cls_indexs = torch.zeros(1)
    else:
        boxes = torch.cat(boxes,0) #(n,4)
        probs = torch.cat(probs,0) #(n,)
        # cls_indexs = torch.cat(cls_indexs,torch.tensor([0])) #(n,)
    keep = nms(boxes,probs)
    return boxes[keep],cls_indexs[keep],probs[keep]


def nms(bboxes,scores,threshold=0.2):
    '''
    bboxes(tensor) [N,4]
    scores(tensor) [N,]
    '''
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]
    areas = (x2-x1) * (y2-y1)

    _,order = scores.sort(0,descending=True)
    keep = []
    while order.numel() > 0:
        if len(order.size()) == 0:
            break
        i = order[0].item()
        keep.append(i)

        if order.numel() == 1:
            break

        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2-xx1).clamp(min=0)
        h = (yy2-yy1).clamp(min=0)
        inter = w*h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        ids = (ovr<=threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids+1]
    return torch.LongTensor(keep)


def show_bb(img, x, y, w, h, text, textcolor, bbcolor):
    draw = ImageDraw.Draw(img)
    text_w, text_h = draw.textsize(text)
    label_y = y if y <= text_h else y - text_h
    draw.rectangle((x, label_y, x+w, label_y+h), outline=bbcolor)
    draw.rectangle((x, label_y, x+text_w, label_y+text_h), outline=bbcolor, fill=bbcolor)
    draw.text((x, label_y), text, fill=textcolor)


def draw_bb(img, pred, text, bbox, label):
    """get pil image and return bbox pil image

    args
    img: pil image
    pred: bbox, label tensor
    text: text color
    bbox: bbox color

    return 
    img: pil image bboxed image
    """
    pred = pred.cpu()
    pred_boxes, pred_cls_indexs, probs = decoder(pred)

    # image = img[0].cpu().numpy().transpose(1,2,0)
    h,w = img.size
    result = []
    for i, box in enumerate(pred_boxes):
        x1 = int(box[0]*w)
        x2 = int(box[2]*w)
        y1 = int(box[1]*h)
        y2 = int(box[3]*h)
        cls_index = pred_cls_indexs[i]
        cls_index = int(cls_index) # convert LongTensor to int
        prob = probs[i]
        result.append([(x1,y1),(x2,y2),prob])

    for (x_min, y_min), (x_max, y_max), prob in result:
        w = x_max - x_min; h = y_max - y_min
        # textcolor = (255,255,255)
        # label = class_name+str(round(prob.item(),2))
        # bbcolor = Color[VOC_CLASSES.index(class_name)]
        # bbcolor = tuple(bbcolor)
        show_bb(img, x_min, y_min, w, h, label, text, bbox)
    return img
    # img.save(os.path.join(root, name))


def save_res_im(img, pred, ans, name, root, draw_ans=True):
    """draw pred bboxes and ans bboxes and save to root/name
    args
    img: torch tensor
    pred: torch tensor model output
    ans: torch tensor target
    name: file name you want to save im
    root: save dir
    """
    save_image(img,"/tmp/tmp.jpg")
    img = Image.open("/tmp/tmp.jpg")

    # pred bboxes
    img = draw_bb(img, pred, (255,255,255),(255,0,0), "pred")
    # ans bboxes
    if draw_ans:
        img = draw_bb(img, ans, (255,255,255), (0,255,0), "ans")

    img.save(os.path.join(root, name))
