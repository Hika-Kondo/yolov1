import os
import sys
import os.path

import random
import numpy as np

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import cv2
import matplotlib.pyplot as plt
import pandas as pd


class YoloDataSet(data.Dataset):

    def __init__(self, data_file_path, dataframe_path, pred_classes, image_size, output_size, num_overlap):
        self.image_size = image_size
        dataframe = pd.read_csv(dataframe_path)



def main():
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms
    from torchvision.utils import save_image
    from PIL import Image
    from matplotlib import pyplot as plt

    from visualize import save_res_im
    import warnings
    warnings.simplefilter('ignore')

    image_size = [1292,485]
    output_size = image_size
    for _ in range(5):
        output_size = [output_size[0]//2, output_size[1]//2]

    file_root = '/images/'
    train_dataset = yoloDataset(root=file_root,list_file='/tmp/bt_im.txt',train=True,transform = [transforms.ToTensor()],pred_classes=1,
            image_size=[1292,485], output_size=output_size)
    # file_root = "/val"
    # train_dataset = yoloDataset(root=file_root,list_file='/tmp/val.txt',train=False,transform = [transforms.ToTensor()],pred_classes=1)
    train_loader = DataLoader(train_dataset,batch_size=1,shuffle=False,num_workers=0)
    for idx, (image, target) in enumerate(train_loader):
        # print("idx :{} image: {} target: {}".format(idx, image.size(), target.size()))
        save_res_im(image, target, target, "res_img.jpg", "/res/test/", draw_ans=False, output_size=output_size)
        target = target.permute(0,3,1,2)
        print(image.size(), target.size())
        cunt = 0
        for i in range(target.size(2)):
            li = target[0][4][i].tolist()
            li = [1 if i != 0 else 0 for i in li ]
            # print(li)
            for l in li:
                if i == 1:
                    cunt += 1
        # print(cunt)
        break


if __name__ == '__main__':
    main()
