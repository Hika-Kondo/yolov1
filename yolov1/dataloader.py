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


class yoloDataset(data.Dataset):
    """
    YOLO dataset Class
    """

    def __init__(self,root,list_file,train,transform, pred_classes, image_size=448):
        """
        __init__
        perse input files
        Args:
            root (str) : dataset root dirctory
            list_file (str or list) : if str, list file of input file name
            train(bool) : this dataset is train dataset or not
            transform (torchvision.trainsforms) : preprocess from torchvision
        """

        print('data init')
        self.root=root
        self.train = train
        self.transform=transform
        self.fnames = []
        self.boxes = []
        self.labels = []
        self.mean = (123,117,104)#RGB
        self.image_size = image_size
        self.pred_classes = pred_classes

        # list_fileがlistだった場合、入力の内容を/tmp/listfile.txtに保存する
        if isinstance(list_file, list):
            # Cat multiple list files together.
            # This is especially useful for voc07/voc12 combination.
            tmp_file = '/tmp/listfile.txt'
            os.system('cat %s > %s' % (' '.join(list_file), tmp_file))
            list_file = tmp_file

        with open(list_file) as f:
            lines  = f.readlines()

        # 入力ファイルのパース
        # self.boxesに対応する座標
        # labelに対応するクラス
        # fnames に画像ファイル
        for line in lines:
            # 入力のファイルのフォーマットは
            # file name -> x min -> y min -> x max -> y max -> class
            # の順番で入っている
            splited = line.strip().split()
            self.fnames.append(splited[0])
            num_boxes = (len(splited) - 1) // 5
            box=[]
            label=[]
            for i in range(num_boxes):
                x = float(splited[1+5*i])
                y = float(splited[2+5*i])
                x2 = float(splited[3+5*i])
                y2 = float(splited[4+5*i])
                c = splited[5+5*i]
                box.append([x,y,x2,y2])
                label.append(int(c)+1)
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))
        self.num_samples = len(self.boxes)

    def __getitem__(self,idx):
        """__getitem__
        get item
        if train, random flip, random scale, blur, Hue, saturartion, shift, crop
        else not data augumentaion
        Args:
            idx (int) : index of fname
        Return:
            img (cv2) : image to input YOLO
            target (torch.Tensor) : ans of input image
        """

        # open image and Get the corresponding label and bbox
        fname = self.fnames[idx]
        img = cv2.imread(os.path.join(self.root,fname))
        boxes = self.boxes[idx].clone()
        labels = self.labels[idx].clone()
        img, boxes, labels = self.RandomImageCrop(img, boxes, labels)

        # if dataset is train dataset, preprocess
        if self.train:
            #img = self.random_bright(img)
            # img, boxes = self.random_flip(img, boxes)
            # img,boxes = self.randomScale(img,boxes)
            img = self.randomBlur(img)
            img = self.RandomBrightness(img)
            img = self.RandomHue(img)
            img = self.RandomSaturation(img)
            # img,boxes,labels = self.randomShift(img,boxes,labels)
            # img,boxes,labels = self.randomCrop(img,boxes,labels)
        # #debug
        # box_show = boxes.numpy().reshape(-1)
        # print(box_show)
        # img_show = self.BGR2RGB(img)
        # pt1=(int(box_show[0]),int(box_show[1])); pt2=(int(box_show[2]),int(box_show[3]))
        # cv2.rectangle(img_show,pt1=pt1,pt2=pt2,color=(0,255,0),thickness=1)
        # plt.figure()

        # # cv2.rectangle(img,pt1=(10,10),pt2=(100,100),color=(0,255,0),thickness=1)
        # plt.imshow(img_show)
        # plt.show()
        # #debug
        h,w,_ = img.shape
        boxes /= torch.Tensor([w,h,w,h]).expand_as(boxes)
        img = self.BGR2RGB(img) #because pytorch pretrained model use RGB
        # img = self.subMean(img,self.mean) #减去均值
        # img = cv2.resize(img,(self.image_size,self.image_size))
        target = self.encoder(boxes,labels)# 7x7x30
        for t in self.transform:
            img = t(img)
        return img,target

    def __len__(self):
        """get length of dataset
        Return:
            self.num_sample (int) : length of dataset
        """
        return self.num_samples

    def encoder(self,boxes,labels):
        """Encode the tensor that will be the target of yolo for the image.
        this method return ans of yolo.
        tensor shape is 14x14x30
        14x14 is
        """
        # '''encoder

        # boxes (tensor) [[x1,y1,x2,y2],[]]
        # labels (tensor) [...]
        # return 7x7x30
        # '''

        grid_num = 14 # グリッドサイズ論文では7
        target = torch.zeros((grid_num,grid_num,10+self.pred_classes)) # 出力のグリットのTensor
        cell_size = 1./grid_num
        wh = boxes[:,2:]-boxes[:,:2] # バウンディングボックスのサイズ
        cxcy = (boxes[:,2:]+boxes[:,:2])/2 # バウンディングボックスの中心の座標
        for i in range(cxcy.size()[0]):
            cxcy_sample = cxcy[i]
            ij = (cxcy_sample/cell_size).ceil() - 1 #
            target[int(ij[1]),int(ij[0]),4] = 1
            target[int(ij[1]),int(ij[0]),9] = 1
            target[int(ij[1]),int(ij[0]),int(labels[i])+9] = 1
            xy = ij*cell_size #匹配到的网格的左上角相对坐标
            delta_xy = (cxcy_sample -xy)/cell_size
            target[int(ij[1]),int(ij[0]),2:4] = wh[i]
            target[int(ij[1]),int(ij[0]),:2] = delta_xy
            target[int(ij[1]),int(ij[0]),7:9] = wh[i]
            target[int(ij[1]),int(ij[0]),5:7] = delta_xy
        return target

    def BGR2RGB(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    def BGR2HSV(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    def HSV2BGR(self,img):
        return cv2.cvtColor(img,cv2.COLOR_HSV2BGR)

    def RandomBrightness(self,bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h,s,v = cv2.split(hsv)
            adjust = random.choice([0.5,1.5])
            v = v*adjust
            v = np.clip(v, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def RandomSaturation(self,bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h,s,v = cv2.split(hsv)
            adjust = random.choice([0.5,1.5])
            s = s*adjust
            s = np.clip(s, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def RandomHue(self,bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h,s,v = cv2.split(hsv)
            adjust = random.choice([0.5,1.5])
            h = h*adjust
            h = np.clip(h, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def randomBlur(self,bgr):
        if random.random()<0.5:
            bgr = cv2.blur(bgr,(5,5))
        return bgr

    def randomShift(self,bgr,boxes,labels):
        #平移变换
        center = (boxes[:,2:]+boxes[:,:2])/2
        if random.random() <0.5:
            height,width,c = bgr.shape
            after_shfit_image = np.zeros((height,width,c),dtype=bgr.dtype)
            after_shfit_image[:,:,:] = (104,117,123) #bgr
            shift_x = random.uniform(-width*0.2,width*0.2)
            shift_y = random.uniform(-height*0.2,height*0.2)
            #原图像的平移
            if shift_x>=0 and shift_y>=0:
                after_shfit_image[int(shift_y):,int(shift_x):,:] = \
                        bgr[:height-int(shift_y),:width-int(shift_x),:]
            elif shift_x>=0 and shift_y<0:
                after_shfit_image[:height+int(shift_y),int(shift_x):,:] = \
                        bgr[-int(shift_y):,:width-int(shift_x),:]
            elif shift_x <0 and shift_y >=0:
                after_shfit_image[int(shift_y):,:width+int(shift_x),:] = \
                        bgr[:height-int(shift_y),-int(shift_x):,:]
            elif shift_x<0 and shift_y<0:
                after_shfit_image[:height+int(shift_y),:width+int(shift_x),:] = \
                        bgr[-int(shift_y):,-int(shift_x):,:]

            shift_xy = torch.FloatTensor([[int(shift_x),int(shift_y)]]).expand_as(center)
            center = center + shift_xy
            mask1 = (center[:,0] >0) & (center[:,0] < width)
            mask2 = (center[:,1] >0) & (center[:,1] < height)
            mask = (mask1 & mask2).view(-1,1)
            boxes_in = boxes[mask.expand_as(boxes)].view(-1,4)
            if len(boxes_in) == 0:
                return bgr,boxes,labels
            box_shift = torch.FloatTensor([[int(shift_x),int(shift_y),int(shift_x),int(shift_y)]])\
                    .expand_as(boxes_in)
            boxes_in = boxes_in+box_shift
            labels_in = labels[mask.view(-1)]
            return after_shfit_image,boxes_in,labels_in
        return bgr,boxes,labels

    def randomScale(self,bgr,boxes):
        #固定住高度，以0.8-1.2伸缩宽度，做图像形变
        if random.random() < 0.5:
            scale = random.uniform(0.8,1.2)
            height,width,c = bgr.shape
            bgr = cv2.resize(bgr,(int(width*scale),height))
            scale_tensor = torch.FloatTensor([[scale,1,scale,1]]).expand_as(boxes)
            boxes = boxes * scale_tensor
            return bgr,boxes
        return bgr,boxes

    def randomCrop(self,bgr,boxes,labels):
        if random.random() < 0.5:
            center = (boxes[:,2:]+boxes[:,:2])/2
            height,width,c = bgr.shape
            h = random.uniform(0.6*height,height)
            w = random.uniform(0.6*width,width)
            x = random.uniform(0,width-w)
            y = random.uniform(0,height-h)
            x,y,h,w = int(x),int(y),int(h),int(w)

            center = center - torch.FloatTensor([[x,y]]).expand_as(center)
            mask1 = (center[:,0]>0) & (center[:,0]<w)
            mask2 = (center[:,1]>0) & (center[:,1]<h)
            mask = (mask1 & mask2).view(-1,1)

            boxes_in = boxes[mask.expand_as(boxes)].view(-1,4)
            if(len(boxes_in)==0):
                return bgr,boxes,labels
            box_shift = torch.FloatTensor([[x,y,x,y]]).expand_as(boxes_in)

            boxes_in = boxes_in - box_shift
            boxes_in[:,0]=boxes_in[:,0].clamp_(min=0,max=w)
            boxes_in[:,2]=boxes_in[:,2].clamp_(min=0,max=w)
            boxes_in[:,1]=boxes_in[:,1].clamp_(min=0,max=h)
            boxes_in[:,3]=boxes_in[:,3].clamp_(min=0,max=h)

            labels_in = labels[mask.view(-1)]
            img_croped = bgr[y:y+h,x:x+w,:]
            return img_croped,boxes_in,labels_in
        return bgr,boxes,labels

    def subMean(self,bgr,mean):
        mean = np.array(mean, dtype=np.float32)
        bgr = bgr - mean
        return bgr

    def random_flip(self, im, boxes):
        """
        左右反転
        """
        if random.random() < 0.5:
            im_lr = np.fliplr(im).copy()
            h,w,_ = im.shape
            xmin = w - boxes[:,2]
            xmax = w - boxes[:,0]
            boxes[:,0] = xmin
            boxes[:,2] = xmax
            return im_lr, boxes
        return im, boxes

    def random_bright(self, im, delta=16):
        alpha = random.random()
        if alpha > 0.3:
            im = im * alpha + random.randrange(-delta,delta)
            im = im.clip(min=0,max=255).astype(np.uint8)
        return im

    def RandomImageCrop(self, im, boxes, labels):
        """
        画像をself.im_sizeの大きさに切り出す
        切り出した画像のなかに当てはまるバウンディングボックスを生成する
        """
        # im -> cv2(numpy)
        # boxes -> list
        # x_min のインデックスは4*idx
        # y_min のインデックスは4*idx + 1
        # x_max のインデックスは4*idx + 2
        # y_max のインデックスは4*idx + 3
        boxes = boxes.tolist()
        labels = labels.tolist()
        h, w, _ = im.shape
        crop_x_min = random.randint(0, w - self.image_size)
        crop_y_min = random.randint(0, h - self.image_size)
        crop_x_max = crop_x_min + self.image_size
        crop_y_max = crop_y_min + self.image_size

        return_box = []
        return_label = []
        return_im = im[crop_y_min:crop_y_max, crop_x_min: crop_x_max]

        for idx in range(len(boxes)):
            box = boxes[idx]
            label = labels[idx]
            box_x_min = box[0]
            box_y_min = box[1]
            box_x_max = box[2]
            box_y_max = box[3]
            # cropの完全に外の場合
            if box_x_max < crop_x_min or box_y_max < crop_y_min or\
            box_x_min > crop_x_max or box_y_min > crop_y_max:
                continue
            # cropにboxが一部でも重なっている場合
            else:
                return_box_item = []
                box_x_min -= crop_x_min; box_y_min -= crop_y_min
                box_x_max -= crop_x_min; box_y_max -= crop_y_min
                x_min = max(box_x_min, 0)
                y_min = max(box_y_min, 0)
                x_max = min(box_x_max, self.image_size)
                y_max = min(box_y_max, self.image_size)
                return_box_item.append(x_min)
                return_box_item.append(y_min)
                return_box_item.append(x_max)
                return_box_item.append(y_max)
                return_box.append(return_box_item)
                return_label.append(label)
        return_box = torch.Tensor(return_box)
        return_label = torch.LongTensor(return_label)
        return return_im, return_box, return_label

def main():
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms
    from torchvision.utils import save_image
    from PIL import Image

    from visualize import save_bboxim

    file_root = '/images/'
    # file_root = "/val"
    # train_dataset = yoloDataset(root=file_root,list_file='/tmp/val.txt',train=True,transform = [transforms.ToTensor()],pred_classes=1)
    train_dataset = yoloDataset(root=file_root,list_file='/tmp/bt_im.txt',train=True,transform = [transforms.ToTensor()],pred_classes=1)
    train_loader = DataLoader(train_dataset,batch_size=1,shuffle=False,num_workers=0)
    for idx, (image, target) in enumerate(train_loader):
        print("idx :{} image: {} target: {}".format(idx, image.size(), target.size()))
        save_bboxim(image, target, "test_im.jpg", "/res/")
        break


if __name__ == '__main__':
    main()
