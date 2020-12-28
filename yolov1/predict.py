from PIL import Image
from torchvision.transforms import ToTensor
import hydra
import torch

import argparse
import random
from pathlib import Path

from yolo import Yolo
from visualize import save_res_im
from net import resnet50
from loss import yoloLoss


@hydra.main("./config.yml")
def main(cfg):
    parser = argparse.ArgumentParser(description="Yolov1 prediction")
    parser.add_argument("--model_path", type=str,
            help="model checkpoint path", default="/res/model/yolov1-epoch=12-val_loss=0.27.ckpt")
    parser.add_argument("--image_path", type=str,
            help="image path", default="/test")

    args = parser.parse_args()

    net = resnet50(pred_classes=cfg.pred_classes)
    net = Yolo.load_from_checkpoint(checkpoint_path=args.model_path,
            net=net, loss_func=yoloLoss(5, 0.5, cfg.pred_classes),
            train_root=cfg.train_im, val_root=cfg.val_im, 
            target_file=cfg.target_f, lr=cfg.lr,epochs=cfg.epochs,
            pred_classes=cfg.pred_classes, batch_size=cfg.batch_size, im_save=cfg.im_save)

    images = Path(args.image_path).glob("**/*.jpg")

    Path("/res/test").mkdir(exist_ok=True, parents=True)
    out_root = Path("/res/test")

    for image in images:
        img = Image.open(image)
        h,w = random.randint(0,img.size[0]-448), random.randint(0,img.size[1]-448)
        img = img.crop((h,w,h+448,w+448))
        img = ToTensor()(img)
        img = img.view(1,img.size(0), img.size(1), img.size(2))
        output = net(img)
        print(output.size())
        save_res_im(img, output, output,"output.jpg", str(out_root), False)


if __name__ == "__main__":
    main()
