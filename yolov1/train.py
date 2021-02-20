import os
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
import hydra

from net import resnet50
from loss import yoloLoss
from dataloader import yoloDataset
from yolo import Yolo


def get_output_size(image_size):
    """
    画像の出力サイズを計算する

    """
    for _ in range(5):
        image_size = (image_size[0]//2,image_size[1]//2)

    return image_size


@hydra.main("./config.yml")
def main(cfg):
    output_size = get_output_size(cfg.image_size)

    # def net
    net = resnet50(pred_classes=cfg.pred_classes)
    resnet = models.resnet50(pretrained=True)
    new_state_dict = resnet.state_dict()
    dd = net.state_dict()
    for k in new_state_dict.keys():
        if k in dd.keys() and not k.startswith('fc'):
            dd[k] = new_state_dict[k]
    net.load_state_dict(dd)
    yolo = Yolo(net, yoloLoss(5, 0.5, cfg.pred_classes, output_size), cfg.train_im, cfg.val_im, cfg.target_f, cfg.lr,
            cfg.epochs, cfg.pred_classes, cfg.batch_size, cfg.im_save)

    # def dataloader

    dataset = yoloDataset(root=cfg.train_im, list_file=[cfg.target_f],
            train=True, transform=[transforms.ToTensor()],
            pred_classes=cfg.pred_classes, image_size=cfg.image_size, output_size=output_size)
    trainloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=6)

    dataset = yoloDataset(root=cfg.train_im, list_file=[cfg.target_f],
            train=False, transform=[transforms.ToTensor()],
            pred_classes=cfg.pred_classes, image_size=cfg.image_size, output_size=output_size)
    valloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=6)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='/res/model/',
        filename='yolov1-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
    )

    trainer = Trainer(
            logger=False,
            max_epochs=cfg.epochs,
            gpus=1,
            # reload_dataloaders_evey_epoch=True,
            callbacks=[checkpoint_callback]
    )

    trainer.fit(yolo, trainloader, valloader, )


if __name__ == "__main__":
    main()
