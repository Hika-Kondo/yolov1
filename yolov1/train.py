import os
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
import hydra

from net import resnet50
from loss import yoloLoss
from dataloader import yoloDataset
from visualize import save_res_im


class Yolo(LightningModule):

    def __init__(self, net, loss_func, train_root, val_root, target_file, lr, epochs, pred_classes,
            batch_size, im_save):
        super().__init__()
        self.net = net
        self.criterion = loss_func
        self.train_root = train_root
        self.val_root = val_root,
        self.target_file = target_file
        self.lr = lr
        self.epochs = epochs
        self.pred_classes = pred_classes
        self.batch_size = batch_size

        # self.im_save = Path("/res/res_im").mkdir(exist_ok=True, parents=True)
        self.im_save = Path("/res/res_im")
        self.im_save.mkdir(exist_ok=True, parents=True)
        self.now_epoch = 1
    
    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        save_res_im(x, logits, y, "res_{}.jpg".format(self.now_epoch), str(self.im_save))
        self.now_epoch += 1
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.net.parameters(), lr=self.lr)



@hydra.main("./config.yml")
def main(cfg):

    net = resnet50(pred_classes=cfg.pred_classes)
    resnet = models.resnet50(pretrained=True)
    new_state_dict = resnet.state_dict()
    dd = net.state_dict()
    for k in new_state_dict.keys():
        if k in dd.keys() and not k.startswith('fc'):
            dd[k] = new_state_dict[k]
    net.load_state_dict(dd)
    print(cfg.im_save)
    yolo = Yolo(net, yoloLoss(5, 0.5, cfg.pred_classes), cfg.train_im, cfg.val_im, cfg.target_f, cfg.lr,
            cfg.epochs, cfg.pred_classes, cfg.batch_size, cfg.im_save)

    dataset = yoloDataset(root=cfg.train_im, list_file=[cfg.target_f],
            train=True, transform=[transforms.ToTensor()], pred_classes=cfg.pred_classes)
    trainloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=6)

    dataset = yoloDataset(root=cfg.train_im, list_file=[cfg.target_val],
            train=True, transform=[transforms.ToTensor()], pred_classes=cfg.pred_classes)
    valloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=6)

    trainer = Trainer(
            logger=False,
            max_epochs=cfg.epochs,
            gpus=1,
            checkpoint_callback=False,
            # reload_dataloaders_evey_epoch=True,
            )

    trainer.fit(yolo, trainloader, valloader)


if __name__ == "__main__":
    main()
