from pytorch_lightning.core.lightning import LightningModule
import torch
import torch.nn as nn

from pathlib import Path

from visualize import save_res_im


class Yolo(LightningModule):

    def __init__(self, net, loss_func, train_root, val_root, target_file, lr, epochs, pred_classes,
            batch_size, im_save, output_size):
        super().__init__()
        self.net = net
        # self.criterion = loss_func
        self.train_root = train_root
        self.val_root = val_root,
        self.target_file = target_file
        self.lr = lr
        self.epochs = epochs
        self.pred_classes = pred_classes
        self.batch_size = batch_size
        self.output_size = output_size

        # self.im_save = Path("/res/res_im").mkdir(exist_ok=True, parents=True)
        self.im_save = Path("/res/res_im")
        self.im_save.mkdir(exist_ok=True, parents=True)
        self.now_epoch = 1

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        # loss = self.criterion(logits, y)
        loss = nn.MSELoss()(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        # loss = self.criterion(logits, y)
        loss = nn.MSELoss()(y, logits)
        save_res_im(x, logits, y, "res_{}.jpg".format(self.now_epoch), str(self.im_save), self.output_size)
        self.now_epoch += 1
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.net.parameters(), lr=self.lr)
