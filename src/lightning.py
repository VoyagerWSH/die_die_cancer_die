import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchvision
from src.cindex import concordance_index
from math import floor
import random

from torchvision.models import resnet18

class Classifer(pl.LightningModule):
    def __init__(self, num_classes=9, init_lr=1e-4, optimizer="Adam", loss="Cross Entropy"):
        super().__init__()
        self.init_lr = init_lr
        self.num_classes = num_classes
        self.optimizer = optimizer

        # define loss
        if loss == "Cross Entropy":
            self.loss = nn.CrossEntropyLoss()
        if loss == "Binary Cross Entropy":
            self.loss = nn.BCELoss()

        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.auc = torchmetrics.AUROC(task="binary" if self.num_classes == 2 else "multiclass", num_classes=self.num_classes)

        # store pred
        self.training_outputs = []
        self.validation_outputs = []
        self.test_outputs = []

    def get_xy(self, batch):
        if isinstance(batch, list):
            x, y = batch[0], batch[1]
        else:
            assert isinstance(batch, dict)
            x, y = batch["x"], batch["y_seq"][:,0]
        return x, y.to(torch.long).view(-1)

    def training_step(self, batch, batch_idx):
        x, y = self.get_xy(batch)

        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)

        self.log('train_acc', self.accuracy(y_hat, y), prog_bar=True)
        self.log('train_loss', loss, prog_bar=True)

        ## Store the predictions and labels for use at the end of the epoch
        self.training_outputs.append({
            "y_hat": y_hat,
            "y": y
        })
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = self.get_xy(batch)

        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)

        self.log('val_loss', loss, sync_dist=True, prog_bar=True)
        self.log("val_acc", self.accuracy(y_hat, y), sync_dist=True, prog_bar=True)

        self.validation_outputs.append({
            "y_hat": y_hat,
            "y": y
        })
        return loss

    def test_step(self, batch, batch_idx):
        x, y = self.get_xy(batch)
        
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)

        self.log('test_loss', loss, sync_dist=True, prog_bar=True)
        self.log('test_acc', self.accuracy(y_hat, y), sync_dist=True, prog_bar=True)

        self.test_outputs.append({
            "y_hat": y_hat,
            "y": y
        })
        return loss
    def on_train_epoch_end(self):
        y_hat = torch.cat([o["y_hat"] for o in self.training_outputs])
        y = torch.cat([o["y"] for o in self.training_outputs])
        if self.num_classes == 2:
            probs = F.softmax(y_hat, dim=-1)[:,-1]
        else:
            probs = F.softmax(y_hat, dim=-1)
        self.log("train_auc", self.auc(probs, y.view(-1)), sync_dist=True, prog_bar=True)
        self.training_outputs = []

    def on_validation_epoch_end(self):
        y_hat = torch.cat([o["y_hat"] for o in self.validation_outputs])
        y = torch.cat([o["y"] for o in self.validation_outputs])
        if self.num_classes == 2:
            probs = F.softmax(y_hat, dim=-1)[:,-1]
        else:
            probs = F.softmax(y_hat, dim=-1)
        self.log("val_auc", self.auc(probs, y.view(-1)), sync_dist=True, prog_bar=True)
        self.validation_outputs = []

    def on_test_epoch_end(self):
        y_hat = torch.cat([o["y_hat"] for o in self.test_outputs])
        y = torch.cat([o["y"] for o in self.test_outputs])

        if self.num_classes == 2:
            probs = F.softmax(y_hat, dim=-1)[:,-1]
        else:
            probs = F.softmax(y_hat, dim=-1)

        self.log("test_auc", self.auc(probs, y.view(-1)), sync_dist=True, prog_bar=True)
        self.test_outputs = []

    def configure_optimizers(self):
        if self.optimizer == "Adam":
            return torch.optim.Adam(self.parameters(), lr=self.init_lr)
        elif self.optimizer == "AdamW":
            return torch.optim.AdamW(self.parameters(), lr=self.init_lr)
        elif self.optimizer == "SGD":
            return torch.optim.SGD(self.parameters(), lr=self.init_lr)




class MLP(Classifer):
    def __init__(self, in_features=28*28*3, num_classes = 9, n_fc=2, hidden_dim=1024, use_bn=True, init_lr = 1e-3, dropout_p=0, optimizer = "Adam", loss = "Cross Entropy",**kwargs):
        super().__init__(num_classes=num_classes, init_lr=init_lr, optimizer=optimizer, loss=loss)
        self.save_hyperparameters()
        
        self.fc_layers = nn.ModuleList()
        self.use_bn = use_bn

        out_features = hidden_dim
        for i in range(n_fc):
            self.fc_layers.append(nn.Linear(in_features, out_features))
            if self.use_bn:
                self.fc_layers.append(nn.BatchNorm1d(out_features))

            self.fc_layers.append (nn.ReLU())
            self.fc_layers.append(nn.Dropout(dropout_p))
            in_features = out_features
            
        self.fc_layers.append(nn.Linear(out_features, num_classes))

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, channels*width*height)

        for layer in self.hidden_layers:
            x = layer(x)

        return x

"""
Utility function for computing output of convolutions
takes a tuple of (h,w) and returns a tuple of (h,w)
"""
def conv_output_shape(dim, kernel_size=1, stride=1, padding=0, dilation=1):
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    dim = floor( ((dim + (2 * padding) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
    return dim

class CNN(Classifer):
    def __init__(self, conv_layers=[], in_dim = 28, num_classes = 9, pooling=None, use_bn=True, init_lr = 1e-3, optimizer = "Adam", loss = "Cross Entropy",**kwargs):
        super().__init__(num_classes=num_classes, init_lr=init_lr, optimizer=optimizer, loss=loss)
        self.save_hyperparameters()
        
        self.conv_layers = nn.ModuleList()
        self.use_bn = use_bn
        self.dim = in_dim
        self.num_classes = num_classes
        
        # conv -> norm -> relu -> pool
        for i in range(len(conv_layers)-2):
            self.conv_layers.append(nn.Conv2d(in_channels=conv_layers[i], out_channels=conv_layers[i+1], kernel_size=3, stride=1, padding=0, padding_mode='zeros'))
            self.dim = conv_output_shape(self.dim, kernel_size=3, stride=1, padding=0)
            
            if use_bn:
                self.conv_layers.append(nn.BatchNorm2d(conv_layers[i+1], eps=1e-5, momentum=0.1))
            
            self.conv_layers.append(nn.ReLU())

            if pooling == "max":
                self.conv_layers.append(nn.MaxPool2d(3, stride=2))
                self.dim = conv_output_shape(self.dim, kernel_size=3, stride=2)
            if pooling == "avg":
                self.conv_layers.append(nn.AvgPool2d(3, stride=2))
                self.dim = conv_output_shape(self.dim, kernel_size=3, stride=2)
        
        self.conv_layers.append(nn.Conv2d(in_channels=conv_layers[-2], out_channels=conv_layers[-1], kernel_size=3, stride=1, padding=0, padding_mode='zeros'))
        self.dim = conv_output_shape(self.dim, kernel_size=3, stride=1, padding=0)
        
        # global pooling to obtain C*1*1 image
        self.conv_layers.append(nn.MaxPool2d(self.dim))
        self.conv_layers.append(nn.Linear(conv_layers[-1], self.num_classes))
    
    def forward(self, x):
        for layer in self.conv_layers[:-1]:
            x = layer(x)
        return self.conv_layers[-1](x.flatten(1))

class Resnet(Classifer):
    def __init__(self, num_classes = 9, use_bn=True, init_lr = 1e-3, optimizer = "Adam", loss = "Cross Entropy", pre_train = True, dropout_p=0, n_fc = 2, **kwargs):
        super().__init__(num_classes=num_classes, init_lr=init_lr, optimizer=optimizer, loss=loss)
        self.save_hyperparameters()

        self.use_bn = use_bn
        self.num_classes = num_classes
        self.fc_layers = nn.ModuleList()
        self.pre_train = pre_train

        if pre_train:
            self.backbone = resnet18(weights="DEFAULT")
        else:
            self.backbone = resnet18(weights=None)
        in_features = self.backbone.fc.out_features
        out_features = 512

        self.fc_layers.append (nn.ReLU())
        for i in range(n_fc):
            self.fc_layers.append(nn.Linear(in_features, out_features))
            self.fc_layers.append (nn.ReLU())
            self.fc_layers.append(nn.Dropout(dropout_p))
            in_features = out_features
        self.fc_layers.append(nn.Linear(out_features, num_classes))

    def forward(self, x):
        x = self.backbone(x)
        for layer in self.fc_layers:
            x = layer(x)

        return x


class CNN_3D(Classifer):
    def __init__(self, conv_layers=[], in_dim = 256, in_depth = 200, num_classes = 2, pooling=None, use_bn=True, init_lr = 1e-3, optimizer = "Adam", loss = "Binary Cross Entropy",**kwargs):
        super().__init__(num_classes=num_classes, init_lr=init_lr, optimizer=optimizer, loss=loss)
        self.save_hyperparameters()
        
        self.conv_layers = nn.ModuleList()
        self.use_bn = use_bn
        self.dim = in_dim
        self.depth = in_depth
        self.num_classes = num_classes
        
        # conv -> norm -> relu -> pool
        for i in range(len(conv_layers)-2):
            self.conv_layers.append(nn.Conv3d(in_channels=conv_layers[i], out_channels=conv_layers[i+1], kernel_size=5, stride=1, padding=0, padding_mode='zeros'))
            self.dim = conv_output_shape(self.dim, kernel_size=5, stride=1, padding=0)
            self.depth = conv_output_shape(self.depth, kernel_size=5, stride=1, padding=0)
            
            if use_bn:
                self.conv_layers.append(nn.BatchNorm3d(conv_layers[i+1], eps=1e-5, momentum=0.1))
            
            self.conv_layers.append(nn.ReLU())

            if pooling == "max":
                self.conv_layers.append(nn.MaxPool3d(5, stride=2))
                self.dim = conv_output_shape(self.dim, kernel_size=5, stride=2)
                self.depth = conv_output_shape(self.depth, kernel_size=5, stride=2)
            if pooling == "avg":
                self.conv_layers.append(nn.AvgPool3d(5, stride=2))
                self.dim = conv_output_shape(self.dim, kernel_size=5, stride=2)
                self.depth = conv_output_shape(self.depth, kernel_size=5, stride=2)
        
        self.conv_layers.append(nn.Conv3d(in_channels=conv_layers[-2], out_channels=conv_layers[-1], kernel_size=5, stride=2, padding=0, padding_mode='zeros'))
        self.dim = conv_output_shape(self.dim, kernel_size=5, stride=2, padding=0)
        self.depth = conv_output_shape(self.depth, kernel_size=5, stride=2, padding=0)
    
        # global pooling to obtain C*1*1*1 image
        self.conv_layers.append(nn.MaxPool3d((self.depth, self.dim, self.dim)))
        self.conv_layers.append(nn.Linear(conv_layers[-1], 1))
    
    def forward(self, x):
        for layer in self.conv_layers[:-1]:
            x = layer(x)
        return nn.Sigmoid(self.conv_layers[-1](x.flatten(1)))
    
class Resnet_3D(Classifer):
    def __init__(self, num_classes = 2, use_bn=True, init_lr = 1e-3, optimizer = "Adam", loss = "Binary Cross Entropy", pre_train = True, dropout_p=0, n_fc = 2, **kwargs):
        super().__init__(num_classes=num_classes, init_lr=init_lr, optimizer=optimizer, loss=loss)
        self.save_hyperparameters()

        self.use_bn = use_bn
        self.num_classes = num_classes
        self.fc_layers = nn.ModuleList()
        self.pre_train = pre_train

        if pre_train:
            self.backbone = resnet18(weights="DEFAULT")
        else:
            self.backbone = resnet18(weights=None)
        in_features = self.backbone.fc.out_features
        out_features = 512

        self.fc_layers.append (nn.ReLU())
        for i in range(n_fc):
            self.fc_layers.append(nn.Linear(in_features, out_features))
            self.fc_layers.append (nn.ReLU())
            self.fc_layers.append(nn.Dropout(dropout_p))
            in_features = out_features
        self.fc_layers.append(nn.Linear(out_features, num_classes))

    def forward(self, x):
        _, C, D, H, W = x.shape
        x = nn.Conv3d(in_channels=C, out_channels=C, kernel_size=(D, 1, 1), stride=1)
        x = x.squeeze(2)
        if self.use_bn:
            x = nn.BatchNorm2d(C, eps=1e-5, momentum=0.1)
        x = nn.ReLU(x)

        x = self.backbone(x)
        for layer in self.fc_layers:
            x = layer(x)

        return x


NLST_CENSORING_DIST = {
    "0": 0.9851928130104401,
    "1": 0.9748317321074379,
    "2": 0.9659923988537479,
    "3": 0.9587252204657843,
    "4": 0.9523590830936284,
    "5": 0.9461840310101468,
}
class RiskModel(Classifer):
    def __init__(self, input_num_chan=1, num_classes=2, init_lr = 1e-3, max_followup=6, **kwargs):
        super().__init__(num_classes=num_classes, init_lr=init_lr)
        self.save_hyperparameters()

        self.hidden_dim = 512

        ## Maximum number of followups to predict (set to 6 for full risk prediction task)
        self.max_followup = max_followup

        # TODO: Initalize components of your model here
        raise NotImplementedError("Not implemented yet")



    def forward(self, x):
        raise NotImplementedError("Not implemented yet")

    def get_xy(self, batch):
        """
            x: (B, C, D, W, H) -  Tensor of CT volume
            y_seq: (B, T) - Tensor of cancer outcomes. a vector of [0,0,1,1,1, 1] means the patient got between years 2-3, so
            had cancer within 3 years, within 4, within 5, and within 6 years.
            y_mask: (B, T) - Tensor of mask indicating future time points are observed and not censored. For example, if y_seq = [0,0,0,0,0,0], then y_mask = [1,1,0,0,0,0], we only know that the patient did not have cancer within 2 years, but we don't know if they had cancer within 3 years or not.
            mask: (B, D, W, H) - Tensor of mask indicating which voxels are inside an annotated cancer region (1) or not (0).
                TODO: You can add more inputs here if you want to use them from the NLST dataloader.
                Hint: You may want to change the mask definition to suit your localization method

        """
        return batch['x'], batch['y_seq'][:, :self.max_followup], batch['y_mask'][:, :self.max_followup], batch['mask']

    def step(self, batch, batch_idx, stage, outputs):
        x, y_seq, y_mask, region_annotation_mask = self.get_xy(batch)

        # TODO: Get risk scores from your model
        y_hat = None ## (B, T) shape tensor of risk scores.
        # TODO: Compute your loss (with or without localization)
        loss = None

        raise NotImplementedError("Not implemented yet")
        
        # TODO: Log any metrics you want to wandb
        metric_value = -1
        metric_name = "dummy_metric"
        self.log('{}_{}'.format(stage, metric_name), metric_value, prog_bar=True, on_epoch=True, on_step=True, sync_dist=True)

        # TODO: Store the predictions and labels for use at the end of the epoch for AUC and C-Index computation.
        outputs.append({
            "y_hat": y_hat, # Logits for all risk scores
            "y_mask": y_mask, # Tensor of when the patient was observed
            "y_seq": y_seq, # Tensor of when the patient had cancer
            "y": batch["y"], # If patient has cancer within 6 years
            "time_at_event": batch["time_at_event"] # Censor time
        })

        return loss
    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train", self.training_outputs)

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "val", self.validation_outputs)
    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "test", self.test_outputs)

    def on_epoch_end(self, stage, outputs):
        y_hat = F.sigmoid(torch.cat([o["y_hat"] for o in outputs]))
        y_seq = torch.cat([o["y_seq"] for o in outputs])
        y_mask = torch.cat([o["y_mask"] for o in outputs])

        for i in range(self.max_followup):
            '''
                Filter samples for either valid negative (observed followup) at time i
                or known pos within range i (including if cancer at prev time and censoring before current time)
            '''
            valid_probs = y_hat[:, i][(y_mask[:, i] == 1) | (y_seq[:,i] == 1)]
            valid_labels = y_seq[:, i][(y_mask[:, i] == 1)| (y_seq[:,i] == 1)]
            self.log("{}_{}year_auc".format(stage, i+1), self.auc(valid_probs, valid_labels.view(-1)), sync_dist=True, prog_bar=True)

        y = torch.cat([o["y"] for o in outputs])
        time_at_event = torch.cat([o["time_at_event"] for o in outputs])

        if y.sum() > 0 and self.max_followup == 6:
            c_index = concordance_index(time_at_event.cpu().numpy(), y_hat.detach().cpu().numpy(), y.cpu().numpy(), NLST_CENSORING_DIST)
        else:
            c_index = 0
        self.log("{}_c_index".format(stage), c_index, sync_dist=True, prog_bar=True)

    def on_train_epoch_end(self):
        self.on_epoch_end("train", self.training_outputs)
        self.training_outputs = []

    def on_validation_epoch_end(self):
        self.on_epoch_end("val", self.validation_outputs)
        self.validation_outputs = []

    def on_test_epoch_end(self):
        self.on_epoch_end("test", self.test_outputs)
        self.test_outputs = []
