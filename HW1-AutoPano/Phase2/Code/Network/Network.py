"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

import torch.nn as nn
import sys
import torch
import numpy as np
import torch.nn.functional as F
import lightning as pl
import kornia  # You can use this to get the transform and warp in this project
# from tensorDLT import TensorDLT

# Don't generate pyc codes
sys.dont_write_bytecode = True


# def LossFn(delta, img_a, patch_b, corners):
#     ###############################################
#     # Fill your loss function of choice here!
#     ###############################################

#     ###############################################
#     # You can use kornia to get the transform and warp in this project
#     # Bonus if you implement it yourself
#     ###############################################
#     loss = ...
#     return loss

def LossFn(PredicatedCoordinatesBatch, CoordinatesBatch):
    """
    Inputs:
    PredicatedCoordinatesBatch - Predicted Coordinates of size (MiniBatchSize, 4)
    CoordinatesBatch - Actual Coordinates of size (MiniBatchSize, 4)
    Outputs:
    Loss - Smooth L1 Loss between the Predicted and Actual Coordinates
    """
    # L2 Loss / MSE loss
    criterion = torch.nn.MSELoss()
    Loss = criterion(PredicatedCoordinatesBatch, CoordinatesBatch)
    return Loss


class HomographyModel(pl.LightningModule):
    def __init__(self):
        super(HomographyModel, self).__init__()
        # self.hparams = hparams
        self.model = SupModel()

    def forward(self, b):
        return self.model(b)

    def training_step(self, batch, batch_idx):
        img_a, patch_a, patch_b, corners, gt = batch
        delta = self.model(patch_a, patch_b)
        loss = LossFn(delta, img_a, patch_b, corners)
        logs = {"loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, batch, labels):
        pred = self.model(batch)
        loss = LossFn(pred, labels)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        # logs = {"val_loss": avg_loss.item()}
        # return {"avg_val_loss": avg_loss, "log": logs}
        return {"avg_val_loss": avg_loss.item()}    
    
    

    

class UnSupModel(HomographyModel):
    def __init__(self, InputSize, OutputSize):
        """
        Inputs:
        InputSize - Size of the Input
        OutputSize - Size of the Output
        """
        super(HomographyModel).__init__()
        #############################
        # Fill your network initialization of choice here!
        #############################
        self.tensorDLT = TensorDLT()

    def forward(self, corners_a, preds, img_a_batch):
        """
        Input:
        xa is a MiniBatch of the image a
        xb is a MiniBatch of the image b
        Outputs:
        out - output of the network
        """
        #############################
        # Fill your network structure of choice here!
        #############################
        H = self.tensorDLT(corners_a, preds)
        H_inv = torch.inverse(H)
        out = kornia.geometry.transform.warp_perspective(img_a_batch, H_inv, dsize=(128, 128))
        
        return out
    



class SupModel(nn.Module):
    def __init__(self, InputSize=2, OutputSize=8):
        """
        Inputs:
        InputSize - Size of the Input
        OutputSize - Size of the Output
        """
        super().__init__()
        #############################
        # Fill your network initialization of choice here!
        #############################
        self.conv1 = nn.Conv2d(InputSize, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(128)
        
        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm2d(128)
        
        self.dropout = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(128*16*16, 1024)
        self.fc2 = nn.Linear(1024, OutputSize)
        

    def forward(self, x):
        """
        Input:
        x is a MiniBatch of the image a
        Outputs:
        out - output of the network
        """
        #############################
        # Fill your network structure of choice here!
        #############################
        # x = x.view(-1, 2, 128, 128)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        x = self.maxpool(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.maxpool(x)
        
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.maxpool(x)
        
        x = F.relu(self.bn7(self.conv7(x)))
        x = self.conv8(x)
        
        x = self.dropout(x)

        
        # print(f"Size {x.size()}")
        
        x = self.fc1(x.reshape(-1, 128*16*16))
        out = self.fc2(x)
        
        return out
    
    


