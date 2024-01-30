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

# Don't generate pyc codes
sys.dont_write_bytecode = True

def LossFn(patch_b_pred, patch_b):
    """
    Inputs:
    PredicatedCoordinatesBatch - Predicted Coordinates of size (MiniBatchSize, 4)
    CoordinatesBatch - Actual Coordinates of size (MiniBatchSize, 4)
    Outputs:
    Loss - Smooth L1 Loss between the Predicted and Actual Coordinates
    """
    # L1 Loss
    criterion = torch.nn.L1Loss()
    Loss = criterion(patch_b_pred, patch_b)
    return Loss

def get_patch_from_corners(image, corners_a, patch_size):
    """
    Extract patches from the image based on the corner coordinates.

    Parameters:
        image: torch.Tensor, input image tensor (shape: B x C x H x W)
        corners_a: torch.Tensor, corner coordinates [x1, y1, x2, y2, x3, y3, x4, y4] (shape: B x 8)
        patch_size: int, size of the patch

    Returns:
        patches: torch.Tensor, extracted patches (shape: B x C x patch_size x patch_size)
    """
    B, C, H, W = image.size()
    half_size = patch_size // 2
    patches = []
    for b in range(B):
        x1, y1, x2, y2, x3, y3, x4, y4 = corners_a[b].squeeze()
        x_min = int(min(x1, x2, x3, x4))
        y_min = int(min(y1, y2, y3, y4))
        
        patch = image[b, :, x_min:x_min + patch_size, y_min:y_min + patch_size]
        patches.append(patch)
    return torch.stack(patches, dim=0)

class HomographyModel(pl.LightningModule):
    def __init__(self):
        super(HomographyModel, self).__init__()
        # self.hparams = hparams
        self.model = UnSupModel()

    def forward(self, b):
        return self.model(b)

    def training_step(self, x, corners_a, img_a_batch):

        patch_a, patch_b = torch.unsqueeze(x[:, 0, :, :], 1), torch.unsqueeze(x[:, 1, :, :],1)
        # print(f"Patch B {patch_b.size()}")
        pred = self.model(x, corners_a, patch_a, img_a_batch)
      
        out = get_patch_from_corners(pred, corners_a, 128)
        # print(f"Out {out}")
        loss = LossFn(out, patch_b)
        return {"loss": loss}

    def validation_step(self, batch, corners_a, img_a_batch):
        
        patch_a, patch_b = torch.unsqueeze(batch[:, 0, :, :], 1), torch.unsqueeze(batch[:, 1, :, :],1)
        pred = self.model(batch, corners_a, patch_a, img_a_batch)

        out = get_patch_from_corners(pred, corners_a, 128)

        loss = LossFn(out, patch_b)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        return {"avg_val_loss": avg_loss.item()}    
     

class UnSupModel(nn.Module):
    def __init__(self, InputSize=2, OutputSize=1):
        """
        Inputs:
        InputSize - Size of the Input
        OutputSize - Size of the Output
        """
        super().__init__()
        #############################
        # Fill your network initialization of choice here!
        #############################
        self.homography = SupModel()
        self.tensorDLT = TensorDLT()
        

    def forward(self, x, corners_a, patch_a_batch, img_a_batch):
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
        preds = self.homography(x)
        H = self.tensorDLT(corners_a, preds)
        H_inv = torch.inverse(H)
        # out = kornia.geometry.transform.warp_perspective(img_a_batch, H_inv, dsize=(128, 128))
        out = kornia.geometry.transform.warp_perspective(img_a_batch, H_inv, dsize=(480, 480))

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
    
    
class TensorDLT(nn.Module):
    def __init__(self):
        super().__init__()

    def tensorDLT(self, corners_a, preds):
        H = torch.tensor([], device=preds.device)
        batch_size = corners_a.shape[0]

        A = torch.zeros(size=(batch_size, 8, 8), dtype=torch.float64, device=preds.device)
        b = torch.zeros(size=(batch_size, 8), dtype=torch.float64, device=preds.device)

        idx_x = torch.tensor([0, 2, 4, 6])
        idx_y = torch.tensor([1, 3, 5, 7])
        zeros = torch.zeros(size=(batch_size, 4), device=preds.device)
        ones = torch.ones(size=(batch_size, 4), device=preds.device)

        u_i, v_i = corners_a.reshape(shape=(-1, 4, 2)).transpose(1, 2).swapaxes(0, 1).to(preds.device)
        u_i_p, v_i_p = preds.reshape(shape=(-1, 4, 2)).transpose(1, 2).swapaxes(0, 1).to(preds.device)

        A[:, idx_x] = torch.stack([zeros, zeros, zeros, -u_i, -v_i, -ones, v_i_p * u_i, v_i_p * v_i], dim=2)
        A[:, idx_y] = torch.stack([u_i, v_i, ones, zeros, zeros, zeros, -u_i_p * u_i, -u_i_p * v_i], dim=2)
        b[:, idx_x] = -v_i_p
        b[:, idx_y] = u_i_p

        b = b.reshape(batch_size, 8, 1)
        ret = torch.linalg.pinv(A) @ b
        ones_2 = torch.ones(size=(batch_size, 1, 1), device=preds.device)
        ret = torch.cat([ret, ones_2], dim=1)
        ret = ret.reshape(shape=(-1, 3, 3))
        return ret

    def forward(self, corners_a, preds):
        return self.tensorDLT(corners_a, preds)


class TensorDLT(nn.Module):
    def __init__(self):
        super().__init__()
        # self.tensorDLT = kornia.geometry.transform.dlt()
        
    def tensorDLT(self, corners_a, preds):
        H = torch.tensor([], device=preds.device)
        batch_size = corners_a.shape[0]

        A = torch.zeros(size=(batch_size,8,8),dtype=torch.float32).to(preds.device)
        b = torch.zeros(size=(batch_size,8),dtype=torch.float32).to(preds.device)

        idx_x = torch.tensor([0,2,4,6])
        idx_y = torch.tensor([1,3,5,7])
        zeros = torch.zeros(size=(batch_size,4)).to(preds.device)
        ones = torch.ones(size=(batch_size,4)).to(preds.device)

        
        u_i,v_i = corners_a.reshape(shape=(-1,4,2)).transpose(1,2).swapaxes(0,1).to(preds.device)
        u_i_p,v_i_p = preds.reshape(shape=(-1,4,2)).transpose(1,2).swapaxes(0,1).to(preds.device)
     
        A[:,idx_x] = torch.stack([zeros,zeros,zeros,-u_i,-v_i,-ones,v_i_p*u_i,v_i_p*v_i],dim=2)
        A[:,idx_y] = torch.stack([u_i,v_i,ones,zeros,zeros,zeros,-u_i_p*u_i,-u_i_p*v_i],dim=2)
        b[:,idx_x] = -v_i_p
        b[:,idx_y] = u_i_p
        
        # print(A)
        # print(b)

        b = b.reshape(batch_size,8,1)
        ret = torch.linalg.pinv(A) @ b
        ones_2 = torch.ones(size=(batch_size,1,1)).to(preds.device)
        ret = torch.cat([ret,ones_2],dim=1)
        ret = ret.reshape(shape=(-1,3,3))
        return ret
    
    def forward(self, corners_a, preds):
        return self.tensorDLT(corners_a, preds)

    