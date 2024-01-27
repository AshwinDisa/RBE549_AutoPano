#!/usr/bin/env python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""


# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)
# termcolor, do (pip install termcolor)

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.optim import AdamW
from Network.Network import HomographyModel
from Network.Network import LossFn
from torchvision.transforms import ToTensor, Normalize, Compose
import cv2
import sys
import os
import numpy as np
import random
import skimage
import PIL
import os
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
import time
from Misc.MiscUtils import *
from Misc.DataUtils import *
from torchvision.transforms import ToTensor
import argparse
import shutil
import string
from termcolor import colored, cprint
import math as m
from tqdm import tqdm
import json


if torch.cuda.is_available():
    print("Cuda (GPU support) is available and enabled!")
    device = torch.device("cuda")
else:
    print("Cuda (GPU support) is not available :(")
    device = torch.device("cpu")


def eval_model(model, NumValSamples, MiniBatchSize, ValCoordinates, BasePath, DirNamesVal, ImageSize):

    model.eval()
    result = []
    NumIterationsPerEpoch = int(NumValSamples / MiniBatchSize)
    
    with torch.no_grad():
        for PerEpochCounter in range(NumIterationsPerEpoch):
            
            ValI1Batch, ValCoordinatesBatch = GenerateBatch(
                os.path.join(BasePath, 'Val'), DirNamesVal, ValCoordinates, ImageSize, MiniBatchSize
            )
            ValI1Batch = ValI1Batch.permute(0, 3, 1, 2)
            
            result.append(model.validation_step(ValI1Batch, ValCoordinatesBatch))

    return model.validation_epoch_end(result)

def GenerateBatch(BasePath, DirNamesTrain, TrainCoordinates, ImageSize, MiniBatchSize):
    """
    Inputs:
    BasePath - Path to COCO folder without "/" at the end
    DirNamesTrain - Variable with Subfolder paths to train files
    NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
    TrainCoordinates - Coordinatess corresponding to Train
    NOTE that TrainCoordinates can be replaced by Val/TestCoordinatess for generating batch corresponding to validation (held-out testing in this case)/testing
    ImageSize - Size of the Image
    MiniBatchSize is the size of the MiniBatch
    Outputs:
    I1Batch - Batch of images
    CoordinatesBatch - Batch of coordinates
    """
    I1Batch = []
    CoordinatesBatch = []
  
    ImageNum = 0
    while ImageNum < MiniBatchSize:
        # Generate random image
        RandIdx = random.randint(0, len(DirNamesTrain) - 1)

        # RandImageName = BasePath + os.sep + DirNamesTrain[RandIdx] + ".jpg"
        original_image = os.path.join(BasePath, 'Original', DirNamesTrain[RandIdx].split('/')[1] + ".jpg")
        warped_image = os.path.join(BasePath, 'Warped', DirNamesTrain[RandIdx].split('/')[1] + ".jpg")

        ImageNum += 1

        ##########################################################
        # Add any standardization or data augmentation here!
        ##########################################################
        
        I1A = cv2.imread(original_image, cv2.IMREAD_GRAYSCALE)
        I1B = cv2.imread(warped_image, cv2.IMREAD_GRAYSCALE)
        
        I1A = np.expand_dims(I1A, axis=2)
        I1B = np.expand_dims(I1B, axis=2)
        
        # print(I1A.shape)
        # print(I1B.shape)

        I1 = np.float32(np.concatenate((I1A, I1B), axis=2)/255.0)
        
        Coordinates = np.float32(np.reshape(TrainCoordinates[RandIdx], (8)))

        # Append All Images and Mask
        I1Batch.append(torch.from_numpy(I1))
        CoordinatesBatch.append(torch.tensor(Coordinates))
        
    return torch.stack(I1Batch).to(device), torch.stack(CoordinatesBatch).to(device)


def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print("Number of Epochs Training will run for " + str(NumEpochs))
    print("Factor of reduction in training data is " + str(DivTrain))
    print("Mini Batch Size " + str(MiniBatchSize))
    print("Number of Training Images " + str(NumTrainSamples))
    if LatestFile is not None:
        print("Loading latest checkpoint with the name " + LatestFile)


def TrainOperation(
    DirNamesTrain,
    DirNamesVal,
    TrainCoordinates,
    ValCoordinates,
    NumTrainSamples,
    NumValSamples,
    ImageSize,
    NumEpochs,
    MiniBatchSize,
    SaveCheckPoint,
    CheckPointPath,
    DivTrain,
    LatestFile,
    BasePath,
    LogsPath,
    ModelType,
):
    """
    Inputs:
    ImgPH is the Input Image placeholder
    DirNamesTrain - Variable with Subfolder paths to train files
    TrainCoordinates - Coordinates corresponding to Train/Test
    NumTrainSamples - length(Train)
    ImageSize - Size of the image
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
    LatestFile - Latest checkpointfile to continue training
    BasePath - Path to COCO folder without "/" at the end
    LogsPath - Path to save Tensorboard Logs
        ModelType - Supervised or Unsupervised Model
    Outputs:
    Saves Trained network in CheckPointPath and Logs to LogsPath
    """
    # Predict output with forward pass
    model = HomographyModel().to(device)

    ###############################################
    # Fill your optimizer of choice here!
    ###############################################
    # Optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    Optimizer = torch.optim.Adam(model.parameters(), lr=0.0025, weight_decay=0.0001)

    # Tensorboard
    # Create a summary to monitor loss tensor
    Writer = SummaryWriter(LogsPath)

    if LatestFile is not None:
        CheckPoint = torch.load(CheckPointPath + LatestFile + ".ckpt")
        # Extract only numbers from the name
        StartEpoch = int("".join(c for c in LatestFile.split("a")[0] if c.isdigit()))
        model.load_state_dict(CheckPoint["model_state_dict"])
        print("Loaded latest checkpoint with the name " + LatestFile + "....")
    else:
        StartEpoch = 0
        print("New model initialized....")

    train_metrics_per_epoch = []
    test_metrics_per_epoch = []

    for Epochs in tqdm(range(StartEpoch, NumEpochs)):
        
        NumIterationsPerEpoch = int(NumTrainSamples / MiniBatchSize / DivTrain)
        running_loss = 0
        model.train()

        for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
            I1Batch, CoordinatesBatch = GenerateBatch(
                os.path.join(BasePath, 'Train'), DirNamesTrain, TrainCoordinates, ImageSize, MiniBatchSize
            )
            
            I1Batch = I1Batch.permute(0, 3, 1, 2)
            
            # Predict output with forward pass
            PredicatedCoordinatesBatch = model(I1Batch)
            
            LossThisBatch = LossFn(PredicatedCoordinatesBatch, CoordinatesBatch)
            running_loss += LossThisBatch.item()
            
            # print(f"Epoch: {Epochs}, Iteration: {PerEpochCounter}, Loss: {LossThisBatch}")
            Optimizer.zero_grad()
            LossThisBatch.backward()
            Optimizer.step()

            # Save checkpoint every some SaveCheckPoint's iterations
            # if PerEpochCounter % SaveCheckPoint == 0:
            #     # Save the Model learnt in this epoch
            #     SaveName = (
            #         CheckPointPath
            #         + str(Epochs)
            #         + "a"
            #         + str(PerEpochCounter)
            #         + "model.ckpt"
            #     )

            #     torch.save(
            #         {
            #             "epoch": Epochs,
            #             "model_state_dict": model.state_dict(),
            #             "optimizer_state_dict": Optimizer.state_dict(),
            #             "loss": LossThisBatch,
            #         },
            #         SaveName,
            #     )
            #     print("\n" + SaveName + " Model Saved...")

            # result = model.validation_step(ValI1Batch)
            # # Tensorboard
            # Writer.add_scalar(
            #     "LossEveryIter",
            #     result["val_loss"],
            #     Epochs * NumIterationsPerEpoch + PerEpochCounter,
            # )
            # If you don't flush the tensorboard doesn't update until a lot of iterations!
            Writer.flush()

        # Save model every epoch
        SaveName = CheckPointPath + str(Epochs) + "model.ckpt"
        torch.save(
            {
                "epoch": Epochs,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": Optimizer.state_dict(),
                "loss": LossThisBatch,
            },
            SaveName,
        )
        print("\n" + SaveName + " Model Saved...")

        #Loss per epoch
        print("Finding loss and accuracy on the whole train set...")
        train_result = {'loss': running_loss/NumIterationsPerEpoch}
        train_metrics_per_epoch.append(train_result)
        print(train_result)

        print("Running Eval...")
        val_result = eval_model(model, NumValSamples, MiniBatchSize, ValCoordinates, BasePath, DirNamesVal, ImageSize)
        test_metrics_per_epoch.append(val_result)
        print(val_result)
        
    train_metrics_array = np.array(train_metrics_per_epoch, dtype=np.dtype(object))
    val_metrics_array = np.array(test_metrics_per_epoch, dtype=np.dtype(object))

    file_path = os.path.join(BasePath, 'train_loss.json')
    val_path = os.path.join(BasePath, 'val_loss.json')

    with open(file_path, 'w') as file:
        json.dump(train_metrics_per_epoch, file, indent=2)
        
    with open(val_path, 'w') as file:
        json.dump(test_metrics_per_epoch, file, indent=2)

    print(f"Train and Val losses saved to {file_path} and {val_path}")


def main():
    """
    Inputs:
    # None
    # Outputs:
    # Runs the Training and testing code based on the Flag
    #"""
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--BasePath",
        default="./Phase2/Data/GeneratedDataset/",
        help="Base path of images, Default:/home/lening/workspace/rbe549/YourDirectoryID_p1/Phase2/Data/GeneratedDataset/Train/",
    )
    Parser.add_argument(
        "--CheckPointPath",
        default="../Checkpoints/",
        help="Path to save Checkpoints, Default: ../Checkpoints/",
    )

    Parser.add_argument(
        "--ModelType",
        default="Unsup",
        help="Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup",
    )
    Parser.add_argument(
        "--NumEpochs",
        type=int,
        default=30,
        help="Number of Epochs to Train for, Default:50",
    )
    Parser.add_argument(
        "--DivTrain",
        type=int,
        default=1,
        help="Factor to reduce Train data by per epoch, Default:1",
    )
    Parser.add_argument(
        "--MiniBatchSize",
        type=int,
        default=128,
        help="Size of the MiniBatch to use, Default:1",
    )
    Parser.add_argument(
        "--LoadCheckPoint",
        type=int,
        default=0,
        help="Load Model from latest Checkpoint from CheckPointsPath?, Default:0",
    )
    Parser.add_argument(
        "--LogsPath",
        default="Logs/",
        help="Path to save Logs for Tensorboard, Default=Logs/",
    )

    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    BasePath = Args.BasePath
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath
    ModelType = Args.ModelType

    # Setup all needed parameters including file reading
    (
        DirNamesTrain,
        DirNamesVal,
        SaveCheckPoint,
        ImageSize,
        NumTrainSamples,
        NumValSamples,
        TrainCoordinates,
        ValCoordinates,
        NumClasses,
    ) = SetupAll(BasePath, CheckPointPath)

    # Find Latest Checkpoint File
    if LoadCheckPoint == 1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None

    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

    TrainOperation(
        DirNamesTrain,
        DirNamesVal,
        TrainCoordinates,
        ValCoordinates,
        NumTrainSamples,
        NumValSamples,
        ImageSize,
        NumEpochs,
        MiniBatchSize,
        SaveCheckPoint,
        CheckPointPath,
        DivTrain,
        LatestFile,
        BasePath,
        LogsPath,
        ModelType,
    )


if __name__ == "__main__":
    main()
