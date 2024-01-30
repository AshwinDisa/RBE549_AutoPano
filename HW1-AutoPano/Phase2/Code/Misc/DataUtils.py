"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

import os
import cv2
import numpy as np
import random
import skimage
import PIL
import sys

# Don't generate pyc codes
sys.dont_write_bytecode = True


def SetupAll(BasePath, CheckPointPath):
    """
    Inputs:
    BasePath is the base path where Images are saved without "/" at the end
    CheckPointPath - Path to save checkpoints/model
    Outputs:
    DirNamesTrain - Variable with Subfolder paths to train files
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    ImageSize - Size of the image
    NumTrainSamples - length(Train)
    NumTestRunsPerEpoch - Number of passes of Val data with MiniBatchSize
    Trainabels - Labels corresponding to Train
    NumClasses - Number of classes
    """
    # Setup DirNames
    # DirNamesTrain = None
    DirNamesTrain = SetupDirNames(os.path.join(BasePath, 'Train/train_names.txt'))
    DirNamesVal = SetupDirNames(os.path.join(BasePath, 'Val/val_names.txt'))        

    # Read and Setup Labels
    LabelsPathTrain = './Phase2/Data/GeneratedDataset/Train/'
    TrainLabels = np.load(os.path.join(LabelsPathTrain, 'Labels.npy'), allow_pickle=True)
    
    LabelsPathTrain = './Phase2/Data/GeneratedDataset/Train/'
    TrainCorners = np.load(os.path.join(LabelsPathTrain, 'Corners.npy'), allow_pickle=True)
    
    # Read and Setup Labels
    LabelsPathVal = './Phase2/Data/GeneratedDataset/Val/'
    ValLabels = np.load(os.path.join(LabelsPathVal, 'Labels.npy'), allow_pickle=True)
    
    LabelsPathVal = './Phase2/Data/GeneratedDataset/Val/'
    ValCorners = np.load(os.path.join(LabelsPathVal, 'Corners.npy'), allow_pickle=True)
    
    # TrainLabels = ReadLabels(LabelsPathTrain)

    # If CheckPointPath doesn't exist make the path
    if not (os.path.isdir(CheckPointPath)):
        os.makedirs(CheckPointPath)

    # Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    SaveCheckPoint = 1000
    # Number of passes of Val data with MiniBatchSize
    NumTestRunsPerEpoch = 5

    # Image Input Shape
    ImageSize = [128, 128, 2]
    NumTrainSamples = len(DirNamesTrain)
    NumValSamples = len(DirNamesVal)

    # Number of classes
    NumClasses = 10

    return (
        DirNamesTrain,
        DirNamesVal,
        SaveCheckPoint,
        ImageSize,
        NumTrainSamples,
        NumValSamples,
        TrainLabels,
        ValLabels,
        NumClasses,
        TrainCorners,
        ValCorners
    )


def ReadLabels(LabelsPathTrain):
    if not (os.path.isfile(LabelsPathTrain)):
        print("ERROR: Train Labels do not exist in " + LabelsPathTrain)
        sys.exit()
    else:
        TrainLabels = open(LabelsPathTrain, "r")
        TrainLabels = TrainLabels.read()
        TrainLabels = list(map(float, TrainLabels.split()))

    return TrainLabels


def SetupDirNames(BasePath):
    """
    Inputs:
    BasePath is the base path where Images are saved without "/" at the end
    Outputs:
    Writes a file ./TxtFiles/DirNames.txt with full path to all image files without extension
    """
    DirNamesTrain = ReadDirNames(os.path.join(BasePath))

    return DirNamesTrain


def ReadDirNames(ReadPath):
    """
    Inputs:
    ReadPath is the path of the file you want to read
    Outputs:
    DirNames is the data loaded from ./TxtFiles/DirNames.txt which has full path to all image files without extension
    """
    # Read text files
    DirNames = open(ReadPath, "r")
    DirNames = DirNames.read()
    DirNames = DirNames.split()
    return DirNames
