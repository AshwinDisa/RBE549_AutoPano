#!/usr/bin/evn python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""


# Code starts here:

import numpy as np
import cv2
import os
import torch
from Network.Network  import HomographyModel
from Network.Network import LossFn
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    print("Cuda (GPU support) is available and enabled!")
    device = torch.device("cuda")
else:
    print("Cuda (GPU support) is not available :(")
    device = torch.device("cpu")

# Add any python libraries here
def get_patch(image, x, y, patch_size):
    # Get the patch from the image
    # x, y is the center of the patch
    # patch_size is the size of the patch
    # return the patch
    return image[x - patch_size // 2:x + patch_size // 2, y - patch_size // 2:y + patch_size // 2]

def generate_data(image, image_name, patch_size=128, num_patches=10, perturbation_factor=24):
    
    h, w = image.shape[0], image.shape[1]
    roIh, roIw = [160, h - 160], [160, w - 160]
   
    # randomly select a patch from the image
    x = np.random.randint(roIh[0], roIh[1])
    y = np.random.randint(roIw[0], roIw[1])
    
    original_patch = get_patch(image, x, y, patch_size)
    
    corners = np.array([[x - patch_size // 2, y - patch_size // 2],
                        [x + patch_size // 2, y - patch_size // 2],
                        [x - patch_size // 2, y + patch_size // 2],
                        [x + patch_size // 2, y + patch_size // 2]])
    
    # add random perturbation to the patch [-perbutation_factor, perbutation_factor]
    perturbation = np.random.randint(-perturbation_factor, perturbation_factor, size=(4,2))
    
    perturbed_corners = corners + perturbation
    corners = corners.astype(np.float32)
    perturbed_corners = perturbed_corners.astype(np.float32)
    
    # get homography from the corners of the patch
    homography_ab = cv2.getPerspectiveTransform(corners, perturbed_corners)
    
    # warp image using inverse homography
    warped_image = cv2.warpPerspective(image, np.linalg.inv(homography_ab), (h, w), flags=cv2.INTER_LINEAR)
    # print(f'Warped Image shape: {warped_image.shape}')
    
    # cv2.imshow('frame', image) # Initial Capture
    # cv2.imshow('frame1', warped_image) # Transformed Capture
    # cv2.waitKey(0)
    # # closing all open windows 
    # cv2.destroyAllWindows()
    
    # Get the original patch, warped patch, and homography and save them to the output directory
    warped_patch = get_patch(warped_image, x, y, patch_size)
    # print(f'Warped patch shape: {warped_patch.shape}')
    
    
    # cv2.imshow('frame', original_patch) # Initial Capture
    # cv2.imshow('frame1', warped_patch) # Transformed Capture
    # cv2.waitKey(0)
    # # closing all open windows 
    # cv2.destroyAllWindows()
    
  
    
    return original_patch, warped_patch, perturbation, corners, homography_ab


def generate_DataSet():

    imgdir = 'Phase2/Data/Train/'
    # imgdir = 'Phase2/Data/Val/'
    
    OrigOutputDir = 'Phase2/Data/GeneratedDataset/Train/Original/'
    WarpOutputDir = 'Phase2/Data/GeneratedDataset/Train/Warped/'
    LabelsOutputDir = 'Phase2/Data/GeneratedDataset/Train/'
    
    # OrigOutputDir = 'Phase2/Data/GeneratedDataset/Val/Original/'
    # WarpOutputDir = 'Phase2/Data/GeneratedDataset/Val/Warped/'
    # LabelsOutputDir = 'Phase2/Data/GeneratedDataset/Val/'
    
    if not os.path.exists(OrigOutputDir):    
        os.makedirs(OrigOutputDir)
    if not os.path.exists(WarpOutputDir):   
        os.makedirs(WarpOutputDir)
    
    perturbation_list = []
    homography_list = []
    corners_a = []
    train_samples = 5000
    num_patches = 10
    count = 1
    
    
    with open('Phase2/Data/GeneratedDataset/Train/train_names.txt', 'w') as train_file:
    # with open('Phase2/Data/GeneratedDataset/Val/val_names.txt', 'w') as train_file:
    
        
        for i in range(train_samples):
            image_path = f'{imgdir}{i+1}.jpg'
            
            print(f'Processing image: {image_path}')
            
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            h, w = img.shape[0], img.shape[1]
            image = cv2.resize(img, (480, 480))
            
            for j in range(num_patches):

                # Generate 5 patches from each image
                original_patch, warped_patch, perturbation, corners, homography_ab = generate_data(image, i+1)
                
                cv2.imwrite(f'{OrigOutputDir}{count}_{i}.jpg', original_patch)
                cv2.imwrite(f'{WarpOutputDir}{count}_{i}.jpg', warped_patch)
                perturbation_list.append(perturbation)
                homography_list.append(homography_ab)
                corners_a.append(corners)
                train_file.write(f'Val/{count}_{i}\n')
                
                count += 1
    
    np.save(f'{LabelsOutputDir}Labels.npy', perturbation_list)
    np.save(f'{LabelsOutputDir}Homography.npy', homography_list)
    np.save(f'{LabelsOutputDir}Corners.npy', corners_a)


        
def get_warped_image(image1, image2, h_mat):
    
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]])
    pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]])
    pts2_ = cv2.perspectiveTransform(pts2.reshape(-1, 1, 2), h_mat).reshape(-1, 2)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0))
    [xmax, ymax] = np.int32(pts.max(axis=0))
    t = [-xmin, -ymin]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])

    result = cv2.warpPerspective(image1, Ht.dot(h_mat), (xmax - xmin, ymax - ymin), flags=cv2.INTER_LINEAR)

    result_region = result[t[1]:h2 + t[1], t[0]:w2 + t[0]]
    
    if result_region.shape[0] != h2 or result_region.shape[1] != w2:
        result_region = result_region[:h2, :w2]

    result[t[1]:h2 + t[1], t[0]:w2 + t[0]] = image2

    print("result shape: ", result.shape)

    return result
    

def main():
    # Add any Command Line arguments here
    # Parser = argparse.ArgumentParser()
    # Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')

    # Args = Parser.parse_args()
    # NumFeatures = Args.NumFeatures

    # generate_DataSet()

    # Supervised Model
    sup_model = HomographyModel().to(device)

    checkpoint = torch.load('/home/ashd/WPI Spring 2024/Computer Vision/HW1/YourDirectoryID_p1/19model.ckpt')
    sup_model.load_state_dict(checkpoint['model_state_dict'])

    sup_model.eval()

    """
    Read a set of images for Panorama stitching
    """

    imagepath = "P1TestSet/Phase2Pano/tower"
    # imagepath = 'P1TestSet/Phase1/TestSet3'
    images = []
    
    # for filename in os.listdir(imagepath):
    for i in range(len(os.listdir(imagepath))):
        img = cv2.imread(os.path.join(imagepath, f"{i}.jpg"))
        images.append(img)

    stitched_panorama = images[0]

    for i in range(len(images)-1):

        gray_pano = cv2.cvtColor(stitched_panorama, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(images[i+1], cv2.COLOR_BGR2GRAY)
        #resize them to 480x480

        gray_pano_resized = cv2.resize(gray_pano, (128, 128))
        gray_resized = cv2.resize(gray, (128, 128))

        I1A = np.expand_dims(gray_pano_resized, axis=2)
        I1B = np.expand_dims(gray_resized, axis=2)
        

        I1 = np.float32(np.concatenate((I1A, I1B), axis=2)/255.0)
        I1 = np.expand_dims(I1, axis=0)

        I1 = torch.tensor(I1).to(device)
        I1 = I1.permute(0, 3, 1, 2)
        # print(I1.size())

        """
        Obtain Homography using Deep Learning Model (Supervised and Unsupervised)
        """

        out = sup_model(I1)

        out = out.detach().cpu().numpy().reshape(8)

        # Unsupervised Model

        # get homography
        # image1_corners is a 8, 1 array [x, y, x, y, x, y, x, y]
        image1_corners = np.array([0, 0, 128, 0, 0, 128, 128, 128], dtype=np.float32)
        # print(out.T.shape)
        perturbed_corners = image1_corners + out.T
        # print(perturbed_corners)

        perturbed_corners = perturbed_corners.reshape(-1, 2)
        image1_corners = image1_corners.reshape(-1, 2)

        homography_ab = cv2.getPerspectiveTransform(image1_corners, perturbed_corners)
        # get_warp function

        # scaling_factor_x = 128 / 1080
        # scaling_factor_y = 128 / 1920

        # S = np.array([[1/scaling_factor_x, 0, 0], [0, 1/scaling_factor_y, 0], [0, 0, 1]])

        # homography_ab_original = S @ homography_ab
        stitched_panorama = get_warped_image(stitched_panorama, images[i+1], homography_ab/10)

        plt.imshow(cv2.cvtColor(stitched_panorama, cv2.COLOR_BGR2RGB))
        plt.show()
    


if __name__ == "__main__":
    main()
