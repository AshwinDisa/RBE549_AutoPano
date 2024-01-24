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

# Add any python libraries here
def get_patch(image, x, y, patch_size):
    # Get the patch from the image
    # x, y is the center of the patch
    # patch_size is the size of the patch
    # return the patch
    return image[x - patch_size // 2:x + patch_size // 2, y - patch_size // 2:y + patch_size // 2]

def generate_data(image, image_name, patch_size=128, num_patches=10, perturbation_factor=16):
    
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
    print(f'Warped Image shape: {warped_image.shape}')
    
    # cv2.imshow('frame', image) # Initial Capture
    # cv2.imshow('frame1', warped_image) # Transformed Capture
    # cv2.waitKey(0)
    # # closing all open windows 
    # cv2.destroyAllWindows()
    
    # Get the original patch, warped patch, and homography and save them to the output directory
    warped_patch = get_patch(warped_image, x, y, patch_size)
    print(f'Warped patch shape: {warped_patch.shape}')
    
    
    # cv2.imshow('frame', original_patch) # Initial Capture
    # cv2.imshow('frame1', warped_patch) # Transformed Capture
    # cv2.waitKey(0)
    # # closing all open windows 
    # cv2.destroyAllWindows()
    
  
    
    return original_patch, warped_patch, perturbation, homography_ab
        
        
    

def main():
    # Add any Command Line arguments here
    # Parser = argparse.ArgumentParser()
    # Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')

    # Args = Parser.parse_args()
    # NumFeatures = Args.NumFeatures

    """
    Read a set of images for Panorama stitching
    """

    """
	Obtain Homography using Deep Learning Model (Supervised and Unsupervised)
	"""

    """
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""
    traindir = 'Phase2/Data/Train/'
    valdir = 'Phase2/Data/Val/'
    
    OrigOutputDir = 'Phase2/Data/GeneratedDataset/Train/Original/'
    WarpOutputDir = 'Phase2/Data/GeneratedDataset/Train/Warped/'
    LabelsOutputDir = 'Phase2/Data/GeneratedDataset/Train/'
    
    if not os.path.exists(OrigOutputDir):    
        os.makedirs(OrigOutputDir)
    if not os.path.exists(WarpOutputDir):   
        os.makedirs(WarpOutputDir)
    
    perturbation_list = []
    homography_list = []
    train_samples = 5000
    num_patches = 10
    count = 0
    
    for i in range(train_samples):
        image_path = f'{traindir}{i+1}.jpg'
        print(f'Processing image: {image_path}')
        
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        h, w = img.shape[0], img.shape[1]
        image = cv2.resize(img, (480, 480))
        
        for j in range(num_patches):

            # Generate 10 patches from each image
            original_patch, warped_patch, perturbation, homography_ab = generate_data(image, i+1)
            
            cv2.imwrite(f'{OrigOutputDir}{count}_{i}.jpg', original_patch)
            cv2.imwrite(f'{WarpOutputDir}{count}_{i}.jpg', warped_patch)
            perturbation_list.append(perturbation)
            homography_list.append(homography_ab)
            
            count += 1
    
    np.save(f'{LabelsOutputDir}Labels.npy', perturbation_list)
    np.save(f'{LabelsOutputDir}Homography.npy', homography_list)
    
    


if __name__ == "__main__":
    main()
