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

def save_image(filename, image):
    cv2.imwrite(filename, image)

def get_best_corners(image_anms, corners, corner_coords, iter, best = 200):

    Nstrong = len(corner_coords)
    inf = np.inf
    r = inf * np.ones((Nstrong,3)) 
    ED = 0

    for i in range(Nstrong):

        for j in range(Nstrong):

            x = corner_coords[i][0]
            y = corner_coords[i][1]
            other_x = corner_coords[j][0]
            other_y = corner_coords[j][1]

            if corners[y, x] > corners[other_y, other_x]:
                
                ED = (other_x - x)**2 + (other_y - y)**2

            if ED < r[i, 0]:
                r[i, 0] = ED
                r[i, 1] = x
                r[i, 2] = y
            
    r = r[r[:, 0].argsort()]
    r = np.flip(r)
    n_best = r[:best]

    for i in range(len(n_best)):
        cv2.circle(image_anms, (int(n_best[i][1]),int(n_best[i][0])), 3, [0, 255, 0], -1)
    cv2.imshow('Image with N best corners', image_anms)

    save_image('/home/ashd/WPI Spring 2024/Computer Vision/HW1/YourDirectoryID_p1/Phase1/Code/ANMS/anms' + str(iter) + '.jpg', image_anms)
    if cv2.waitKey(0) & 0xff == 27: 
         cv2.destroyAllWindows() 
    cv2.waitKey(0)

    return None

def get_corners(image_corner, iter, threshold = 0.01):

    # Convert image to grayscale
    gray = cv2.cvtColor(image_corner, cv2.COLOR_BGR2GRAY)

    # Detect corners using goodFeaturesToTrack

    # corners = cv2.goodFeaturesToTrack(gray, 500, 0.01, 10)
    # corners = np.intp(corners)
    # print(corners)
    # corner_coords = []

    # for corner in corners:

    #     x, y = corner.ravel()
    #     corner_coords.append([x, y])
    #     cv2.circle(image, (x, y), 2, 255, -1)


    # Apply cornerHarris algorithm
        
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    ret, dst_int = cv2.threshold(dst,threshold*dst.max(),255,0)
    dst_int = np.uint8(dst_int)
    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst_int)

    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
    corner_coords = []

    for corner in corners:

        x, y = corner[0], corner[1]
        x, y = np.intp(x), np.intp(y)
        corner_coords.append([x, y])
        cv2.circle(image_corner, (x, y), 3, 255, -1)


    # Display the image with corners
    cv2.imshow('Image with Borders', image_corner)
    save_image('/home/ashd/WPI Spring 2024/Computer Vision/HW1/YourDirectoryID_p1/Phase1/Code/corners/corner' + str(iter) + '.jpg', image_corner)        

    # De-allocate any associated memory usage  
    if cv2.waitKey(0) & 0xff == 27: 
        cv2.destroyAllWindows() 
    cv2.waitKey(0)

    return corner_coords, dst

def main():
    # Add any Command Line arguments here
    # Parser = argparse.ArgumentParser()
    # Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')

    # Args = Parser.parse_args()
    # NumFeatures = Args.NumFeatures

    """
    Read a set of images for Panorama stitching
    """
    
    imagepath = "/home/ashd/WPI Spring 2024/Computer Vision/HW1/YourDirectoryID_p1/Phase1/Data/Train/Set1"
    images = []
    
    for filename in os.listdir(imagepath):
        img = cv2.imread(os.path.join(imagepath, filename))
        images.append(img)

    """
    Corner Detection
    Save Corner detection output as corners.png
    """

    for iter, image in enumerate(images):
        
        corner_coords, dst  = get_corners(image, iter)

        best_corners = get_best_corners(image, dst, corner_coords, iter, best = 200) 


    """
    Perform ANMS: Adaptive Non-Maximal Suppression
    Save ANMS output as anms.png
    """

           
        
        

    """
    Feature Descriptors
    Save Feature Descriptor output as FD.png
    """

    """
    Feature Matching
    Save Feature Matching output as matching.png
    """

    """
    Refine: RANSAC, Estimate Homography
    """

    """
    Image Warping + Blending
    Save Panorama output as mypano.png
    """


if __name__ == "__main__":
    main()
