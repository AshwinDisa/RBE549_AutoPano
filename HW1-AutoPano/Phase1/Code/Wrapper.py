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
import matplotlib.pyplot as plt

def save_image(filename, image):
    cv2.imwrite(filename, image)

def get_feature_vector(image, best_corners, iter):
    patch_size = 41
    half_patch_size = patch_size // 2
    padded_image = np.pad(image, ((patch_size, patch_size), (patch_size, patch_size), (0, 0)), 'constant')
    feature_vectors = []
    
    for i in range(len(best_corners)):

        x, y = int(best_corners[i,0] - half_patch_size), int(best_corners[i,1] - half_patch_size)

        if x < 0 or y < 0:
            continue
            
        gray_image = cv2.cvtColor(padded_image, cv2.COLOR_BGR2GRAY)    
        patch = gray_image[x:x + patch_size, y:y + patch_size]
        blurred_patch = cv2.GaussianBlur(patch, (7, 7), 0)
        sub_sampled_patch = cv2.resize(blurred_patch, (8, 8))
        feature_vector = sub_sampled_patch.reshape((64, 1))
        feature_vector = (feature_vector - np.mean(feature_vector)) / np.std(feature_vector) + 0.0001
        feature_vectors.append(feature_vector)

    return feature_vectors

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

    save_image('Phase1/Code/ANMS/anms' + str(iter) + '.jpg', image_anms)
    if cv2.waitKey(0) & 0xff == 27: 
         cv2.destroyAllWindows() 
    # cv2.waitKey(0)

    return n_best

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
        # cv2.circle(image_corner, (x, y), 3, 255, -1)


    # Display the image with corners
    # cv2.imshow('Image with Borders', image_corner)
    # save_image('/home/ashd/WPI Spring 2024/Computer Vision/HW1/YourDirectoryID_p1/HW1-AutoPano/Phase1/Code/corners/corner' + str(iter) + '.jpg', image_corner)        

    # De-allocate any associated memory usage  
    # if cv2.waitKey(0) & 0xff == 27: 
    #     cv2.destroyAllWindows() 
    # cv2.waitKey(0)

    return corner_coords, dst


def match_features(feature_vectors1, feature_vectors2, corner1, corner2, ratio = 0.8):
    
    matches = []
    for j in range(len(feature_vectors1)):
        min_dist = [np.inf, np.inf]
        min_index = [0, 0]
        for k in range(len(feature_vectors2)):
            dist = np.linalg.norm(feature_vectors1[j] - feature_vectors2[k])**2
            if dist < min_dist[0]:
                min_dist[1] = min_dist[0]
                min_index[1] = min_index[0]
                min_dist[0] = dist
                min_index[0] = k
            elif dist < min_dist[1]:
                min_dist[1] = dist
                min_index[1] = k
                
        print(f"ratio : {min_dist[0] / min_dist[1]}")
        
        if min_dist[0] / min_dist[1] < ratio:
            matches.append([corner1[j,:],corner2[min_index[0],:]])
    print(len(matches))
    # print(matches)
    return matches

# def convert_to_keypoints(corners):
#     keypoints = [cv2.KeyPoint(x=c[0], y=c[1], _size=2) for c in corners]
#     return keypoints

def convert_to_keypoints(points):
	kp1 = []
	for i in range(len(points)):
		kp1.append(cv2.KeyPoint(int(points[i][0]), int(points[i][1]), 3))
	return kp1

def convert_to_dmatches(matched_features):
	m = []
	for i in range(len(matched_features)):
		m.append(cv2.DMatch(int(matched_features[i][0]), int(matched_features[i][1]), 2))
	return m



def main():
    # Add any Command Line arguments here
    # Parser = argparse.ArgumentParser()
    # Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')

    # Args = Parser.parse_args()
    # NumFeatures = Args.NumFeatures

    """
    Read a set of images for Panorama stitching
    """
    
    imagepath = "Phase1/Data/Train/Set1"
    images = []
    
    # for filename in os.listdir(imagepath):
    for i in range(3):
        img = cv2.imread(os.path.join(imagepath, f"{i+1}.jpg"))
        images.append(img)
        

    """
    Corner Detection
    Save Corner detection output as corners.png
    """
    feature_vectors = []
    best_corners = []

    for iter, image in enumerate(images):
        
        corner_coords, dst  = get_corners(image, iter)

        """
        Perform ANMS: Adaptive Non-Maximal Suppression
        Save ANMS output as anms.png
        """

        corners = get_best_corners(image, dst, corner_coords, iter, best = 200)
        best_corners.append(corners)
        print(np.array(corners).shape)
        """
        Feature Descriptors
        Save Feature Descriptor output as FD.png
        """
        feature_vector = get_feature_vector(image, corners, iter)
        feature_vectors.append(feature_vector)
        # print(np.array(feature_vector).shape)
        

    """
    Feature Matching
    Save Feature Matching output as matching.png
    """
    
    matches = []
    print("")
    for i in range(len(images)):
        for j in range(i+1, len(images)):
            matched_features = match_features(feature_vectors[i], feature_vectors[j], best_corners[i], best_corners[j], ratio = 0.8)
            # print(matched_features)
            if matched_features:
                keypoints1 = convert_to_keypoints([elem[0] for elem in matched_features])
                keypoints2 = convert_to_keypoints([elem[1] for elem in matched_features])
                matched_pairs_idx = [(i,i) for i,j in enumerate(matched_features)]
                dmatches = convert_to_dmatches(matched_pairs_idx)
                # dmatches = convert_to_dmatches(matched_features)

                matched_image = cv2.drawMatches(images[i], keypoints1, images[j], keypoints2, dmatches, None, flags=2)
                
                plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
                plt.show()
            
            
            

    """
    Refine: RANSAC, Estimate Homography
    """

    """
    Image Warping + Blending
    Save Panorama output as mypano.png
    """


if __name__ == "__main__":
    main()
