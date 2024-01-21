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

import copy
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import random

def save_image(filename, image):
    cv2.imwrite(filename, image)

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

    result[t[1]:h1 + t[1], t[0]:w1 + t[0]] = image2

    return result

def plotter(image1, keypoints1, image2, keypoints2, dmatches):

    matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, dmatches, None, flags=2)

    plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
    plt.show()


def get_ransac(dmatches, keypoints1, keypoints2):

    count = []
    inliers = []

    for _ in range(1000): 

        random_pts = random.sample(range(len(dmatches)), 4)

        keypoints1_for_ransac = np.array([keypoints1[keypoint.queryIdx].pt for keypoint in dmatches])
        keypoints2_for_ransac = np.array([keypoints2[keypoint.trainIdx].pt for keypoint in dmatches])

        points1_for_ransac = keypoints1_for_ransac[random_pts]
        points2_for_ransac = keypoints2_for_ransac[random_pts]

        homography, mask = cv2.findHomography(points1_for_ransac, points2_for_ransac, cv2.RANSAC, 5.0)

        points = []
        final_keypoint1 = []
        final_keypoint2 = []
        count_inliers = 0

        for i in range(len(keypoints1_for_ransac)):
            keypoint1_array = np.array([keypoints1_for_ransac[i][0], keypoints1_for_ransac[i][1], 1])
            keypoint2_array = np.array([keypoints2_for_ransac[i][0], keypoints2_for_ransac[i][1], 1])

            keypoint1_array_for_homo = [keypoints1_for_ransac[i][0], keypoints1_for_ransac[i][1]]
            keypoint2_array_for_homo = [keypoints2_for_ransac[i][0], keypoints2_for_ransac[i][1]]

            ssd = np.linalg.norm(np.array(keypoint2_array.T) - np.dot(homography, keypoint1_array.T))

            if ssd < 30:
                final_keypoint1.append(keypoint1_array_for_homo)
                final_keypoint2.append(keypoint2_array_for_homo)
                count_inliers += 1

        count.append(count_inliers)
        inliers.append((homography, (final_keypoint1, final_keypoint2)))

    max_count_idx = np.argmax(count)
    final_matched_pairs = inliers[max_count_idx][1]

    pts_1 = final_matched_pairs[0]
    pts_2 = final_matched_pairs[1]
    h_final_matrix, status = cv2.findHomography(np.float32(pts_1), np.float32(pts_2))

    # print(h_final_matrix)

    return h_final_matrix, final_matched_pairs

def get_feature_vector(image, best_corners, iter):
    patch_size = 41
    half_patch_size = patch_size // 2
    padded_image = cv2.copyMakeBorder(image, half_patch_size, half_patch_size, half_patch_size, half_patch_size, cv2.BORDER_CONSTANT)
    feature_vectors = []
    
    for x, y in best_corners:
        
        x_padded = int(x + half_patch_size)
        y_padded = int(y + half_patch_size)

        patch = padded_image[y_padded - patch_size//2:y_padded + patch_size//2 + 1,
                      x_padded - patch_size//2:x_padded + patch_size//2 + 1]
        
        blurred_patch = cv2.GaussianBlur(patch, (7, 7), 0)
        sub_sampled_patch = cv2.resize(blurred_patch, (8, 8))
        feature_vector = sub_sampled_patch.reshape((64, 1))
        # print(patch.shape)
        
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
                
        # print(f"ratio : {min_dist[0] / min_dist[1]}")
        if min_dist[0] / min_dist[1] < ratio:
            matches.append([corner1[j,:],corner2[min_index[0],:]])
    # print(f"Number of matches: {len(matches)}")
    return matches

def convert_to_keypoints(corners):
    keypoints = [cv2.KeyPoint(c[0], c[1], 3) for c in corners]
    return keypoints

def convert_to_dmatches(matched_features):
    matches = [cv2.DMatch(i, i, 2) for i,j in enumerate(matched_features)]
    return matches

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
    for i in range(len(os.listdir(imagepath))):
        img = cv2.imread(os.path.join(imagepath, f"{i+1}.jpg"))
        images.append(img)

    """
    Corner Detection
    Save Corner detection output as corners.png
    """
    
    feature_vectors = []
    best_corners = []

    for iter, image in enumerate(images):
        
        # corner_coords, dst  = get_corners(image, iter)

        """
        Perform ANMS: Adaptive Non-Maximal Suppression
        Save ANMS output as anms.png
        """

        # corners = get_best_corners(image, dst, corner_coords, iter, best = 200)
        # print(corners)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
       
        corners = cv2.goodFeaturesToTrack(gray, 500, 0.001, 10, useHarrisDetector=False)
        
        corners = np.array(corners).reshape(-1, 2)
        best_corners.append(corners)
        # print(corners.shape)
        
        # print(corners)
        
        """
        Feature Descriptors
        Save Feature Descriptor output as FD.png
        """
        feature_vector = get_feature_vector(gray, corners, iter)
        feature_vectors.append(feature_vector)
        # print(np.array(feature_vector).shape)

    
    print("")
    for i in range(len(images)-1):

        """
        Feature Matching
        Save Feature Matching output as matching.png
        """
        matches = []
        
        matched_features = match_features(feature_vectors[i], feature_vectors[i+1], best_corners[i], best_corners[i+1], ratio = 0.8)
        
        if matched_features:

            keypoints1 = convert_to_keypoints([elem[0] for elem in matched_features])
            keypoints2 = convert_to_keypoints([elem[1] for elem in matched_features])
            dmatches = convert_to_dmatches(matched_features)

            # plotter(images[i], keypoints1, images[j], keypoints2, dmatches)                

            
            """
            Refine: RANSAC, Estimate Homography
            """

            h_final_matrix, final_matched_pairs = get_ransac(dmatches, keypoints1, keypoints2)

            keypoints1 = convert_to_keypoints(final_matched_pairs[0])
            keypoints2 = convert_to_keypoints(final_matched_pairs[1])
            for k in range(len(keypoints1)):
                matches.append([keypoints1[k], keypoints2[k]])

            dmatches = convert_to_dmatches(matches)

            # plotter(images[i], keypoints1, images[j], keypoints2, dmatches)

            """
            Image Warping + Blending
            Save Panorama output as mypano.png
            """

            warped_image = get_warped_image(images[i], images[i+1], h_final_matrix)

            plt.imshow(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
            plt.show()

            save_image("Phase1/Data/Results/Set1/mypano" + str(i) + ".png", warped_image)


if __name__ == "__main__":
    main()
