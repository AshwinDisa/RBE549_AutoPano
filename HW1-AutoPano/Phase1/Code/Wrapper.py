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

def crop_panorama(panorama):
    gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    cropped_panorama = panorama[y:y+h, x:x+w]
    return cropped_panorama

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

        if (np.all(points1_for_ransac == points1_for_ransac[0]) or np.all(points2_for_ransac == points2_for_ransac[0])):
            print("points are same")
            continue

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

def get_feature_vector(image, best_corners):
    patch_size = 41
    half_patch_size = patch_size // 2
    padded_image = cv2.copyMakeBorder(image, half_patch_size, half_patch_size, half_patch_size, half_patch_size, cv2.BORDER_CONSTANT)
    feature_vectors = []
   
    for x, y in best_corners:

        if (x == np.inf or y == np.inf):
            continue

        x_padded = int(x + half_patch_size)
        y_padded = int(y + half_patch_size)

        patch = padded_image[y_padded - half_patch_size:y_padded + half_patch_size, 
                             x_padded - half_patch_size:x_padded + half_patch_size+1]

        blurred_patch = cv2.GaussianBlur(patch, (7, 7), 0)
        sub_sampled_patch = cv2.resize(blurred_patch, (8, 8))
        feature_vector = sub_sampled_patch.reshape((64, 1))
        
        feature_vector = (feature_vector - np.mean(feature_vector)) / np.std(feature_vector) + 0.0001
        feature_vectors.append(feature_vector)

    return feature_vectors


def get_best_corners(image_anms, corners, corner_coords, best):

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
            
            if (x >= corners.shape[1] or y >= corners.shape[0] or other_x >= corners.shape[1] or other_y >= corners.shape[0]):
                continue

            if corners[y, x] > corners[other_y, other_x]:
                
                ED = (other_x - x)**2 + (other_y - y)**2

            if ED < r[i, 0]:
                r[i, 0] = ED
                r[i, 1] = y
                r[i, 2] = x

    r = r[r[:, 0].argsort()]
    r = np.flip(r)
    n_best = r[:best]
    n_best = n_best[:, 0:2]

    # for i in range(len(n_best)):
    #     cv2.circle(image_anms, (int(n_best[i][0]),int(n_best[i][1])), 3, [0, 255, 0], -1)
    # cv2.imshow('Image with N best corners', image_anms)

    # # save_image('Phase1/Code/ANMS/anms' + str(iter) + '.jpg', image_anms)
    # if cv2.waitKey(0) & 0xff == 27: 
    #      cv2.destroyAllWindows() 
    # cv2.waitKey(0)

    return n_best

def get_corners(gray, stitched_panorama, threshold=0.01):
    
    # Apply cornerHarris algorithm
        
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    ret, dst_int = cv2.threshold(dst,threshold * np.max(dst),255,0)
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
        # cv2.circle(stitched_panorama, (x, y), 3, [0, 255, 0], -1)


    # cv2.imshow('Image with Borders', stitched_panorama)
    # # save_image('/home/ashd/WPI Spring 2024/Computer Vision/HW1/YourDirectoryID_p1/HW1-AutoPano/Phase1/Code/corners/corner' + str(iter) + '.jpg', image_corner)        
    
    # if cv2.waitKey(0) & 0xff == 27: 
    #     cv2.destroyAllWindows() 
    # cv2.waitKey(0)

    return corner_coords, dst


def match_features(feature_vectors1, feature_vectors2, corner1, corner2, ratio = 0.6):
    
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
    
    # imagepath = "Phase1/Data/Train/Set1"
    imagepath = "P1TestSet/Phase1/TestSet3"
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

    stitched_panorama = images[0]

    print("")
    for i in range(len(images)-1):

        gray_pano = cv2.cvtColor(stitched_panorama, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(images[i+1], cv2.COLOR_BGR2GRAY)
    
        corners = cv2.goodFeaturesToTrack(gray, 300, 0.001, 10, useHarrisDetector=False)
        corners_pano = cv2.goodFeaturesToTrack(gray_pano, 300, 0.001, 10, useHarrisDetector=False)

        # corner_coords, dst = get_corners(gray, images[i+1])
        # corner_coords_pano, dst_pano = get_corners(gray_pano, stitched_panorama)

        # corners = get_best_corners(images[i+1], dst, corner_coords, best=500)
        # corners_pano = get_best_corners(stitched_panorama, dst_pano, corner_coords_pano, best=500)

        corners = np.array(corners).reshape(-1, 2)
        corners_pano = np.array(corners_pano).reshape(-1, 2)
        best_corners.append(corners_pano)
        best_corners.append(corners)
        
        """
        Feature Descriptors
        Save Feature Descriptor output as FD.png
        """
        
        feature_vector = get_feature_vector(gray, corners)
        feature_vector_pano = get_feature_vector(gray_pano, corners_pano)
        feature_vectors.append(feature_vector_pano)
        feature_vectors.append(feature_vector)

        """
        Feature Matching
        Save Feature Matching output as matching.png
        """
        matches = []
        
        matched_features = match_features(feature_vectors[2*i], feature_vectors[2*i+1], best_corners[2*i], best_corners[2*i+1], ratio = 0.8)
        
        if matched_features:

            keypoints1 = convert_to_keypoints([elem[0] for elem in matched_features])
            keypoints2 = convert_to_keypoints([elem[1] for elem in matched_features])
            dmatches = convert_to_dmatches(matched_features)

            # plotter(stitched_panorama, keypoints1, images[i+1], keypoints2, dmatches)                
            
            """
            Refine: RANSAC, Estimate Homography
            """

            h_final_matrix, final_matched_pairs = get_ransac(dmatches, keypoints1, keypoints2)

            keypoints1 = convert_to_keypoints(final_matched_pairs[0])
            keypoints2 = convert_to_keypoints(final_matched_pairs[1])
            for k in range(len(keypoints1)):
                matches.append([keypoints1[k], keypoints2[k]])

            dmatches = convert_to_dmatches(matches)

            # plotter(stitched_panorama, keypoints1, images[i+1], keypoints2, dmatches)

            """
            Image Warping + Blending
            Save Panorama output as mypano.png
            """

            stitched_panorama = get_warped_image(stitched_panorama, images[i+1], h_final_matrix)

            # stitched_panorama = crop_panorama(stitched_panorama)
            
            plt.imshow(cv2.cvtColor(stitched_panorama, cv2.COLOR_BGR2RGB))
            plt.show()

            save_image("Phase1/Data/Results/Set1/" + str(i+1) + ".jpg", stitched_panorama)

        # best_corners = []
        # feature_vectors = []


if __name__ == "__main__":
    main()
