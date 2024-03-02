# AutoPano
Part of coursework for Computer Vision at Worcester Polytechnic Institute (WPI) by Prof. Nitin J. Sanket. Course website: https://pear.wpi.edu/teaching/rbe549/spring2024.html

The purpose of this project is to stitch two or more images in order to create one seamless panorama image. Each image have few repeated local features (∼30-50% or more).

## Phase1: Traditional Approach

### Input Images

<p float="left">
    <img src="https://github.com/AshwinDisa/RBE549_AutoPano/blob/master/HW1-AutoPano/Phase1/Data/Train/Set1/1.jpg?raw=true" width="300" height="300">
    <img src="https://github.com/AshwinDisa/RBE549_AutoPano/blob/master/HW1-AutoPano/Phase1/Data/Train/Set1/2.jpg?raw=true" width="300" height="300">
    <img src="https://github.com/AshwinDisa/RBE549_AutoPano/blob/master/HW1-AutoPano/Phase1/Data/Train/Set1/3.jpg?raw=true" width="300" height="300">
</p>

### Feature extraction and ANMS

<p float="left">
    <img src="https://github.com/AshwinDisa/RBE549_AutoPano/blob/master/HW1-AutoPano/Phase1/Code/ANMS/anms0.jpg?raw=true" width="300" height="300">
    <img src="https://github.com/AshwinDisa/RBE549_AutoPano/blob/master/HW1-AutoPano/Phase1/Code/ANMS/anms1.jpg?raw=true" width="300" height="300">
    <img src="https://github.com/AshwinDisa/RBE549_AutoPano/blob/master/HW1-AutoPano/Phase1/Code/ANMS/anms2.jpg?raw=true" width="300" height="300">
</p>

### Feature matching

<p float="left">
    <img src="https://github.com/AshwinDisa/RBE549_AutoPano/blob/master/HW1-AutoPano/Phase1/Code/matching/matching.png?raw=true" width="400" height="200">
    <img src="https://github.com/AshwinDisa/RBE549_AutoPano/blob/master/HW1-AutoPano/Phase1/Code/matching/matching2.png?raw=true" width="400" height="250">
</p>


### Feature matching after RANSAC

<p float="left">
    <img src="https://github.com/AshwinDisa/RBE549_AutoPano/blob/master/HW1-AutoPano/Phase1/Code/matching/ransac.png?raw=true" width="400" height="200">
    <img src="https://github.com/AshwinDisa/RBE549_AutoPano/blob/master/HW1-AutoPano/Phase1/Code/matching/ransac2.png?raw=true" width="400" height="250">
</p>

### Stitched Panaroma

<img src="https://github.com/AshwinDisa/RBE549_AutoPano/blob/master/HW1-AutoPano/Phase1/Code/matching/pano1.png?raw=true" width="300" height="300">

## Phase2: Deep Learning Approach

## Supervised Approach to estimate the homography

### Supervised model

<img src="https://github.com/AshwinDisa/RBE549_AutoPano/blob/master/HW1-AutoPano/Phase1/Code/matching/sup_model.png?raw=true" width="300" height="200">

### Results (Green: Ground truth, Blue: predicted)

<p float="left">
    <img src="https://github.com/AshwinDisa/RBE549_AutoPano/blob/master/HW1-AutoPano/Phase1/Code/matching/sup_result.png?raw=true" width="450" height="250">
    <img src="https://github.com/AshwinDisa/RBE549_AutoPano/blob/master/HW1-AutoPano/Phase1/Code/matching/sup_result2.png?raw=true" width="450" height="250">
</p>


## Unsupervised Approach to estimate the homography

### Unsupervised model

<img src="https://github.com/AshwinDisa/RBE549_AutoPano/blob/master/HW1-AutoPano/Phase1/Code/matching/unsup_model.png?raw=true" width="500" height="200">

### Results (Green: Ground truth, Blue: predicted)

<p float="left">
    <img src="https://github.com/AshwinDisa/RBE549_AutoPano/blob/master/HW1-AutoPano/Phase1/Code/matching/unsup_result.png?raw=true" width="450" height="250">
    <img src="https://github.com/AshwinDisa/RBE549_AutoPano/blob/master/HW1-AutoPano/Phase1/Code/matching/unsup_result2.png?raw=true" width="450" height="250">
</p>

### Detailed explanations can be found in the project report. 
