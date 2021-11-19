# Course Projects in Computer Vision at CMU
> ***Any Copying from the work of another person is a violation of [Carnegie Mellon University Policy on Academic Integrity](https://www.cmu.edu/policies/student-and-student-life/academic-integrity.html).***
  
### [0: Color Channel Alignment and Image Warping ](HW0/)
<img src="https://github.com/Geniussh/Computer-Vision/blob/main/HW0/results/transformed.jpg" width="400px">
  
  
### [1: Scene Classification System using Bag-of-words Approach with Spatial Pyramid Extension](HW1/)
- Feature Extraction based on Filter Banks
- K Means Clustering
- Visual Word Dictionary
- Scene Classification
- Hyperparameters Tuning
- CNN Implementation  
<img src="https://user-images.githubusercontent.com/44150278/136886336-e9075072-380a-4dbb-9338-4f023448f83b.png" width="500px">
  
  
### [2: Augmented Reality with Planar Homographies](HW2/)
- Direct Linear Transform
- Matrix Decomposition to calculate Homography
- Limitations of Planar Homography
- FAST Detector and BRIEF Descriptors
- Feature Matching
- Compute Homography via RANSAC
- Automated Homography Estimation and Warping
- Augmented Reality Application using Homography
- Real-Time Augmented Reality with High FPS
- Panorama Generation based on Homography  
<img src="https://github.com/Geniussh/Computer-Vision/blob/main/HW2/result/ar.gif" width="400px">
  
  
### [3: Lucas-Kanade Object Tracking](HW3/)
- Simple Lucas & Kanade Tracker with Naive Template Update
- Lucas & Kanade Tracker with Template Correction
- Two-dimensional Tracking with a Pure Translation Warp Function
- Two-dimensional Tracking with a Plane Affine Warp Function
- Lucas & Kanade Forward Additive Approach
- Lucas & Kanade Inverse Compositional Approach
<p float="left">
  <img src="https://github.com/Geniussh/Computer-Vision/blob/main/HW3/result/car.gif" width="250px">
  <img src="https://github.com/Geniussh/Computer-Vision/blob/main/HW3/result/aerial.gif" width="250px">
  <img src="https://github.com/Geniussh/Computer-Vision/blob/main/HW3/result/ant.gif" width="250px">
</p>
  
  
### [4: Structure from Motion - 3D Reconstruction](HW4/)
- Fundamental Matrix Estimation using Point Correspondence
- Metric Reconstruction
- Retrieval of Camera Matrices up to a Scale and Four-Fold Rotation Ambiguity
- Triangulation using the Homogeneous Least Squares Solution
- 3D Visualization from a Stereo-Pair by Triangulation and 3D Locations Rendering
- Bundle Adjustment
    - Estimated fundamental matrix through RANSAC for noisy correspondences
    - Jointly optmized reprojection error w.r.t 3D estimated points and camera matrices
    - Non-linear optimization using SciPy least square optimizer
<p float="left">
  <img src="https://github.com/Geniussh/Computer-Vision/blob/main/HW4/data/im1.png" width="250px">
  <img src="https://github.com/Geniussh/Computer-Vision/blob/main/HW4/data/im1.png" width="250px">
</p>
<p float="left">
  <img src="https://github.com/Geniussh/Computer-Vision/blob/main/HW4/result/rotation2.gif" width="250px">
  <img src="https://github.com/Geniussh/Computer-Vision/blob/main/HW4/result/rotation1.gif" width="250px">
</p>
  
  
### [5: Neural Networks for Recognition](HW5/)
- Manual Implementation of a Fully Connected Network
- Text Extraction from Images of Handwritten Characters
- Image Compression with Autoencoders
- PyTorch Implementation of a Convolutional Neural Network
- Fine Tuning of SqueezeNet in PyTorch
- Comparison between Fine Tuning and Training from Scratch
