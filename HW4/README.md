## To use
* Run ```Q2 - Q4 Testing``` in ```main.py``` to visualize the fundamental matrix estimation (```displayEpipolarF```) and matching point searching (```epipolarMatchGUI```) through an interactive GUI. 
* Run ```visualize.py``` to visualize the 3D point cloud reconstruction from a pair of images taken at different angles. 
* Run ```Ransac F``` in ```main.py``` to estimate fundamental matrix using RANSAC on a list of correspondences with noise.
* Run ```Bundle Adjustment``` in ```main.py``` to jointly optimize the best camera matrix and 3D points.
    * Using the inlier correspondences and the RANSAC estimate of the extrinsics & 3D points as an intialization
  
## Results
* Stereo Pair
<p float="left">
  <img src="https://github.com/Geniussh/Computer-Vision/blob/main/HW4/data/im1.png" width="470px">
  <img src="https://github.com/Geniussh/Computer-Vision/blob/main/HW4/data/im1.png" width="470px">
</p>
  
* Epipolar Lines Visualization  
  
![image](https://user-images.githubusercontent.com/44150278/138610334-546ace26-0405-4a21-8a3f-0ee99ec970b8.png)
  
* Matching Point Searching  
  
![image](https://user-images.githubusercontent.com/44150278/138610345-804a1b25-36a8-4df0-bedf-84602205c6f4.png)
  
* 3D Reconstruction
<p float="left">
  <img src="https://github.com/Geniussh/Computer-Vision/blob/main/HW4/result/rotation2.gif" width="470px">
  <img src="https://github.com/Geniussh/Computer-Vision/blob/main/HW4/result/rotation1.gif" width="470px">
 </p>
  
* Bundle Adjustment w/ Nonlinear Optimization  
  
![image](https://user-images.githubusercontent.com/44150278/138610364-5800365e-393c-463d-954d-071d7e59e76c.png)
