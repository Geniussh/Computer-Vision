> ***Any Copying from the work of another person is a violation of [Carnegie Mellon University Policy on Academic Integrity](https://www.cmu.edu/policies/student-and-student-life/academic-integrity.html).***
  
## To Use
* Run ```q1.py``` to see the results of calibrated photometric stereo
    * Simulation of a Lambertian sphere in an orthographic camera with changing light directions
    * Estimation of albedos and normals of a human face given 7 images of the face lit from different directions
    * Estimation of the shape of the face by applying the Frankot-Chellappa algorithm to the normals
* Run ```q2.py``` to see the results of uncalibrated photometric stereo
    * Factorization of the pseudo-normals and the lighting directions with a rank-3 approximation
    * Investigation and solution of the ambiguity by enforcing integrability of the pseudo-normals
    * Investigation of the *generalized bas-relief ambiguity*, ![equation](https://latex.codecogs.com/gif.latex?%5Cbg_white%20%5Clambda%20f%28x%2Cy%29&plus;%5Cmu%20x&plus;%5Cnu%20y)
   

## Results
* Simulation of a Lambertian sphere with changing light directions
<p float="left">
  <img src="https://github.com/Geniussh/Computer-Vision/blob/main/HW6/results/Q1b_1.png" height="200px">
  <img src="https://github.com/Geniussh/Computer-Vision/blob/main/HW6/results/Q1b_2.png" height="200px">
</p>
  
* Data
<p float="left">
  <img src="https://user-images.githubusercontent.com/44150278/143800484-3c056242-dab9-4165-a5c1-2a75299117d7.png" height="100px">
  <img src="https://user-images.githubusercontent.com/44150278/143800508-e7ec2bff-ce4c-4203-8be8-4d239d40dc42.png" height="100px">
  <img src="https://user-images.githubusercontent.com/44150278/143800534-ebea025d-d88a-48d0-b714-1693223991ee.png" height="100px">
  <img src="https://user-images.githubusercontent.com/44150278/143800552-3860425c-72ed-47ae-8b1b-38aa87554f5e.png" height="100px">
  <img src="https://user-images.githubusercontent.com/44150278/143800576-3e57c50a-81e5-4a19-8481-5a23f7a0ca3d.png" height="100px">
  <img src="https://user-images.githubusercontent.com/44150278/143800619-45a986e8-f647-4059-84c8-eb88ade7c64c.png" height="100px">
  <img src="https://user-images.githubusercontent.com/44150278/143800675-71a8bf9f-79db-40f5-bd3c-2eb4b2ae4e43.png" height="100px">
</p>
  
* Calibrated Photometric Stereo
<p float="left">
  <img src="https://github.com/Geniussh/Computer-Vision/blob/main/HW6/results/demo.gif" height="300px">
</p>
  
* Generalized Bas-Relief Ambiguity
<p float="left">
  <img src="https://user-images.githubusercontent.com/44150278/143801597-a169fd25-cf7a-4583-ae83-7621250c7584.png" width="500px">
</p>

