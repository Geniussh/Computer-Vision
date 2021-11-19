## To use
* Run ```get_data.sh``` to download both NIST36 and images used for text extraction
* Run ```fetch_flowers17.sh``` to download [flowers 17](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/index.html) or ```fetch_flowers102.sh``` to download [flowers 102](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)
* ```run_q2.py``` validates the manual implementation of a fully-connected network
* ```run_q3.py``` trains the FC network from scratch, visualizes the weights learned, and computes the confusion matrix on test set
* ```run_q4.py``` extracts text from four images of handwritten characters **by row**, and classifies them with the FC network
* ```run_q5.py``` designs an autoencoder, trained with SGD with momentum, and reconstructs images in NIST36, which is then evaluated using Peak Signal-to-Noise Ratio
* ```run_q6_1.py``` trains and evaluates the following networks in PyTorch
    * A 2-layer FC on NIST36
    * A CNN (3 Conv layers followed by a FC) on NIST36
    * The same CNN on CIFAR-10
    * A CNN (4 Conv layers followed by a FC) on SUN
* ```run_q6_2.py``` fine tunes the last classifier layer of [SqueezeNet](https://pytorch.org/vision/stable/models.html) on [Flowers17](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/index.html) and compares its performance with a [LeNet](https://en.wikipedia.org/wiki/LeNet) trained from scratch, in PyTorch
  
## Results
* Confusion Matrix on NIST36 using the 2-layer FC Network  
  
<p float="left">
  <img src="https://github.com/Geniussh/Computer-Vision/blob/main/HW5/results/q3_4.png" height="350px">
</p>

* Text Extraction By Row (Original -- Detected -- Extracted)  
  
<p float="left">
  <img src="https://github.com/Geniussh/Computer-Vision/blob/main/HW5/images/04_deep.jpg" height="200px">
  <img src="https://github.com/Geniussh/Computer-Vision/blob/main/HW5/results/q4_3_4.png" height="200px">
  <img src="https://github.com/Geniussh/Computer-Vision/blob/main/HW5/results/input4.png" height="200px">
</p>
  
* Autoencoder on NIST36
  
<p float="left">
  <img src="https://github.com/Geniussh/Computer-Vision/blob/main/HW5/results/q5_3_A.png" height="200px">
  <img src="https://github.com/Geniussh/Computer-Vision/blob/main/HW5/results/q5_3_H.png" height="200px">
  <img src="https://github.com/Geniussh/Computer-Vision/blob/main/HW5/results/q5_3_9.png" height="200px">
</p>
  
* Fine Tuning of SqueezeNet vs. Training LeNet from Scratch on Flowers 17
<p float="left">
  <img src="https://github.com/Geniussh/Computer-Vision/blob/main/HW5/results/q6_2_ft.png" height="300px" width="400px">
  <img src="https://github.com/Geniussh/Computer-Vision/blob/main/HW5/results/q6_2_ln2.png" height="300px" width="400px">
</p>
