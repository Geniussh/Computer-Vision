## To use  
* Navigate to ```data``` and unzip all the data files. (I used up my LFS bandwidth so ...)  
	* ```carseq.npy``` and ```girlseq.npy``` are datasets with purely translational objects (hence using a pure translation warp function in my Lucas Kanade tracker)
	* ```antseq.npy``` and ```aerialseq.npy``` are datasets with aerial views of moving objects from a non-stationary camera (hence using an affine warp function in my Lucas Kanade tracker)
* Run ```testCarSequence.py``` or ```testGirlSequence.py``` to see how **Lucas Kanade (vanilla forward additive approach)** works on ```carseq.npy``` and ```girlseq.npy``` with **naive template update**   
* Run ```testCarSequenceWithTemplateCorrection.py``` or ```testGirlSequenceWithTemplateCorrection.py``` to see how **Lucas Kanade (vanilla forward additive approach)** works on ```carseq.npy``` and ```girlseq.npy``` with **template correction**   
	* Different template update approaches are discussed at section 2.1 in [Iain Matthews et al.](https://www.ri.cmu.edu/publication_view.html?pub_id=4433)  
  
* Run ```testAntSequence.py``` or ```testAerialSequence.py``` to see how **Lucas Kanade tracker** works on ```antseq.npy``` and ```aerialseq.npy```
	* Modify ```SubtractDominantMotion.py``` to choose **[Lucas Kanade Forward Additive approach](https://www.ri.cmu.edu/pub_files/pub3/baker_simon_2002_3/baker_simon_2002_3.pdf)** or **[Lucas Kanade Inverse Compositional approach](https://www.ri.cmu.edu/pub_files/pub3/baker_simon_2003_3/baker_simon_2003_3.pdf)** (a faster variant of LK) as the tracker
	* Modify ```SubtractDominantMotion.py``` to choose different combinations of erosion and dilation to visualize better results

## Results
* Lucas Kanade w/ Forward Additive approach on ```carseq.npy```  
	* Baseline with naive template update in blue. Template correction in red.
  
![img](https://github.com/Geniussh/ComputerVision/blob/main/HW3/result/carseq_wcrt.png)

* Lucas Kanade w/ Forward Additive approach on ```girlseq.npy```  
	* Baseline with naive template update in blue. Template correction in red.
  
![img](https://github.com/Geniussh/ComputerVision/blob/main/HW3/result/girlseq_wcrt.png)

* Lucas Kanade w/ Forward Additive approach or Inverse Compositional approach on ```antseq.npy```  
  
![img](https://github.com/Geniussh/ComputerVision/blob/main/HW3/result/ant.png)
  
* Lucas Kanade w/ Forward Additive approach or Inverse Compositional approach on ```aerialseq.npy```  
  
![img](https://github.com/Geniussh/ComputerVision/blob/main/HW3/result/aerial.png)
