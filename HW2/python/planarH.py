from matplotlib.pyplot import hist2d
import numpy as np
import cv2


def computeH(x1, x2):
	#Q2.2.1
	#Compute the homography between two sets of points
	n = len(x1)
	assert len(x1) == len(x2)
	A = [ [ [-x2[i,0], -x2[i,1], -1, 0, 0, 0, x1[i,0]*x2[i,0], x1[i,0]*x2[i,1], x1[i,0]],
			[0, 0, 0, -x2[i,0], -x2[i,1], -1, x1[i,1]*x2[i,0], x1[i,1]*x2[i,1], x1[i,1]] ]
			for i in range(n)
		 ]
	A = np.array(A).reshape(2*n, -1)
	w, v = np.linalg.eig(A.T @ A)
	w = np.abs(w)  # distance is absolute value
	H2to1 = v[:, np.argmin(w)].reshape(3, 3)

	return H2to1


def computeH_norm(x1, x2):
	#Q2.2.2
	#Compute the centroid of the points
	n = len(x1)
	assert len(x1) == len(x2)
	x1x, x1y = np.sum(x2[:, 0]) / n, np.sum(x2[:, 1]) / n
	x2x, x2y = np.sum(x2[:, 0]) / n, np.sum(x2[:, 1]) / n

	#Shift the centroid of the points to the origin
	x1_normalized = x1 - [x1x, x1y]
	x2_normalized = x2 - [x2x, x2y]

	#Normalize the points so that the largest distance from the origin is equal to sqrt(2)
	s1 = np.max([x[0]**2 + x[1]**2 for x in x1_normalized])
	s1 = np.sqrt(2) / s1
	x1_normalized = x1_normalized * s1
	s2 = np.max([x[0]**2 + x[1]**2 for x in x2_normalized])
	s2 = np.sqrt(2) / s2
	x2_normalized = x2_normalized * s2

	#Similarity transform 1
	T1 = np.array([[s1, 0, -x1x * s1], [0, s1, -x1y * s1], [0, 0, 1]])

	#Similarity transform 2
	T2 = np.array([[s2, 0, -x2x * s2], [0, s2, -x2y * s2], [0, 0, 1]])

	#Compute homography
	H2to1 = computeH(x1_normalized, x2_normalized)

	#Denormalization
	H2to1 = np.linalg.inv(T1) @ H2to1 @ T2

	return H2to1


def computeH_ransac(locs1, locs2, opts):
	#Q2.2.3 Compute the best fitting homography given a list of matching points
	max_iters = opts.max_iters  # the number of iterations to run RANSAC for
	inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier
	n = len(locs1)
	assert len(locs1) == len(locs2)

	# Swap columns in locs because they are in the form of [y, x] returned by matchPics
	x1 = np.fliplr(locs1)
	x2 = np.fliplr(locs2)
	x2_homo = np.hstack((x2, np.ones((n,1))))  # Make x2 homogeneous

	iter = 0
	bestH2to1 = None
	inliers = None
	best_inliers_count = -1
	while iter < max_iters:
		sample_indices = np.random.choice(n, 4, replace=False)
		x1_samples = x1[sample_indices]
		x2_samples = x2[sample_indices]
		H2to1 = computeH_norm(x1_samples, x2_samples)
		_preds = H2to1 @ (x2_homo.T)
		preds = np.array([[x[0] / x[2], x[1] / x[2]] if x[2] != 0 
							else [x[0] / 1e-10, x[1] / 1e-10] for x in _preds.T])  # in case blowing up
		inliers_binary = np.linalg.norm(x1 - preds, axis=1) <= inlier_tol
		inliers_count = np.sum(inliers_binary)
		if inliers_count > best_inliers_count:
			bestH2to1 = H2to1
			inliers = inliers_binary.astype(int)
			best_inliers_count = inliers_count
		if inliers_count == n:
			break
		
		iter += 1

	return bestH2to1, inliers



def compositeH(H2to1, template, img):
	
	#Create a composite image after warping the template image on top
	#of the image using the homography

	#Note that the homography we compute is from the image to the template;
	#x_template = H2to1*x_photo
	#For warping the template to the image, we need to invert it.
	H2to1_inv = np.linalg.inv(H2to1)

	#Create mask of same size as template
	mask = np.ones(template.shape)

	#Warp mask by appropriate homography
	warped_m = cv2.warpPerspective(mask, H2to1_inv, (img.shape[1],img.shape[0]))

	#Warp template by appropriate homography
	warped_t = cv2.warpPerspective(template, H2to1_inv, (img.shape[1],img.shape[0]))

	#Use mask to combine the warped template and the image
	composite_img = warped_t + img * np.logical_not(warped_m)
	
	return composite_img