Package for comaring images by content.

image fingerprints: simple and fast
-----------------------------------
These methods basically squash down the image to something like 16x16,
transform to gray scale and store that as a feature vector of length 16x16, for
example -> fast. But the method s not invariant against rotation, only scaling
along x and/or y. 

The idea is always to calculate a database of image fingerprints ("hashes",
feature vectors) and then do searches in feature space (all fingerprints) using
some form of KD-tree / nearest neighbor search.

* google: calculate image fingerprint
* [a|p|d]hash: https://realpython.com/blog/python/fingerprinting-images-for-near-duplicate-detection/ 
* espcially: phash.org
* older Perl implemention of a ahash(?)-like method: http://www.jhnc.org/findimagedupes/manpage.html, also as Debian
  package

more scientific "feature extraction"
------------------------------------

* classical CV (computer vision): SIFT (good but slow, old-school
  hand-engineered feature detector), SURF (faster version of
  SIFT)
* http://opencv-python-tutroals.readthedocs.org/en/latest/index.html
    * SIFT and SURF are patented, so fuck them and use ORB
      http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_feature2d/py_orb/py_orb.html#orb
* opencv Bag Of Words: http://stackoverflow.com/questions/7205489/opencv-fingerprint-image-and-compare-against-database

Python image processing
-----------------------
* google: python image processing :)
* http://scikit-image.org/
* PIL vs. Pillow: http://docs.python-guide.org/en/latest/scenarios/imaging/
* http://www.scipy-lectures.org/advanced/image_processing

better methods
--------------
read about: Content-based image classification
