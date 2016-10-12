About
=====
Package for comparing images by content. Uses simple image hashing
("fingerprints"). See "Methods" below for details.

Usage
=====

There are 3 scripts in ``bin/``::

    $ ls bin/
    00resize.py  10fingerprints.py  20cluster.py

These must be executed one after another. Have a look at the help (use
``<script> -h``).

The first one, resizing images, is optional. However, if you have large images
(e.g. from your very expensive new camera), you should really resize them first
in order to lower the computational cost of subsequent operations. This makes
sense of you want (and you will want) to play with various parameters of the
clustering.

::

    $ ./bin/00resize.py 0.2 /path/to/large/pics/* -vv

This will resize all images to 20% (factor 0.2) of their original size. Resized
files are written to ``~/.imgcmp/convert/`` by default. Now, calculate the
fingerprint database::

    $ ./bin/10fingerprints.py ~/.imgcmp/convert/*

This creates a file ``~/.imgcmp/fingerprints.hdf`` (HDF5 file format). Last,
cluster images by using a similarity index (``0.3`` below). A small index means
to put only very similar images in one cluster. The extreme case 0.0 means to
allow zero dissimilarity and thus put each image in a cluster of size 1. In
contrast, large values imply less strict clustering and will put more but less
similar images in a cluster. A value of 1.0 is equal to putting all images in
one single cluster (e.g. all are treated as equal)::

    $ ./bin/20cluster.py 0.3
    items per cluster : number of such clusters
    2 : 41
    3 : 2
    4 : 2

Have a look at the clusters (as dirs with symlinks to the relevant files)::

    $ ls ~/.imgcmp/cluster/
    cluster_with_2  cluster_with_3  cluster_with_4

    $ ls ~/.imgcmp/cluster/cluster_with_2/
    cluster_0   cluster_1  cluster_2

    $ qiv -ifm ~/.imgcmp/cluster/cluster_with_2/cluster_0/*

Methods
=======

What we can do and what not
---------------------------

We use a variant of the phash method -- a simple and fast way to calculate
fingerprints. The package can detect images which are rather similar, e.g. the
same scene photographed twice or more with some camera movement in between, or
a scene with the same background and e.g. one person exchanged. Good parameter
values are x=8 ... 16 (``10fingerprints.py -x``) and similarity index sim=0.1
... 0.3 (``20cluster.py <sim>``). However, values above sim=0.3 will quickly
recognize all sorts of images as equal and usually, results obtained with sim >
0.5 are seldom useful. Next, the higher x, the more detailed and close to the
original image the fingerprint is, since the original is squashed down to a x
times x grayscale image, converted to binary (pixed is black or white), and
flattened into a vector of x**2 length (the "fingerprint"). High values such as
64 already result in detection of only extremely similar images, since too much
information of the original image is kept. Instead, a value of x=2 is obviously
useless because it reduces each image to a 2x2 matrix (and fingerprint vector of
length 4) with almost no information from the original image left. So x=8 and
sim=0.1 is roughly equivalent to x=16 and sim=0.3. Therefore, if you get too
many "false positives" (images counted as similar when they are not), reduce
sim or increase x.

The original goal was to have a clustering based on classification of image
*content* -- smth like: image A this an image of my kitchen; image B is also an
image of my kitchen, only from a different angle and some persons in the
foreground, but the information -- this is my kitchen -- is the same. This is a
feature-detection task which relies on the ability to recognize *objects*
within a scene, regardless of other scene parameters (like view angle, color,
light, ...). It turns out that we need Neural Networks (you know: Tensor Flow
etc) and some real machine learning for the generation of better feature
vectors, e.g. a feature vector that always encodes the information "my
kitchen". The simple image hashing done here is rather limited in that respect.
It only does a very pedestrian smoothing / low-pass filtering to reduce the
noise and extract the "important" parts of the image. But this helps to find
duplicates and almost-duplicates in a collection of photos. And we learn how
to do clustering with scipy!


image fingerprints: simple and fast
-----------------------------------
These methods basically squash down the image to something like 16x16,
transform to gray scale and store that as a feature vector of length 16x16, for
example -> fast. But the method is not invariant against rotation, only scaling
along x and/or y. 

The idea is always to calculate a database of image fingerprints ("hashes",
feature vectors) and then do searches in feature space (all fingerprints) using
some form of KD-tree / nearest neighbor search.

* google: calculate image fingerprint
* [a|p|d]hash:
  https://realpython.com/blog/python/fingerprinting-images-for-near-duplicate-detection/ 
* especially: phash.org
* older Perl implementation of a ahash(?)-like method:
  http://www.jhnc.org/findimagedupes/manpage.html, also as Debian package

more scientific "feature extraction"
------------------------------------

* classical CV (computer vision): SIFT (good but slow, old-school
  hand-engineered feature detector), SURF (faster version of
  SIFT)
* http://opencv-python-tutroals.readthedocs.org/en/latest/index.html
    * SIFT and SURF are patented, so use ORB instead
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
