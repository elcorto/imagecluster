About
=====

Package for comparing and clustering images by content. We use a pre-trained
deep convolutional neural network for calculating image fingerprints, which are
then used to cluster similar images.

Install
=======

::

    $ git clone https://github.com/elcorto/imagecluster.git
    $ cd imagecluster
    $ pip3 install -e .

or::

    $ python3 setup develop --prefix=...

Usage
=====

We use a pre-trained keras NN model. The weights will be downloaded *once* by
keras automatically upon first import and placed into ``~/.keras/models/``.

See ``imagecluster.main.main()`` for a usage example.

If there is no fingerprints database, it will first run all images through the
NN model and calculate fingerprints. Then it will cluster the images based on
the fingerprints and a similarity index (more details below).

Example session::

    >>> from imagecluster import main
    >>> main.main('/path/to/testpics/', sim=0.5)
    no fingerprints database /path/to/testpics/fingerprints.pk found
    running all images through NN model ...
    /path/to/testpics/DSC_1061.JPG
    /path/to/testpics/DSC_1080.JPG
    ...
    /path/to/testpics/DSC_1087.JPG
    clustering ...
    cluster dir: /path/to/testpics/clusters
    items per cluster : number of such clusters
    2 : 7
    3 : 2
    4 : 4
    5 : 1
    10 : 1

Have a look at the clusters (as dirs with symlinks to the relevant files)::

    $ tree /path/to/testpics
    /path/to/testpics/clusters
    ├── cluster_with_10
    │   └── cluster_0
    │       ├── DSC_1068.JPG -> /path/to/testpics/DSC_1068.JPG
    │       ├── DSC_1070.JPG -> /path/to/testpics/DSC_1070.JPG
    │       ├── DSC_1071.JPG -> /path/to/testpics/DSC_1071.JPG
    │       ├── DSC_1072.JPG -> /path/to/testpics/DSC_1072.JPG
    │       ├── DSC_1073.JPG -> /path/to/testpics/DSC_1073.JPG
    │       ├── DSC_1074.JPG -> /path/to/testpics/DSC_1074.JPG
    │       ├── DSC_1075.JPG -> /path/to/testpics/DSC_1075.JPG
    │       ├── DSC_1076.JPG -> /path/to/testpics/DSC_1076.JPG
    │       ├── DSC_1077.JPG -> /path/to/testpics/DSC_1077.JPG
    │       └── DSC_1078.JPG -> /path/to/testpics/DSC_1078.JPG
    ├── cluster_with_2
    │   ├── cluster_0
    │   │   ├── DSC_1037.JPG -> /path/to/testpics/DSC_1037.JPG
    │   │   └── DSC_1038.JPG -> /path/to/testpics/DSC_1038.JPG
    │   ├── cluster_1
    │   │   ├── DSC_1053.JPG -> /path/to/testpics/DSC_1053.JPG
    │   │   └── DSC_1054.JPG -> /path/to/testpics/DSC_1054.JPG
    │   ├── cluster_2
    │   │   ├── DSC_1046.JPG -> /path/to/testpics/DSC_1046.JPG
    │   │   └── DSC_1047.JPG -> /path/to/testpics/DSC_1047.JPG
    ...

If you run this again on the same directory, only the clustering will be
repeated.

Methods
=======

Image fingerprints
------------------

The original goal was to have a clustering based on classification of image
*content* such as: image A this an image of my kitchen; image B is also an
image of my kitchen, only from a different angle and some persons in the
foreground, but the information -- this is my kitchen -- is the same. This is a
feature-detection task which relies on the ability to recognize the content of
the scene, regardless of other scene parameters (like view angle, color, light,
...). It turns out that we can use deep convolutional neural networks
(convnets) for the generation of good *feature vectors*, e.g. a feature vector
that always encodes the information "my kitchen", since deep nets, once trained
on many different images, have developed an internal representation of objects
like chair, boat, car .. and kitchen. Simple image hashing, which we used
previously, is rather limited in that respect. It only does a very pedestrian
smoothing / low-pass filtering to reduce the noise and extract the "important"
parts of the image. This helps to find duplicates and almost-duplicates in a
collection of photos. 

To this end, we use a pre-trained NN (VGG16_ as implemented by Keras_). The
network was trained on ImageNet_ and is able to categorize images into 1000
classes (the last layer has 1000 nodes). We chop off the last layer (`thanks
for the hint! <alexcnwy_>`_) and use the activations of the second to last fully
connected layer (4096 nodes) as image fingerprints (numpy 1d array of shape
``(4096,)``).

The package can detect images which are rather similar, e.g. the same scene
photographed twice or more with some camera movement in between, or a scene
with the same background and e.g. one person exchanged. This was also possible
with image hashes. 

Now with NN-based fingerprints, we also cluster all sorts of images which have,
e.g. mountains, tents, or beaches, so this is far better. However, if you run
this on a large collection of images which contain images with tents or
beaches, then the system won't recognize that certain images belong together
because they were taken on the same trip, for instance. All tent images will be
in one cluster, and so will all beaches images. This is probably b/c in this
case, the human classification of the image works by looking at the background
as well. A tent in the center of the image will always look the same, but it is
the background which makes humans distinguish the context. The problem is:
VGG16 and all the other popular networks have been trained on ridiculously
small images of 224x224 size because of computational limitations, where it is
impossible to recognize background details. Another point is that the
background image triggers the activation of meta-information associated with
that background in the human -- data which wasn't used when training ImageNet,
of course. Thus, one way to improve things would be to re-train the network
using this information. But then one would have labeled all images by hand
again.


Clustering
----------

We use hierarchical clustering, see ``imagecluster.calc.cluster()``. The image
fingerprints (4096-dim vectors) are compared using a distance metric and
similar images are put together in a cluster. The threshold for what counts as
similar is defined by a similar index (again, see ``calc.cluster()``).

The index can be thought of as the allowed *dissimilarity* or a similarity
tolerance. A small index means to put only very similar images in one cluster.
The extreme case 0.0 means to allow zero dissimilarity and thus to put each image
in a cluster of size 1. In contrast, large values imply less strict clustering
and will put more but less similar images in a cluster. A value of 1.0 is equal
to putting all images in one single cluster (all images are treated as
equal).

Tests
=====

Run ``nosetests3`` (nosetests for Python3, Linux).

.. _VGG16: https://arxiv.org/abs/1409.1556
.. _Keras: https://keras.io
.. _ImageNet: http://www.image-net.org/
.. _alexcnwy: https://github.com/alexcnwy
