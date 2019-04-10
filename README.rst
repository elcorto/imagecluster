About
=====

Package for clustering images by content. We use a pre-trained deep
convolutional neural network to calculate image fingerprints which represent
content. Those are used to cluster similar images. In addition to pure
image content, it is possible to mix in timestamp information which improves
clustering for temporally uncorrelated images.

Usage
=====

The package is designed as a library. See ``examples/example_api.py``.

.. Here is what you can do:

.. .. code:: python
.. example_api.py

The bottleneck is ``~imagecluster.calc.fingerprints``, all other
operations have negligible relative cost.

Have a look at the clusters (as dirs with symlinks to the relevant files):

.. code:: sh

    $ tree pics/imagecluster/clusters/
    pics/imagecluster/clusters/
    ├── cluster_with_2
    │   ├── cluster_0
    │   │   ├── 140100.jpg -> /path/to/pics/140100.jpg
    │   │   └── 140101.jpg -> /path/to/pics/140101.jpg
    │   ├── cluster_1
    │   │   ├── 140600.jpg -> /path/to/pics/140600.jpg
    │   │   └── 140601.jpg -> /path/to/pics/140601.jpg
    │   ├── cluster_2
    │   │   ├── 140400.jpg -> /path/to/pics/140400.jpg
    │   │   └── 140401.jpg -> /path/to/pics/140401.jpg
    │   ├── cluster_3
    │   │   ├── 140501.jpg -> /path/to/pics/140501.jpg
    │   │   └── 140502.jpg -> /path/to/pics/140502.jpg
    │   ├── cluster_4
    │   │   ├── 140000.jpg -> /path/to/pics/140000.jpg
    │   │   └── 140001.jpg -> /path/to/pics/140001.jpg
    │   ├── cluster_5
    │   │   ├── 140300.jpg -> /path/to/pics/140300.jpg
    │   │   └── 140301.jpg -> /path/to/pics/140301.jpg
    │   └── cluster_6
    │       ├── 140200.jpg -> /path/to/pics/140200.jpg
    │       └── 140201.jpg -> /path/to/pics/140201.jpg
    └── cluster_with_3
        └── cluster_0
            ├── 140801.jpg -> /path/to/pics/140801.jpg
            ├── 140802.jpg -> /path/to/pics/140802.jpg
            └── 140803.jpg -> /path/to/pics/140803.jpg

So there are some clusters with 2 images each, and one with 3 images. Lets look
at the clusters:

.. image:: doc/clusters.png

For this example, we use a very small subset of the `Holiday image dataset
<holiday_>`_ (25 images (all named 140*.jpg) of 1491 total images in the
dataset). See ``examples/inria_holiday.sh`` for how to select such a subset:

.. code:: sh

    $ /path/to/imagecluster/examples/inria_holiday.sh jpg/140*

Here is the result of using a larger subset of 292 images from the same dataset
(``/inria_holiday.sh jpg/14*``):

.. image:: doc/clusters_many.png

Methods
=======

Clustering and similarity index
-------------------------------

We use `hierarchical clustering <hc_>`_ (``calc.cluster()``), which compares
the image fingerprints (4096-dim vectors) using a distance metric and produces
a `dendrogram <dendro_>`_ as an intermediate result. This shows how the images
can be grouped together depending on their similarity (y-axis).

.. image:: doc/dendrogram.png

One can now cut through the dendrogram tree at a certain height (``sim``
parameter 0...1, y-axis) to create clusters of images with that level of
similarity. ``sim=0`` is the root of the dendrogram (top in the plot) where
there is only one node (= all images in one cluster). ``sim=1`` is equal to the
end of the dendrogram tree (bottom in the plot), where each image is its own
cluster. By varying the index between 0 and 1, we thus increase the number of
clusters from 1 to the number of images. However, note that we only report
clusters with at least 2 images, such that ``sim=1`` will in fact produce no
results at all (unless there are completely identical images).

Image fingerprints
------------------

The task of the fingerprints (feature vectors) is to represent an image's
content (mountains, car, kitchen, person, ...). Deep convolutional neural
networks trained on many different images have developed an internal
representation of objects in higher layers, which we use for that purpose.

To this end, we use a pre-trained NN (VGG16_ as implemented by Keras_). The
weights will be downloaded *once* by Keras automatically upon first import and
placed into ``~/.keras/models/``. The network was trained on ImageNet_ and is
able to categorize images into 1000 classes (the last layer has 1000 nodes). We
use (`thanks for the hint! <alexcnwy_>`_) the activations of the second to last
fully connected layer ('fc2', 4096 nodes) as image fingerprints (numpy 1d array
of shape ``(4096,)``) by default.

Content and time distance
-------------------------

Image fingerprints represent content. Clustering based on content ignores time
correlations. Say we have two images of some object that look similar and will
thus be put into the same cluster. However, they might be in fact pictures of
different objects, taken at different times -- which is our original holiday
image use case (e.g. two images of a church from different cities, taken on
separate trips). In this case, we want the images to end up in different
clusters. We have a feature to mix content distance (``d_c`` and time distance
``d_t``) such that

::

    d = (1 - alpha) * d_c * ahpha * d_t

One can thus do pure content-based clustering (``alpha=0``) or pure time-based
(``alpha=1``). The effect of the mixing is that fingerprint points representing
content get pushed further apart when the corresponding images' time distance
is large. That way, we achieve a transparent addition of time information w/o
changing the clustering method.


Quality of clustering & parameters to tune
------------------------------------------

You may have noticed that in the example above, only 17 out of 25 images are
put into clusters. The others are not assigned to any cluster. Technically they
are in clusters of size 1, which we don't report by default (unless you use
``calc.cluster(..., min_csize=1)``). One can now start to lower ``sim`` to
find a good balance of clustering accuracy and the tolerable amount of
dissimilarity among images within a cluster.

Also, the parameters of the clustering method itself are worth tuning. ATM, we
expose only some in ``calc.cluster()``. We tested several distance metrics and
linkage methods, but this could nevertheless use a more elaborate evaluation.
See ``calc.cluster()`` for "method", "metric" and "criterion" and the scipy
functions called. If you do this and find settings which perform much better --
PRs welcome!

Additionally, some other implementations do not use any of the inner fully
connected layers as features, but instead the output of the last pooling
layer (layer 'flatten' in Keras' VGG16). We tested that briefly (see
``get_model(... layer='fc2')`` or ``main(..., layer='fc2')`` and found our
default 'fc2' to perform well enough. 'fc1' performs almost the same, while
'flatten' seems to do worse. But again, a quantitative analysis is in order.

PCA: Because of the `Curse of dimensionality <curse_>`_, it may be helpful to
perform a PCA on the fingerprints before clustering to reduce the feature
vector dimensions to, say, a few 100, thus making the distance metrics used in
clustering more effective. However, our tests so far show no substantial change
in clustering results, in accordance to what `others have found
<gh_beleidy_>`_. See ``examples/example_api.py`` and ``calc.pca()``.


Tests
=====

See ``imagecluster/tests/``. Use a test runner such as ``nosetests`` or
``pytest``.


Install
=======

.. code:: sh

    $ pip3 install -e .

See also samplepkg_.

Contributions
=============

Contributions are welcome. To streamline the git log, consider using one of
the prefixes mentioned `here <commit_pfx_>`_ in your commit message.


Related projects
================

* https://artsexperiments.withgoogle.com/tsnemap/
* https://github.com/YaleDHLab/pix-plot
* https://github.com/beleidy/unsupervised-image-clustering
* https://github.com/zegami/image-similarity-clustering
* https://github.com/sujitpal/holiday-similarity


.. _VGG16: https://arxiv.org/abs/1409.1556
.. _Keras: https://keras.io
.. _ImageNet: http://www.image-net.org/
.. _alexcnwy: https://github.com/alexcnwy
.. _hc: https://en.wikipedia.org/wiki/Hierarchical_clustering
.. _dendro: https://en.wikipedia.org/wiki/Dendrogram
.. _holiday: http://lear.inrialpes.fr/~jegou/data.php
.. _curse: https://en.wikipedia.org/wiki/Curse_of_dimensionality
.. _gh_beleidy: https://github.com/beleidy/unsupervised-image-clustering
.. _commit_pfx: https://github.com/elcorto/libstuff/blob/master/commit_prefixes
.. _samplepkg: https://github.com/elcorto/samplepkg
