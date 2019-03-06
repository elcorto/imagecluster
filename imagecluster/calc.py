import os

import multiprocessing as mp
import functools
from collections import OrderedDict

import PIL.Image
from scipy.spatial import distance
from scipy.cluster import hierarchy
import numpy as np
from sklearn.decomposition import PCA

from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.models import Model

from . import common

pj = os.path.join


def get_model(layer='fc2'):
    """Keras Model of the VGG16 network, with the output layer set to `layer`.

    The default layer is the second-to-last fully connected layer 'fc2' of
    shape (4096,).

    Parameters
    ----------
    layer : str
        which layer to extract (must be of shape (None, X)), e.g. 'fc2', 'fc1'
        or 'flatten'
    """
    # base_model.summary():
    #     ....
    #     block5_conv4 (Conv2D)        (None, 15, 15, 512)       2359808
    #     _________________________________________________________________
    #     block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0
    #     _________________________________________________________________
    #     flatten (Flatten)            (None, 25088)             0
    #     _________________________________________________________________
    #     fc1 (Dense)                  (None, 4096)              102764544
    #     _________________________________________________________________
    #     fc2 (Dense)                  (None, 4096)              16781312
    #     _________________________________________________________________
    #     predictions (Dense)          (None, 1000)              4097000
    #
    base_model = VGG16(weights='imagenet', include_top=True)
    model = Model(inputs=base_model.input,
                  outputs=base_model.get_layer(layer).output)
    return model


# keras.preprocessing.image.load_img() uses img.rezize(shape) with the default
# interpolation of PIL.Image.resize() which is pretty bad (see
# imagecluster/play/pil_resample_methods.py). Given that we are restricted to
# small inputs of 224x224 by the VGG network, we should do our best to keep as
# much information from the original image as possible. This is a gut feeling,
# untested. But given that model.predict() is 10x slower than PIL image loading
# and resizing .. who cares.
#
# (224, 224, 3)
##img = image.load_img(fn, target_size=size)
##... = image.img_to_array(img)
def _img_worker(fn, size):
    print(fn)
    return fn, image.img_to_array(PIL.Image.open(fn).resize(size, 3),
                                  dtype=int)


def image_arrays(imagedir, size):
    """Load images from `imagedir` and resize to `size`.

    Parameters
    ----------
    imagedir : str
    size : sequence length 2
        (width, height), used in ``PIL.Image.open(filename).resize(size)``

    Returns
    -------
    dict
        {filename: 3d array (height, width, 3),
         ...
        }
    """
    _f = functools.partial(_img_worker, size=size)
    with mp.Pool(mp.cpu_count()) as pool:
        ret = pool.map(_f, common.get_files(imagedir))
    return dict(ret)


def fingerprint(img_arr, model):
    """Run image array (3d array) run through `model` (keras.models.Model).

    Parameters
    ----------
    img_arr : 3d array
        (3,x,y) or (x,y,3), depending on
        ``keras.preprocessing.image.img_to_array`` and ``image_data_format``
        (``channels_{first,last}``) in ``~/.keras/keras.json``, see
        :func:`imagecluster.main.image_arrays`
    model : keras.models.Model instance

    Returns
    -------
    fingerprint : 1d array
    """
    # (224, 224, 1) -> (224, 224, 3)
    #
    # Simple hack to convert a grayscale image to fake RGB by replication of
    # the image data to all 3 channels.
    #
    # Deep learning models may have learned color-specific filters, but the
    # assumption is that structural image features (edges etc) contibute more to
    # the image representation than color, such that this hack makes it possible
    # to process gray-scale images with nets trained on color images (like
    # VGG16).
    #
    # We assme channels_last here. Fix if needed.
    if img_arr.shape[2] == 1:
        img_arr = img_arr.repeat(3, axis=2)

    # (1, 224, 224, 3)
    arr4d = np.expand_dims(img_arr, axis=0)

    # (1, 224, 224, 3)
    arr4d_pp = preprocess_input(arr4d)
    return model.predict(arr4d_pp)[0,:]


# Cannot use multiprocessing (only tensorflow backend tested, rumor has it that
# the TF computation graph is not built multiple times, i.e. pickling (what
# multiprocessing does with _worker) doen't play nice with Keras models which
# use Tf backend). The call to the parallel version of fingerprints() starts
# but seems to hang forever. However, Keras with Tensorflow backend runs
# multi-threaded (model.predict()), so we can sort of live with that. Even
# though Tensorflow has not the best scaling on the CPU, on low core counts
# (2-4), it won't matter that much. Also, TF was built to run on GPUs, not
# scale out multi-core CPUs.
#
##def _worker(img_arr, model):
##    print(fn)
##    return fn, fingerprint(img_arr, model)
##
##
##def fingerprints(ias, model):
##    _f = functools.partial(_worker, model=model)
##    with mp.Pool(int(mp.cpu_count()/2)) as pool:
##        ret = pool.map(_f, ias.items())
##    return dict(ret)

def fingerprints(ias, model):
    """Calculate fingerprints for all image arrays in `ias`.

    Parameters
    ----------
    ias : see :func:`image_arrays`
    model : see :func:`fingerprint`

    Returns
    -------
    fingerprints : dict
        {filename1: array([...]),
         filename2: array([...]),
         ...
         }
    """
    fps = {}
    for fn,img_arr in ias.items():
        print(fn)
        fps[fn] = fingerprint(img_arr, model)
    return fps


def pca(fps, n_components=0.9, **kwds):
    if 'n_components' not in kwds.keys():
        kwds['n_components'] = n_components
    # Yes in recent Pythons, dicts are ordered in CPython, but still.
    _fps = OrderedDict(fps)
    X = np.array(list(_fps.values()))
    Xp = PCA(**kwds).fit(X).transform(X)
    return {k:v for k,v in zip(_fps.keys(), Xp)}


def cluster(fps, sim=0.5, method='average', metric='euclidean',
            extra_out=False, print_stats=True, min_csize=2):
    """Hierarchical clustering of images based on image fingerprints.

    Parameters
    ----------
    fps: dict
        output of :func:`fingerprints`
    sim : float 0..1
        similarity index
    method : see scipy.hierarchy.linkage(), all except 'centroid' produce
        pretty much the same result
    metric : see scipy.hierarchy.linkage(), make sure to use 'euclidean' in
        case of method='centroid', 'median' or 'ward'
    extra_out : bool
        additionally return internal variables for debugging
    print_stats : bool
    min_csize : int
        return clusters with at least that many elements

    Returns
    -------
    clusters [, extra]
    clusters : dict
        We call a list of file names a "cluster".
        keys = size of clusters (number of elements (images) `csize`)
        value = list of clusters with that size
        {csize : [[filename, filename, ...],
                  [filename, filename, ...],
                  ...
                  ],
         csize : [...]}
    extra : dict
        if `extra_out` is True
    """
    assert 0 <= sim <= 1, "sim not 0..1"
    assert min_csize >= 1, "min_csize must be >= 1"
    files = list(fps.keys())
    # array(list(...)): 2d array
    #   [[... fingerprint of image1 (4096,) ...],
    #    [... fingerprint of image2 (4096,) ...],
    #    ...
    #    ]
    dfps = distance.pdist(np.array(list(fps.values())), metric)
    # hierarchical/agglomerative clustering (Z = linkage matrix, construct
    # dendrogram), plot: scipy.cluster.hierarchy.dendrogram(Z)
    Z = hierarchy.linkage(dfps, method=method, metric=metric)
    # cut dendrogram, extract clusters
    # cut=[12,  3, 29, 14, 28, 27,...]: image i belongs to cluster cut[i]
    cut = hierarchy.fcluster(Z, t=dfps.max()*(1.0-sim), criterion='distance')
    cluster_dct = dict((iclus, []) for iclus in np.unique(cut))
    for iimg,iclus in enumerate(cut):
        cluster_dct[iclus].append(files[iimg])
    # group all clusters (cluster = list_of_files) of equal size together
    # {number_of_files1: [[list_of_files], [list_of_files],...],
    #  number_of_files2: [[list_of_files],...],
    # }
    clusters = {}
    for cluster in cluster_dct.values():
        csize = len(cluster)
        if csize >= min_csize:
            if not (csize in clusters.keys()):
                clusters[csize] = [cluster]
            else:
                clusters[csize].append(cluster)
    if print_stats:
        print_cluster_stats(clusters)
    if extra_out:
        extra = {'Z': Z, 'dfps': dfps, 'cluster_dct': cluster_dct, 'cut': cut}
        return clusters, extra
    else:
        return clusters


def cluster_stats(clusters):
    """Count clusters of different sizes.

    Returns
    -------
    2d array
        Array with column 1 = csize sorted (number of images in the cluster)
        and column 2 = cnum (number of clusters with that size).

        [[csize, cnum],
         [...],
        ]
    """
    return np.array([[k, len(clusters[k])] for k in
                     np.sort(list(clusters.keys()))], dtype=int)


def print_cluster_stats(clusters):
    print("#images : #clusters")
    stats = cluster_stats(clusters)
    for csize,cnum in stats:
        print(f"{csize} : {cnum}")
    if stats.shape[0] > 0:
        nimg = stats.prod(axis=1).sum()
    else:
        nimg = 0
    print("#images in clusters total: ", nimg)
