# https://github.com/JohannesBuchner/imagehash
# http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html

from imgcmp import env
import numpy as np
import scipy.fftpack as fftpack
from scipy.spatial import distance
from scipy.cluster import hierarchy

INT = np.int32
FLOAT = np.float64


def img2arr(img, size=(8,8), dtype=INT, resample=2):
    """Convert PIL Image to gray scale and resample to numpy array of shape
    `size` and `dtype`.

    Parameters
    ----------
    img : PIL Image
    size : (int, int)
        size of square fingerprint array for `img`
    resample : int
        interpolation method, see help of ``PIL.Image.Image.resize``
    """
    # convert('L'): to 1D grey scale array
    return np.array(img.convert("L").resize(size, resample), dtype=dtype)


def ahash(img, size=(8,8)):
    """
    Parameters
    ----------
    img : PIL image
    size : (int, int)
        size of square fingerprint array for `img`
    """
    pixels = img2arr(img, size=size)
    return (pixels > pixels.mean()).astype(INT)


def phash(img, size=(8,8), highfreq_factor=4, backtransform=False):
    img_size = (size[0]*highfreq_factor, 
                size[1]*highfreq_factor)
    pixels = img2arr(img, size=img_size, dtype=np.float64)
    fpixels = fftpack.dct(fftpack.dct(pixels, axis=0), axis=1)
    # XXX we had fpixels[1:size[0], 1:size[1]] before, find out why
    fpixels_lowfreq = fpixels[:size[0], :size[1]]
    if backtransform:
        tmp = fftpack.idct(fftpack.idct(fpixels_lowfreq, axis=0), axis=1)
    else:
        tmp = fpixels_lowfreq
    return (tmp > np.median(tmp)).astype(INT)


def dhash(img, size=(8,8)):
    pixels = img2arr(img, size=(size[0] + 1, size[1]))
    return (pixels[1:, :] > pixels[:-1, :]).astype(INT)


def cluster(files, fps, sim=0.2, metric='hamming'):
    """Hierarchical clustering of images `files` based on image fingerprints
    `fps`.

    files : list of file names
    sim : float
        similarity (1=max. allowed similarity, all images are considered
        similar and are in one cluster, 0=zero similarity allowed, each image
        is it's own cluster of size 1)
    fps : 2d array (len(files), size[0]*size[1])
        flattened fingerprints (1d array for each image), where `size` is the
        argument to one of the *hash() functions 
    """
    dfps = distance.pdist(fps.astype(bool), metric)
    # hierarchical/agglomerative clustering (Z = linkage matrix, construct
    # dendrogram) 
    Z = hierarchy.linkage(dfps, method='average', metric=metric)
    # cut dendrogram, extract clusters
    cut = hierarchy.fcluster(Z, t=dfps.max()*sim, criterion='distance')
    clusters = dict((ii,[]) for ii in np.unique(cut))
    for iimg,iclus in enumerate(cut):
        clusters[iclus].append(files[iimg])
    return clusters 
