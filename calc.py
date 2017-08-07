# https://github.com/JohannesBuchner/imagehash
# http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html

# help for PIL.Image.Image.resize
# -------------------------------
# 
# PIL.Image.Image.resize(self, size, resample=0)
#
# :param size: The requested size in pixels, as a 2-tuple:
#    (width, height).
# :param resample: An optional resampling filter.  This can be
#    one of :py:attr:`PIL.Image.NEAREST` (use nearest neighbour),
#    :py:attr:`PIL.Image.BILINEAR` (linear interpolation),
#    :py:attr:`PIL.Image.BICUBIC` (cubic spline interpolation), or
#    :py:attr:`PIL.Image.LANCZOS` (a high-quality downsampling filter).
#    If omitted, or if the image has mode "1" or "P", it is
#    set :py:attr:`PIL.Image.NEAREST`.
# :returns: An :py:class:`~PIL.Image.Image` object.
# 
# Each PIL.Image.<method> variable is actually an integer (e.g. Image.NEAREST
# is 0).
#
# We tried the resample interpolation methods and measured the speed (ipython's
# timeit) for resizing an image 3840x2160 -> 8x8. We also compared the resample
# quality as difference to the best possible resample result, which we defined
# to be the LANCZOS method (from visual inspection and b/c it is
# computationally the most elaborate).
#
#                                      speed [ms]     
# Image.NEAREST                   = 0  29.9e-3
# Image.LANCZOS = Image.ANTIALIAS = 1  123          # reference result
# Image.BILINEAR                  = 2  47
# Image.BICUBIC                   = 3  87
#
# resample quality (see pil_resample_methods.py)
# method = 0, diff to ref(1) = 1.0
# method = 1, diff to ref(1) = 0.0
# method = 2, diff to ref(1) = 0.135679761399
# method = 3, diff to ref(1) = 0.0549413095836
#
# -> method=2 is probably the best compromise between quality and speed

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
