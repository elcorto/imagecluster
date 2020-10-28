import os
import shutil

from matplotlib import pyplot as plt
import numpy as np

from . import calc as ic

pj = os.path.join


def plot_clusters(clusters, images, max_csize=None, mem_limit=1024**3):
    """Plot `clusters` of images in `images`.

    For interactive work, use :func:`visualize` instead.

    Parameters
    ----------
    clusters : see :func:`~imagecluster.calc.cluster`
    images : see :func:`~imagecluster.io.read_images`
    max_csize : int
        plot clusters with at most this many images
    mem_limit : float or int, bytes
        hard memory limit for the plot array (default: 1 GiB), increase if you
        have (i) enough memory, (ii) many clusters and/or (iii) large
        max(csize) and (iv) max_csize is large or None
    """
    if len(clusters) < 1:
        print('Similarity is too high, each image is its own cluster, no need to visualize it.')
        return
    stats = ic.cluster_stats(clusters)
    if max_csize is not None:
        stats = stats[stats[:,0] <= max_csize, :]
    # number of clusters
    ncols = stats[:, 1].sum()
    # csize (number of images per cluster)
    nrows = stats[:,0].max()
    shape = images[list(images.keys())[0]].shape[:2]
    mem = nrows * shape[0] * ncols * shape[1] * 3
    if mem > mem_limit:
        raise Exception(f"size of plot array ({mem/1024**2} MiB) > mem_limit "
                        f"({mem_limit/1024**2} MiB)")
    # uint8 has range 0..255, perfect for images represented as integers, makes
    # rather big arrays possible
    arr = np.ones((nrows*shape[0], ncols*shape[1], 3), dtype=np.uint8) * 255
    icol = -1
    for csize in stats[:,0]:
        for cluster in clusters[csize]:
            icol += 1
            for irow, filename in enumerate(cluster):
                image = images[filename]
                arr[irow*shape[0]:(irow+1)*shape[0],
                    icol*shape[1]:(icol+1)*shape[1], :] = image
    print(f"plot array ({arr.dtype}) size: {arr.nbytes/1024**2} MiB")
    fig,ax = plt.subplots()
    ax.imshow(arr)
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig,ax


def visualize(*args, **kwds):
    """Interactive wrapper of :func:`plot_clusters`. Just calls ``plt.show`` at
    the end. Doesn't return ``fig,ax``.
    """
    plot_clusters(*args, **kwds)
    plt.show()


def make_links(clusters, cluster_dr):
    """In `cluster_dr`, create nested dirs with symlinks to image files
    representing `clusters`.

    Parameters
    ----------
    clusters : see :func:`~imagecluster.calc.cluster`
    cluster_dr : str
        path
    """
    print("cluster dir: {}".format(cluster_dr))
    if os.path.exists(cluster_dr):
        shutil.rmtree(cluster_dr)
    for csize, group in clusters.items():
        for iclus, cluster in enumerate(group):
            dr = pj(cluster_dr,
                    'cluster_with_{}'.format(csize),
                    'cluster_{}'.format(iclus))
            for fn in cluster:
                link = pj(dr, os.path.basename(fn))
                os.makedirs(os.path.dirname(link), exist_ok=True)
                os.symlink(os.path.abspath(fn), link)
