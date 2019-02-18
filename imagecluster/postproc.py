import os
import shutil

from matplotlib import pyplot as plt
import numpy as np

from . import calc as ic

pj = os.path.join


def plot_clusters(clusters, ias, max_csize=None):
    """Plot `clusters` of images in `ias`.

    For interactive work, use :func:`visualize` instead.

    Parameters
    ----------
    clusters : see :func:`imagecluster.cluster`
    ias : see :func:`imagecluster.image_arrays`
    """
    stats = ic.cluster_stats(clusters)
    ncols = sum(list(stats.values()))
    nrows = max(stats.keys())
    if max_csize is not None:
        nrows = min(max_csize, nrows)
    shape = ias[list(ias.keys())[0]].shape[:2]
    arr = np.ones((nrows*shape[0], ncols*shape[1], 3), dtype=int) * 255
    icol = -1
    for nelem in np.sort(list(clusters.keys())):
        for cluster in clusters[nelem]:
            icol += 1
            for irow, filename in enumerate(cluster[:nrows]):
                img_arr = ias[filename]
                arr[irow*shape[0]:(irow+1)*shape[0], icol*shape[1]:(icol+1)*shape[1], :] = img_arr
    fig_scale = 1/shape[0]
    figsize = np.array(arr.shape[:2][::-1])*fig_scale
    fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(arr)
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig,ax


def visualize(*args, **kwds):
    plot_clusters(*args, **kwds)
    plt.show()


def make_links(clusters, cluster_dr):
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
