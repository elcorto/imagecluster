import os

from imagecluster import calc as ic
from imagecluster import common as co
from imagecluster import postproc as pp

pj = os.path.join


ic_base_dir = 'imagecluster'


def main(imagedir, sim=0.5, layer='fc2', size=(224,224), links=True, vis=False,
         maxelem=None):
    """Example main app using this library.

    Upon first invocation, the image and fingerprint databases are built and
    written to disk. Each new invocation loads those and only repeats
        * clustering
        * creation of links to files in clusters
        * visualization (if `vis=True`)

    This is good for playing around with the `sim` parameter, for
    instance, which only influences clustering.

    Parameters
    ----------
    imagedir : str
        path to directory with images
    sim : float (0..1)
        similarity index (see :func:`imagecluster.cluster`)
    layer : str
        which layer to use as feature vector (see
        :func:`imagecluster.get_model`)
    size : tuple
        input image size (width, height), must match `model`, e.g. (224,224)
    links : bool
        create dirs with links
    vis : bool
        plot images in clusters
    maxelem : max number of images per cluster for visualization (see
        :mod:`~postproc`)

    Notes
    -----
    imagedir : To select only a subset of the images, create an `imagedir` and
        symlink your selected images there. In the future, we may add support
        for passing a list of files, should the need arise. But then again,
        this function is only an example front-end.
    """
    fps_fn = pj(imagedir, ic_base_dir, 'fingerprints.pk')
    ias_fn = pj(imagedir, ic_base_dir, 'images.pk')
    ias = None
    if not os.path.exists(fps_fn):
        print(f"no fingerprints database {fps_fn} found")
        os.makedirs(os.path.dirname(fps_fn), exist_ok=True)
        model = ic.get_model(layer=layer)
        if not os.path.exists(ias_fn):
            print(f"create image array database {ias_fn}")
            ias = ic.image_arrays(imagedir, size=size)
            co.write_pk(ias, ias_fn)
        else:
            ias = co.read_pk(ias_fn)
        print("running all images through NN model ...")
        fps = ic.fingerprints(ias, model)
        co.write_pk(fps, fps_fn)
    else:
        print(f"loading fingerprints database {fps_fn} ...")
        fps = co.read_pk(fps_fn)
    print("clustering ...")
    clusters = ic.cluster(fps, sim)
    if links:
        pp.make_links(clusters, pj(imagedir, ic_base_dir, 'clusters'))
    if vis:
        if ias is None:
            ias = co.read_pk(ias_fn)
        pp.visualize(clusters, ias, maxelem=maxelem)
