"""Only tests where we need to import tensorflow, which takes long."""

import numpy as np
import scipy.ndimage as ni
from scipy import misc
import tempfile, shutil, os, pickle

# https://stackoverflow.com/a/39708493
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from imagecluster import main
pj = os.path.join


def test():
    """Basic test of main()."""
    try:
        imagedir = tempfile.mkdtemp(prefix='imagecluster_')
        dbfn = pj(imagedir, main.ic_base_dir, 'fingerprints.pk')
        arr = misc.face()
        images = [arr,
                  ni.gaussian_filter(arr, 10),
                  ni.gaussian_filter(arr, 20),
                  arr[...,0], # fake gray-scale image
                  ]
        image_fns = []
        for idx, arr in enumerate(images):
            fn = pj(imagedir, 'image_{}.png'.format(idx))
            misc.imsave(fn, arr)
            image_fns.append(fn)
        # run 1: create fingerprints database, run clustering
        main.main(imagedir)
        # run 2: only run clustering, should be much faster, this time use PCA
        main.main(imagedir, pca=True)
        with open(dbfn, 'rb') as fd:
            fps = pickle.load(fd)
        assert len(fps.keys()) == 4
        assert set(fps.keys()) == set(image_fns)
        for kk,vv in fps.items():
            assert isinstance(vv, np.ndarray)
            assert len(vv) == 4096
    finally:
        shutil.rmtree(imagedir)
