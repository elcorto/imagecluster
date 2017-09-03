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
        filedir = tempfile.mkdtemp(prefix='imagecluster_')
        arr = misc.face()
        images = [arr, 
                  ni.gaussian_filter(arr, 10), 
                  ni.gaussian_filter(arr, 20)]
        image_fns = []
        for idx, arr in enumerate(images):
            fn = pj(filedir, 'image_{}.png'.format(idx))
            misc.imsave(fn, arr)
            image_fns.append(fn)
        # first run: create fingerprints database, run clustering
        main.main(filedir)
        # second run, only run clustering, should be much faster
        main.main(filedir)
        with open(pj(filedir, 'fingerprints.pk'), 'rb') as fd:
            fps = pickle.load(fd)
        assert len(fps.keys()) == 3
        assert set(fps.keys()) == set(image_fns)
        for kk,vv in fps.items():
            assert isinstance(vv, np.ndarray)
            assert len(vv) == 4096
    finally:
        shutil.rmtree(filedir)
