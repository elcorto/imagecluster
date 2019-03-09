import logging
import os
import pickle
import shutil
import tempfile

import numpy as np
from matplotlib.pyplot import imsave
import PIL.Image

from imagecluster import main
from imagecluster import calc as ic


# https://stackoverflow.com/a/39708493
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)

pj = os.path.join


class ImagedirCtx:
    def __init__(self):
        imagedir = tempfile.mkdtemp(prefix='imagecluster_')
        dbfn = pj(imagedir, main.ic_base_dir, 'fingerprints.pk')
        arr = np.ones((500,600,3), dtype=np.uint8)
        white = np.ones_like(arr) * 255
        black = np.zeros_like(arr)
        red = np.ones_like(arr)
        red[...,0] *= 255
        images = dict(red=[red]*2,
                      white=[white]*3,
                      black=[black]*4)
        image_fns = []
        clusters = {}
        for color, arrs in images.items():
            nimg = len(arrs)
            clus = clusters.get(nimg, [])
            for idx, arr in enumerate(arrs):
                fn = pj(imagedir, f'image_{color}_{idx}.png')
                imsave(fn, arr)
                image_fns.append(fn)
                clus.append(fn)
            clusters[nimg] = [clus]
        self.imagedir = imagedir
        self.dbfn = dbfn
        self.image_fns = image_fns
        self.clusters = clusters
        print(clusters)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        shutil.rmtree(self.imagedir)


def test_main_basic():
    with ImagedirCtx() as ctx:
        # run 1: create fingerprints database, run clustering
        main.main(ctx.imagedir)
        # run 2: only run clustering, should be much faster, this time also use PCA
        main.main(ctx.imagedir, pca=True)
        with open(ctx.dbfn, 'rb') as fd:
            fps = pickle.load(fd)
        assert len(fps.keys()) == len(ctx.image_fns)
        assert set(fps.keys()) == set(ctx.image_fns)
        for kk,vv in fps.items():
            assert isinstance(vv, np.ndarray)
            assert len(vv) == 4096


def test_cluster():
    # use API
    # test clustering
    with ImagedirCtx() as ctx:
        ias = ic.image_arrays(ctx.imagedir, size=(224,224))
        model = ic.get_model()
        fps = ic.fingerprints(ias, model)
        fps = ic.pca(fps, n_components=0.95)
        clusters = ic.cluster(fps, sim=0.5)
        assert set(clusters.keys()) == set(ctx.clusters.keys())
        for nimg in ctx.clusters.keys():
            for val_clus, ref_clus in zip(clusters[nimg], ctx.clusters[nimg]):
                msg = f"ref_clus: {ref_clus}, val_clus: {val_clus}"
                assert set(ref_clus) == set(val_clus), msg


def test_png_rgba_io():
    fn = tempfile.mktemp(prefix='imagecluster_') + '.png'
    try:
        shape2d = (123,456)
        shape = shape2d + (3,)
        rgb = (np.random.rand(*shape) * 255).astype(np.uint8)
        alpha1 = np.ones(shape2d, dtype=np.uint8) * 255 # white
        alpha2 = np.zeros(shape2d, dtype=np.uint8) # black
        alpha3 = (np.random.rand(*shape2d) * 255).astype(np.uint8) # noise
        for alpha in [alpha1, alpha2, alpha3]:
            rgba = np.empty(shape2d + (4,), dtype=np.uint8)
            rgba[..., :3] = rgb
            rgba[..., 3] = alpha
            assert rgba.max() <= 255
            imsave(fn, rgba)
            img = PIL.Image.open(fn)
            assert img.mode == 'RGBA', img.mode
            assert img.format == 'PNG', img.format
            rgb2 = np.array(ic.load_img_rgb(fn))
            assert (rgb == rgb2).all()
            assert rgb.dtype == rgb2.dtype
    finally:
        if os.path.exists(fn):
            os.remove(fn)
