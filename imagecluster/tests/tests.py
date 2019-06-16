import logging
import os
import shutil
import tempfile
import copy
import datetime

import numpy as np
from matplotlib.pyplot import imsave
import PIL.Image
import piexif

from imagecluster import calc as ic
from imagecluster import io as icio


# https://stackoverflow.com/a/39708493
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)

pj = os.path.join

# TODO re-use ImagedirCtx where possible, we write files in each context,
# re-use ctxs which don't alter the files

class ImagedirCtx:
    def __init__(self, fmt='png'):
        assert fmt in ['jpg', 'png']
        date_time_base_dct = dict(year=2019,
                                  month=12,
                                  day=31,
                                  hour=23,
                                  minute=42)
        imagedir = tempfile.mkdtemp(prefix='imagecluster_')
        dbfn = pj(imagedir, icio.ic_base_dir, 'fingerprints.pk')
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
        second = 0
        for color, arrs in images.items():
            nimg = len(arrs)
            clus = clusters.get(nimg, [])
            for idx, arr in enumerate(arrs):
                if fmt == 'png':
                    fn = pj(imagedir, f'image_{color}_{idx}.png')
                    imsave(fn, arr)
                elif fmt == 'jpg':
                    fn = pj(imagedir, f'image_{color}_{idx}.jpg')
                    img = PIL.Image.fromarray(arr, mode='RGB')
                    # just the DateTime field
                    date_time_dct = copy.deepcopy(date_time_base_dct)
                    date_time_dct.update(second=second)
                    exif_date_time_fmt = '{year}:{month}:{day} {hour}:{minute}:{second}'
                    exif_date_time_str = exif_date_time_fmt.format(**date_time_dct)
                    piexif_exif_dct = {'0th': {306: exif_date_time_str}}
                    img.save(fn, exif=piexif.dump(piexif_exif_dct))
                image_fns.append(fn)
                clus.append(fn)
                second += 1
            clusters[nimg] = [clus]
        self.imagedir = imagedir
        self.dbfn = dbfn
        self.image_fns = image_fns
        self.clusters = clusters
        self.date_time_base_dct = date_time_base_dct
        print(clusters)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        shutil.rmtree(self.imagedir)


def test_api_get_image_data():
    with ImagedirCtx() as ctx:
        # run 1: create fingerprints database, run clustering
        images,fingerprints,timestamps = icio.get_image_data(ctx.imagedir)
        # run 2: only run clustering, should be much faster, this time use all
        # kwds (test API)
        images,fingerprints,timestamps = icio.get_image_data(
            ctx.imagedir,
            pca_kwds=dict(n_components=0.95),
            model_kwds=dict(layer='fc2'),
            img_kwds=dict(size=(224,224)),
            timestamps_kwds=dict(source='auto'))
        assert len(fingerprints.keys()) == len(ctx.image_fns)
        assert set(fingerprints.keys()) == set(ctx.image_fns)


def test_low_level_api_and_clustering():
    # use low level API (same as get_image_data) but call all funcs
    # test clustering
    with ImagedirCtx() as ctx:
        images = icio.read_images(ctx.imagedir, size=(224,224))
        model = ic.get_model()
        fingerprints = ic.fingerprints(images, model)
        for kk,vv in fingerprints.items():
            assert isinstance(vv, np.ndarray)
            assert len(vv) == 4096, len(vv)
        fingerprints = ic.pca(fingerprints, n_components=0.95)
        clusters = ic.cluster(fingerprints, sim=0.5)
        assert set(clusters.keys()) == set(ctx.clusters.keys())
        assert len(fingerprints.keys()) == len(ctx.image_fns)
        assert set(fingerprints.keys()) == set(ctx.image_fns)
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
        alpha1 = np.ones(shape2d, dtype=np.uint8) * 255  # white
        alpha2 = np.zeros(shape2d, dtype=np.uint8)  # black
        alpha3 = (np.random.rand(*shape2d) * 255).astype(np.uint8)  # noise
        for alpha in [alpha1, alpha2, alpha3]:
            rgba = np.empty(shape2d + (4,), dtype=np.uint8)
            rgba[..., :3] = rgb
            rgba[..., 3] = alpha
            assert rgba.max() <= 255
            imsave(fn, rgba)
            img = PIL.Image.open(fn)
            assert img.mode == 'RGBA', img.mode
            assert img.format == 'PNG', img.format
            rgb2 = np.array(PIL.Image.open(fn).convert('RGB'))
            assert (rgb == rgb2).all()
            assert rgb.dtype == rgb2.dtype
    finally:
        if os.path.exists(fn):
            os.remove(fn)


def test_img_timestamp():
    with ImagedirCtx(fmt='jpg') as ctx:
        for second, fn in enumerate(ctx.image_fns):
            stamp = icio.exif_timestamp(fn)
            dct = copy.deepcopy(ctx.date_time_base_dct)
            dct.update(second=second)
            ref = datetime.datetime(**dct, tzinfo=datetime.timezone.utc).timestamp()
            assert stamp is not None
            assert stamp == ref, f"stamp={stamp} ref={ref}"
            # try EXIF first
            assert stamp == icio.timestamp(fn, source='auto')
            assert stamp == icio.timestamp(fn, source='exif')

    with ImagedirCtx(fmt='png') as ctx:
        fn = ctx.image_fns[0]
        assert icio.stat_timestamp(fn) is not None
        assert icio.timestamp(fn, source='auto') is not None
        assert icio.timestamp(fn, source='auto') == icio.stat_timestamp(fn)
