import datetime
import functools
import multiprocessing as mp
import os
import pickle
import re

from keras.preprocessing import image
import PIL.Image

from . import exceptions
from . import calc as ic

pj = os.path.join

ic_base_dir = 'imagecluster'


def read_pk(filename):
    """Read pickled data from `filename`."""
    with open(filename, 'rb') as fd:
        ret = pickle.load(fd)
    return ret


def write_pk(obj, filename):
    """Write object `obj` pickled to `filename`."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as fd:
        pickle.dump(obj, fd)


def get_files(imagedir, ext='jpg|jpeg|bmp|png'):
    """Return all file names with extension matching the regex `ext` from dir
    `imagedir`.

    Parameters
    ----------
    imagedir : str
    ext : str
        regex

    Returns
    -------
    list
        list of file names
    """
    rex = re.compile(r'^.*\.({})$'.format(ext), re.I)
    return [os.path.join(imagedir,base) for base in os.listdir(imagedir)
            if rex.match(base)]


def exif_timestamp(filename):
    """Read timestamp from image in `filename` from EXIF tag.

    This will probably work for most JPG files, but not for PNG, for instance.

    Raises
    ------
    exceptions.ICExifReadError

    Returns
    -------
    float
        timestamp, seconds since Epoch
    """
    # PIL lazy-loads the image data, so this open and _getexif() is fast.
    img = PIL.Image.open(filename)
    if ('exif' not in img.info.keys()) or (not hasattr(img, '_getexif')):
        raise exceptions.ICExifReadError(f"no EXIF data found in {filename}")
    # Avoid constucting the whole EXIF dict just to extract the DateTime field.
    # DateTime -> key 306 is in the EXIF standard, so let's use that directly.
    ## date_time = {TAGS[k] : v for k,v in exif.items()}['DateTime']
    exif = img._getexif()
    key = 306
    if key not in exif.keys():
        raise exceptions.ICExifReadError(f"key 306 (DateTime) not found in "
                                         f"EXIF data of file {filename}")
    # '2019:03:10 22:42:42'
    date_time = exif[key]
    if date_time.count(':') != 4:
        msg = f"unsupported EXIF DateTime format in '{date_time}' of {filename}"
        raise exceptions.ICExifReadError(msg)
    # '2019:03:10 22:42:42' -> ['2019', '03', '10', '22', '42', '42']
    date_time_str = date_time.replace(':', ' ').split()
    names = ('year', 'month', 'day', 'hour', 'minute', 'second')
    stamp = datetime.datetime(**{nn:int(vv) for nn,vv in zip(names,date_time_str)},
                              tzinfo=datetime.timezone.utc).timestamp()
    return stamp


def stat_timestamp(filename):
    """File timestamp from file stats (mtime)."""
    return os.stat(filename).st_mtime


def timestamp(filename, source='auto'):
    """Read single timestamp for image in `filename`.

    Parameters
    ----------
    filename : str
    source : {'auto', 'stat', 'exif'}
        Read timestamps from file stats ('stat'), or EXIF tags ('exif'). If
        'auto', then try 'exif' first.

    Returns
    -------
    float
        timestamp, seconds since Epoch
    """
    if source == 'auto':
        try:
            return exif_timestamp(filename)
        except exceptions.ICExifReadError:
            return stat_timestamp(filename)
    elif source == 'stat':
        return stat_timestamp(filename)
    elif source == 'exif':
        return exif_timestamp(filename)
    else:
        raise ValueError("source not in ['stat', 'exif', 'auto']")


# TODO some code dups below, fix later by fancy factory functions

# keras.preprocessing.image.load_img() uses img.rezize(shape) with the default
# interpolation of Image.resize() which is pretty bad (see
# imagecluster/play/pil_resample_methods.py). Given that we are restricted to
# small inputs of 224x224 by the VGG network, we should do our best to keep as
# much information from the original image as possible. This is a gut feeling,
# untested. But given that model.predict() is 10x slower than PIL image loading
# and resizing .. who cares.
#
# (224, 224, 3)
##img = image.load_img(filename, target_size=size)
##... = image.img_to_array(img)
def _image_worker(filename, size):
    # Handle PIL error "OSError: broken data stream when reading image file".
    # See https://github.com/python-pillow/Pillow/issues/1510 . We have this
    # issue with smartphone panorama JPG files. But instead of bluntly setting
    # ImageFile.LOAD_TRUNCATED_IMAGES = True and hoping for the best (is the
    # image read, and till the end?), we catch the OSError thrown by PIL and
    # ignore the file completely. This is better than reading potentially
    # undefined data and process it. A more specialized exception from PILs
    # side would be good, but let's hope that an OSError doesn't cover too much
    # ground when reading data from disk :-)
    try:
        print(filename)
        img = PIL.Image.open(filename).convert('RGB').resize(size, resample=3)
        arr = image.img_to_array(img, dtype=int)
        return filename, arr
    except OSError as ex:
        print(f"skipping {filename}: {ex}")
        return filename, None


def _timestamp_worker(filename, source):
    try:
        return filename, timestamp(filename, source)
    except OSError as ex:
        print(f"skipping {filename}: {ex}")
        return filename, None


def read_images(imagedir, size, ncores=mp.cpu_count()):
    """Load images from `imagedir` and resize to `size`.

    Parameters
    ----------
    imagedir : str
    size : sequence length 2
        (width, height), used in ``Image.open(filename).resize(size)``
    ncores : int
        run that many parallel processes

    Returns
    -------
    dict
        {filename: 3d array (height, width, 3), ...}
    """
    _f = functools.partial(_image_worker, size=size)
    with mp.Pool(ncores) as pool:
        ret = pool.map(_f, get_files(imagedir))
    return {k: v for k,v in ret if v is not None}


def read_timestamps(imagedir, source='auto', ncores=mp.cpu_count()):
    """Read timestamps of all images in `imagedir`.

    Parameters
    ----------
    imagedir : str
    source : see :func:`~imagecluster.io.timestamp`
    ncores : int
        run that many parallel processes

    Returns
    -------
    dict
        {filename: timestamp (int, seconds since Epoch)}
    """
    _f = functools.partial(_timestamp_worker, source=source)
    with mp.Pool(ncores) as pool:
        ret = pool.map(_f, get_files(imagedir))
    return {k: v for k,v in ret if v is not None}


# TODO fingerprints and timestamps may have different images which have been
# skipped -> we need a data struct to hold all image data and mask out the
# skipped ones. For now we have a check in calc.cluster()
def get_image_data(imagedir, model_kwds=dict(layer='fc2'),
                   img_kwds=dict(size=(224,224)), timestamps_kwds=dict(source='auto'),
                   pca_kwds=None):
    """Convenience function to create `images`, `fingerprints`,
    `timestamps`.

    It checks for existing `images` and `fingerprints` database files on
    disk and load them if present. Running this again only loads data from
    disk, which is fast. Default locations::

       fingerprints: <imagedir>/imagecluster/fingerprints.pk
       images: <imagedir>/imagecluster/images.pk

    Parameters
    ----------
    imagedir : str
    model_kwds : dict
        passed to :func:`~imagecluster.calc.get_model`
    img_kwds : dict
        passed to :func:`~imagecluster.io.read_images`
    timestamps_kwds : dict
        passed to :func:`~imagecluster.io.read_timestamps`
    pca_kwds : dict
        passed to :func:`~imagecluster.calc.pca`, PCA is skipped if
        ``pca_kwds=None``

    Returns
    -------
    images : see :func:`~imagecluster.io.read_images`
    fingerprints : see :func:`~imagecluster.calc.fingerprints`
    timestamps : see :func:`~imagecluster.io.read_timestamps`
    """
    fingerprints_fn = pj(imagedir, ic_base_dir, 'fingerprints.pk')
    images_fn = pj(imagedir, ic_base_dir, 'images.pk')
    if os.path.exists(images_fn):
        print(f"reading image arrays {images_fn} ...")
        images = read_pk(images_fn)
    else:
        print(f"create image arrays {images_fn}")
        images = read_images(imagedir, **img_kwds)
        write_pk(images, images_fn)
    if os.path.exists(fingerprints_fn):
        print(f"reading fingerprints {fingerprints_fn} ...")
        fingerprints = read_pk(fingerprints_fn)
    else:
        print(f"create fingerprints {fingerprints_fn}")
        fingerprints = ic.fingerprints(images, ic.get_model(**model_kwds))
        if pca_kwds is not None:
            fingerprints = ic.pca(fingerprints, **pca_kwds)
        write_pk(fingerprints, fingerprints_fn)
    print(f"reading timestamps ...")
    if timestamps_kwds is not None:
        timestamps = read_timestamps(imagedir, **timestamps_kwds)
    return images, fingerprints, timestamps
