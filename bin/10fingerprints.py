#!/usr/bin/python3

import sys, multiprocessing, functools, argparse, os
from PIL import Image
from imgcmp import calc, io, cli, env
import numpy as np
pj = os.path.join


def _worker(tup, size_x=None, fpsdct=None):
    ii, name = tup     
    img = Image.open(name)
    fpsdct[ii] = calc.phash(img, 
                            size=(size_x, size_x), 
                            highfreq_factor=4, 
                            backtransform=False).flatten()

if __name__ == '__main__':
    
    desc = """
Calculate fingerprint database.    
"""
    parser = argparse.ArgumentParser(description=desc) 
    parser.add_argument('files', metavar='FILE', nargs='*',
                        default=[pj(cli.convert_dr, x) for x in \
                                 os.listdir(cli.convert_dr)],
                        help='image file names, [default: '
                             '{}/*]'.format(cli.convert_dr))
    parser.add_argument('-x', dest='size_x',
                        default=8, type=int,
                        help='resize images to (size_x, size_x), fingerprints '
                             'are then (size_x**2,) 1d arrays; large = more '
                             'features, more strict similarity [%(default)s]')
    parser.add_argument('-f', dest='dbfile',
                        default=cli.dbfile,
                        help='database HDF file [%(default)s]')
    args = parser.parse_args()

    # "parallel" dict for sharing between procs
    manager = multiprocessing.Manager()
    fpsdct = manager.dict() 

    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    worker = functools.partial(_worker, 
                               size_x=args.size_x,
                               fpsdct=fpsdct)
    pool.map(worker, enumerate(args.files))

    # sort: order array to match file names in list `files`
    fps = np.array([fpsdct[ii] for ii in np.sort(fpsdct.keys())])
    # http://docs.h5py.org/en/latest/strings.html
    files = np.array([f for f in args.files]).astype('S')
    io.write_h5(args.dbfile, dict(files=files, fps=fps))    

