#!/usr/bin/python

import sys, multiprocessing, functools, argparse
from PIL import Image
import numpy as np
from imgcmp import calc, io, cli

def _worker(tup, size_x=None, fpsdct=None):
    print(tup)
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
    parser.add_argument('files', metavar='FILE', nargs='+',
                        help='image file names')
    parser.add_argument('-x', dest='size_x',
                        default=8, type=int,
                        help='resize images to (size_x, size_x), fingerprints '
                             'are then (size_x**2,) 1d arrays [%(default)s]')
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
    io.write_h5(args.dbfile, dict(files=args.files, fps=fps))    

