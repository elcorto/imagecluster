#!/usr/bin/python3

import os, shutil, argparse
from imgcmp import misc, io, calc, cli, env
import numpy as np

pj = os.path.join

if __name__ == '__main__':
    
    desc = """
Read fingerprints database, perform clustering.
"""
    parser = argparse.ArgumentParser(description=desc) 
    parser.add_argument('sim', type=float, metavar='SIM',
                        help='similarity (0..1) where e.g. 0.1 = strict '
                             'similarity (very similar images + small '
                             'clusters); try 0.2 first, then increase for more '
                             'but less similar images per cluster')
    parser.add_argument('-f', dest='dbfile',
                        default=cli.dbfile,
                        help='database HDF file [%(default)s]')
    args = parser.parse_args()
    
    db = io.read_h5(args.dbfile)
    files = db['/files'] 
    fps = db['/fps']
     
    # handle numpy 'S' type strings from HDF file:
    # convert array([b'x', b'y'], dtype='|S1') -> ['a', 'b']
    if type(files[0]) in [type(b'x'), np.bytes_]:
        files = [f.decode() for f in files]

    # {1: [list_of_files], 2: [list_of_files], ...}
    cluster_dct = calc.cluster(files, fps, args.sim, 'hamming')

    # [[list_of_files], [list_of_files], ...]
    clst_multi = [x for x in cluster_dct.values() if len(x) > 1]

    # {number_of_files1: [[list_of_files], [list_of_files],...],
    #  number_of_files2: [[list_of_files],...],
    # }
    cdct_multi = {}
    for x in clst_multi:
        nn = len(x)
        if not (nn in cdct_multi.keys()):
            cdct_multi[nn] = [x]
        else:    
            cdct_multi[nn].append(x)

    print("items per cluster : number of such clusters")
    shutil.rmtree(cli.cluster_dr)
    for n_in_cluster in np.sort(list(cdct_multi.keys())):
        cluster_list = cdct_multi[n_in_cluster]
        print("{} : {}".format(n_in_cluster, len(cluster_list)))
        for iclus, lst in enumerate(cluster_list):
            dr = pj(cli.cluster_dr, 
                    'cluster_with_{}'.format(n_in_cluster),
                    'cluster_{}'.format(iclus))
            for fn in lst:
                link = pj(dr, os.path.basename(fn))
                misc.makedirs(os.path.dirname(link))
                os.symlink(os.path.abspath(fn), link)
