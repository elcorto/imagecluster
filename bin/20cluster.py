#!/usr/bin/python

import os, shutil, argparse
import numpy as np
from imgcmp import misc, io, calc, cli

pj = os.path.join

if __name__ == '__main__':
    
    desc = """
Read fingerprints database, perform clustering.
"""
    parser = argparse.ArgumentParser(description=desc) 
    parser.add_argument('frac', type=float,
                        help='similarity fraction')
    parser.add_argument('-f', dest='dbfile',
                        default=cli.dbfile,
                        help='database HDF file [%(default)s]')
    args = parser.parse_args()
    
    db = io.read_h5(args.dbfile)
    files = db['/files'] 
    fps = db['/fps']
    
    # {1: [list_of_files], 2: [list_of_files], ...}
    cluster_dct = calc.cluster(files, fps, args.frac, 'hamming')

    # [[list_of_files], [list_of_files], ...]
    clst_multi = [x for x in cluster_dct.values() if len(x) > 1]

    # {number_of_files1: [[list_of_files], [list_of_files],...],
    #  number_of_files2: [[list_of_files],...],
    # }
    cdct_multi = {}
    for x in clst_multi:
        nn = len(x)
        if not cdct_multi.has_key(nn):
            cdct_multi[nn] = [x]
        else:    
            cdct_multi[nn].append(x)

    print("items per cluster : number of such clusters")
    shutil.rmtree(cli.view_dr)
    for n_in_cluster in np.sort(cdct_multi.keys()):
        cluster_list = cdct_multi[n_in_cluster]
        print("{} : {}".format(n_in_cluster, len(cluster_list)))
        for iclus, lst in enumerate(cluster_list):
            dr = pj(cli.view_dr, 
                    'cluster_with_{}'.format(n_in_cluster),
                    'cluster_{}'.format(iclus))
            for fn in lst:
                link = pj(dr, os.path.basename(fn))
                misc.makedirs(os.path.dirname(link))
                os.symlink(fn, link)

    ##key = raw_input("View? [N,y] ")
    ##if key.lower() == 'y':
     
