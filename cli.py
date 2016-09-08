import os
from imgcmp import misc, env
pj = os.path.join

base_dir = pj(os.environ['HOME'], '.imgcmp')
convert_dr = pj(base_dir, 'convert')
cluster_dr = pj(base_dir, 'cluster')
dbfile = pj(base_dir, 'fingerprints.hdf')

for pp in base_dir, convert_dr, cluster_dr:
    misc.makedirs(pp)
