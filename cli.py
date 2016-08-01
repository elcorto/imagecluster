import os
from imgcmp import misc
pj = os.path.join

base_dir = pj(os.environ['HOME'], '.imgcmp')
convert_dr = pj(base_dir, 'convert')
view_dr = pj(base_dir, 'view')
dbfile = './fingerprints.hdf'

for pp in base_dir, convert_dr, view_dr:
    misc.makedirs(pp)
