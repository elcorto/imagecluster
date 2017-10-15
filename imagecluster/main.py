import os
import numpy as np
from imagecluster import imagecluster as ic
pj = os.path.join


ic_base_dir = 'imagecluster'


def get_files(dr):
    return [pj(dr,base) for base in os.listdir(dr) if base != ic_base_dir] 


def main(imagedir, sim=0.5):
    """Example main app using this library.
    
    Parameters
    ----------
    imagedir : str
        path to directory with images
    sim : float (0..1)
        similarity index (see imagecluster.cluster())
    """
    dbfn = pj(imagedir, ic_base_dir, 'fingerprints.pk')
    if not os.path.exists(dbfn):
        os.makedirs(os.path.dirname(dbfn), exist_ok=True)
        print("no fingerprints database {} found".format(dbfn))
        files = get_files(imagedir)
        model = ic.get_model()
        print("running all images through NN model ...".format(dbfn))
        fps = ic.fingerprints(files, model, size=(224,224))
        ic.write_pk(fps, dbfn)
    else:
        print("loading fingerprints database {} ...".format(dbfn))
        fps = ic.read_pk(dbfn)
    print("clustering ...")
    ic.make_links(ic.cluster(fps, sim), pj(imagedir, ic_base_dir, 'clusters'))
