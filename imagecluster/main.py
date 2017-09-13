import os
import numpy as np
from imagecluster import imagecluster as ic
pj = os.path.join


def main(imagedir, sim=0.5):
    """Example main app using this library.
    
    Parameters
    ----------
    imagedir : str
        path to directory with images
    sim : float (0..1)
        similarity index (see imagecluster.cluster())
    """
    dbfn = pj(imagedir, 'fingerprints.pk')
    if not os.path.exists(dbfn):
        print("no fingerprints database {} found".format(dbfn))
        files = ic.get_files(imagedir)
        model = ic.get_model()
        print("running all images through NN model ...".format(dbfn))
        fps = ic.fingerprints(files, model, size=(224,224))
        ic.write_pk(fps, dbfn)
    else:
        print("loading fingerprints database {} ...".format(dbfn))
        fps = ic.read_pk(dbfn)
    print("clustering ...")
    ic.make_links(ic.cluster(fps, sim), pj(imagedir, 'clusters'))
