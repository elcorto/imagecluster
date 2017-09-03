import os, pickle
import numpy as np
from imagecluster import imagecluster as ic
pj = os.path.join

def main(imagedir, sim=0.5):
    """Example main app using this library.     """
    dbfn = pj(imagedir, 'fingerprints.pk')
    if not os.path.exists(dbfn):
        print("no fingerprints database {} found".format(dbfn))
        files = ic.get_files(imagedir)
        model = ic.get_model()
        print("running all images thru NN model ...".format(dbfn))
        fps = ic.fingerprints(files, model, size=(224,224))
        with open(dbfn, 'wb') as fd:
            pickle.dump(fps, fd)
        fd.close()
    else:
        print("loading fingerprints database {} ...".format(dbfn))
        with open(dbfn, 'rb') as fd:
            fps = pickle.load(fd)
    print("clustering ...")
    clusters = ic.cluster(list(fps.keys()), 
                          np.array(list(fps.values())), 
                          sim)
    ic.make_links(clusters, pj(imagedir, 'clusters'))
