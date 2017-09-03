from scipy.spatial import distance
from scipy.cluster import hierarchy
import numpy as np
from matplotlib import pyplot as plt

import PIL.Image, os, multiprocessing, shutil
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model

pj = os.path.join

def get_model():
    # base_model.summary():
    #     ....
    #     block5_conv4 (Conv2D)        (None, 15, 15, 512)       2359808   
    #     _________________________________________________________________
    #     block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0         
    #     _________________________________________________________________
    #     flatten (Flatten)            (None, 25088)             0         
    #     _________________________________________________________________
    #     fc1 (Dense)                  (None, 4096)              102764544 
    #     _________________________________________________________________
    #     fc2 (Dense)                  (None, 4096)              16781312  
    #     _________________________________________________________________
    #     predictions (Dense)          (None, 1000)              4097000   
    # 
    # model: get output from pre-last fully connected layer 'fc2'
    base_model = VGG16(weights='imagenet', include_top=True)
    model = Model(inputs=base_model.input,
                  outputs=base_model.get_layer('fc2').output)
    return model


def fingerprint(fn, model, size):
    # keras.preprocessing.image.load_img() uses img.rezize(shape) with the
    # default interpolation which is pretty bad (see
    # imagecluster/play/pil_resample_methods.py). Given that we are restricted
    # to small inputs of 244x244 by the VGG network, we should do our best to
    # keep as much information from the original image as possible. This is a
    # gut feeling, untested. But given that model.predict() is 10x slower than
    # the while PIL image loading and resizing .. who cares.
    #
    # (244, 244, 3)
    ##img = image.load_img(fn, target_size=size)
    img = PIL.Image.open(fn).resize(size, 2)
    # (244, 244, 3)
    arr3d = image.img_to_array(img)
    # (1, 244, 244, 3)
    arr4d = np.expand_dims(arr3d, axis=0)
    # (1, 244, 244, 3)
    arr4d_pp = preprocess_input(arr4d)
    return model.predict(arr4d_pp)[0,:]


def _worker(fn, model, size):
    print(fn)
    return fn, fingerprint(fn, model, size)


def fingerprints(files, model, size=(224,224)):
    # Cannot use multiprocessing:
    # TypeError: can't pickle _thread.lock objects 
    # The error doesn't come from functools.partial since those objects are
    # pickable since python3. The reason is the keras.model.Model, which is not
    # pickable. However keras with tensorflow backend runs multithreaded
    # (model.predict()), so we don't need that.
##    worker = functools.partial(_worker,
##                               model=model,
##                               size=size)
##    pool = multiprocessing.Pool(multiprocessing.cpu_count())
##    return dict(pool.map(worker, files))
    return dict(_worker(fn, model, size) for fn in files)


def make_links(clusters, cluster_dr):
    # [[list_of_files], [list_of_files], ...]
    clst_multi = [x for x in clusters.values() if len(x) > 1]

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

    print("cluster dir: {}".format(cluster_dr))
    print("items per cluster : number of such clusters")
    if os.path.exists(cluster_dr):
        shutil.rmtree(cluster_dr)
    for n_in_cluster in np.sort(list(cdct_multi.keys())):
        cluster_list = cdct_multi[n_in_cluster]
        print("{} : {}".format(n_in_cluster, len(cluster_list)))
        for iclus, lst in enumerate(cluster_list):
            dr = pj(cluster_dr,
                    'cluster_with_{}'.format(n_in_cluster),
                    'cluster_{}'.format(iclus))
            for fn in lst:
                link = pj(dr, os.path.basename(fn))
                os.makedirs(os.path.dirname(link), exist_ok=True)
                os.symlink(os.path.abspath(fn), link)


def get_files(dr):
    return [pj(dr,base) for base in os.listdir(dr)] 


def cluster(files, fps, sim=0.5, method='average', metric='euclidean'):
    """Hierarchical clustering of images `files` based on image fingerprints
    `fps`.

    files : list of file names
    sim : float 0..1
        similarity tolerance (1=max. allowed similarity tolerance, all images
        are considered similar and are in one cluster, 0=zero similarity
        allowed, each image is it's own cluster of size 1)
    fps : 2d array (len(files), N)
        flattened fingerprints (1d array for each image), where `size` is the
        argument to one of the *hash() functions 
    method : see scipy.hierarchy.linkage(), all except 'centroid' produce
        pretty much the same result
    metric : see scipy.hierarchy.linkage(), make sure to use 'euclidean' in
        case of method='centroid', 'median' or 'ward'
    """
    dfps = distance.pdist(fps, metric)
    # hierarchical/agglomerative clustering (Z = linkage matrix, construct
    # dendrogram) 
    Z = hierarchy.linkage(dfps, method=method, metric=metric)
    # cut dendrogram, extract clusters
    cut = hierarchy.fcluster(Z, t=dfps.max()*sim, criterion='distance')
    clusters = dict((ii,[]) for ii in np.unique(cut))
    for iimg,iclus in enumerate(cut):
        clusters[iclus].append(files[iimg])
    return clusters 


def view_image_list(lst):
    for filename in lst:
        fig,ax = plt.subplots()
        ax.imshow(plt.imread(filename))
    plt.show()
