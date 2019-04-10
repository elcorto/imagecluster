#!/usr/bin/python3

from imagecluster import calc as ic
from imagecluster import io as icio
from imagecluster import postproc as pp

# # Create image database in memory. This helps to feed images to the NN model
# # quickly.
# image_arrays = icio.read_image_arrays('pics/', size=(224,224))
#
# # Create Keras NN model.
# model = ic.get_model()
#
# # Feed images through the model and extract fingerprints (feature vectors).
# fingerprints = ic.fingerprints(image_arrays, model)
#
# # Optionally run a PCA on the fingerprints to compress the dimensions. Use a
# # cumulative explained variance ratio of 0.95.
# fingerprints = ic.pca(fingerprints, n_components=0.95)
#
# # Read image timestamps. Need that to calculate the time distance, can be used
# # in clustering.
# timestamps = icio.read_timestamps('pics/')

# XXX where on disk? add to README
# Convenience function to perform the steps above. Check for existing
# `image_arrays` and `fingerprints` database files on disk and load them if
# present. Running this again only loads data from disk, which is fast.
image_arrays,fingerprints,timestamps = icio.get_image_data(
    'pics/',
    pca_kwds=dict(n_components=0.95))

# Run clustering on the fingerprints. Select clusters with similarity index
# sim=0.5. Mix 80% content distance with 20% timestamp distance (alpha=0.2).
clusters = ic.cluster(fingerprints, sim=0.5, timestamps=timestamps, alpha=0.2)

# Create dirs with links to images. Dirs represent the clusters the images
# belong to.
pp.make_links(clusters, 'pics/imagecluster/clusters')

# Plot images arranged in clusters.
pp.visualize(clusters, image_arrays)
