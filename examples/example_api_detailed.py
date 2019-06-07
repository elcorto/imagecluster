#!/usr/bin/python3

# Detailed API example. We show which functions are called inside
# get_image_data() (read_image_arrays(), get_model(), fingerprints(), pca(),
# read_timestamps()) and show more options such as time distance scaling.

from imagecluster import calc, io as icio, postproc


##image_arrays,fingerprints,timestamps = icio.get_image_data(
##    'pics/',
##    pca_kwds=dict(n_components=0.95),
##    img_kwds=dict(size=(224,224)))

# Create image database in memory. This helps to feed images to the NN model
# quickly.
image_arrays = icio.read_image_arrays('pics/', size=(224,224))

# Create Keras NN model.
model = calc.get_model()

# Feed images through the model and extract fingerprints (feature vectors).
fingerprints = calc.fingerprints(image_arrays, model)

# Optionally run a PCA on the fingerprints to compress the dimensions. Use a
# cumulative explained variance ratio of 0.95.
fingerprints = calc.pca(fingerprints, n_components=0.95)

# Read image timestamps. Need that to calculate the time distance, can be used
# in clustering.
timestamps = icio.read_timestamps('pics/')

# Run clustering on the fingerprints. Select clusters with similarity index
# sim=0.5. Mix 80% content distance with 20% timestamp distance (alpha=0.2).
clusters = calc.cluster(fingerprints, sim=0.5, timestamps=timestamps, alpha=0.2)

# Create dirs with links to images. Dirs represent the clusters the images
# belong to.
postproc.make_links(clusters, 'pics/imagecluster/clusters')

# Plot images arranged in clusters and save plot.
fig,ax = postproc.plot_clusters(clusters, image_arrays)
fig.savefig('foo.png')
postproc.plt.show()
