``imagecluster`` is a package for clustering images by content. We use a
pre-trained deep convolutional neural network to calculate image fingerprints
which represent content. Those are used to cluster similar images. In addition
to pure image content, it is possible to mix in timestamp information which
improves clustering for temporally uncorrelated images.
