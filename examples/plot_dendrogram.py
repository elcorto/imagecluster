from matplotlib import pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram

from imagecluster import calc as ic

ias = ic.image_arrays('pics/', size=(224,224))
model = ic.get_model()
fps = ic.fingerprints(ias, model)
clusters,extra = ic.cluster(fps, sim=0.5, extra_out=True)

# linkage matrix Z
Z = extra['Z']

fig,ax = plt.subplots()
dendrogram(Z, ax=ax)

# Adjust yaxis labels (values from Z[:,2]) to our definition of the `sim`
# parameter.
ymin, ymax = ax.yaxis.get_data_interval()
tlocs = np.linspace(ymin, ymax, 5)
ax.yaxis.set_ticks(tlocs)
tlabels = np.linspace(1, 0, len(tlocs))
ax.yaxis.set_ticklabels(tlabels)
ax.set_xlabel("image index")
ax.set_ylabel("sim")

fig.savefig('dendrogram.png')
plt.show()
