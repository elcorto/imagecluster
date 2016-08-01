import PIL
import scipy.misc
from matplotlib import cm, use
use('Agg')
from matplotlib import pyplot as plt
import numpy as np

# face() retruns a (x,y,3) RGB array
im = PIL.Image.fromarray(scipy.misc.face())

fig = plt.figure()
ax = fig.add_subplot(111)
nn = 4
img1d = []
for method in range(4):
    ##arr = np.array(im.convert('L').resize((nn,nn),method).getdata(), dtype=float).reshape(nn,nn)
    # much faster
    arr = np.array(im.convert('L').resize((nn,nn), method), dtype=float)
    ax.matshow(arr, cmap=cm.gray)
    fig.savefig('method_{}.png'.format(method))
    ax.cla()
    img1d.append(arr.flatten())

refidx = 1
ref = img1d[refidx]
diffs = []
for m,arr1d in enumerate(img1d):
    diffs.append(np.sqrt(((ref - arr1d)**2.0).sum()))

diffs = np.array(diffs)
diffs = diffs / diffs.max()
for m,d in enumerate(diffs):
    msg = "method = {m}, diff to ref({refidx}) = {d}"
    print(msg.format(**dict(m=m, refidx=refidx,d=d)))
