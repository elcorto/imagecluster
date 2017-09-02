#!/usr/bin/env python3

# help for PIL.Image.Image.resize
# -------------------------------
# 
# PIL.Image.Image.resize(self, size, resample=0)
#
# :param size: The requested size in pixels, as a 2-tuple:
#    (width, height).
# :param resample: An optional resampling filter.  This can be
#    one of :py:attr:`PIL.Image.NEAREST` (use nearest neighbour),
#    :py:attr:`PIL.Image.BILINEAR` (linear interpolation),
#    :py:attr:`PIL.Image.BICUBIC` (cubic spline interpolation), or
#    :py:attr:`PIL.Image.LANCZOS` (a high-quality downsampling filter).
#    If omitted, or if the image has mode "1" or "P", it is
#    set :py:attr:`PIL.Image.NEAREST`.
# :returns: An :py:class:`~PIL.Image.Image` object.
# 
# Each PIL.Image.<method> variable is actually an integer (e.g. Image.NEAREST
# is 0).
#
# We tried the resample interpolation methods and measured the speed (ipython's
# timeit) for resizing an image 3840x2160 -> 8x8. We also compared the resample
# quality as difference to the best possible resample result, which we defined
# to be the LANCZOS method (from visual inspection and b/c it is
# computationally the most elaborate).
#
#                                      speed [ms]     
# Image.NEAREST                   = 0  29.9e-3
# Image.LANCZOS = Image.ANTIALIAS = 1  123          # reference result
# Image.BILINEAR                  = 2  47
# Image.BICUBIC                   = 3  87
#
# resample quality (see pil_resample_methods.py)
# method = 0, diff to ref(1) = 1.0
# method = 1, diff to ref(1) = 0.0
# method = 2, diff to ref(1) = 0.135679761399
# method = 3, diff to ref(1) = 0.0549413095836
#
# -> method=2 is probably the best compromise between quality and speed


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
nn = 16
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
