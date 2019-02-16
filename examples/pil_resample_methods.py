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
# We test the differrnt resample interpolation methods and measure the speed
# for resizing an image (768, 1024) -> (224,224). We also compare the resample
# quality as difference to the best possible resample result, which we defined
# to be the LANCZOS method (from visual inspection and b/c it is
# computationally the most elaborate).
#
# methods:
#                                         
# NEAREST             = 0
# LANCZOS = ANTIALIAS = 1 # reference result
# BILINEAR            = 2
# BICUBIC             = 3
#
# resample quality (output from this script):
# 
#    method  quality (ref=1)   speedup  speedup/quality      time
# 0     0.0         1.000000  6.233622         6.233622  0.031676
# 1     1.0         0.000000  1.000000              NaN  0.197458
# 2     2.0         0.381500  2.269723         5.949475  0.086996
# 3     3.0         0.160711  1.499384         9.329660  0.131693
# 
# Method 0 is the fastest, but has the highest difference to the reference
# result. Visually, methods 1 (reference) and 3 are almost indistinguishable.
# Taking speed into account, method 3 is the best compromise (speedup/quality).

import timeit

import PIL
import scipy.misc
from matplotlib import cm, use
use('Agg')
from matplotlib import pyplot as plt
import numpy as np

import pandas as pd

# face() retruns a (768, 1024, 3) RGB array
im = PIL.Image.fromarray(scipy.misc.face())

fig = plt.figure()
ax = fig.add_subplot(111)
nn = 224
nmeth = 4
imgs = np.empty((nmeth, nn, nn))
timings = []
for method in range(nmeth):
    stmt = "np.array(im.convert('L').resize((nn,nn), method), dtype=float)"
    ctx = dict(im=im, method=method, nn=nn, np=np)
    timings.append(min(timeit.repeat(stmt, number=20, repeat=5, globals=ctx)))
    arr = eval(stmt)
    ax.matshow(arr, cmap=cm.gray)
    fig.savefig('method_{}.png'.format(method))
    ax.cla()
    imgs[method,...] = arr

refidx = 1
diffs = np.sqrt(((imgs - imgs[refidx,...][None,...])**2.0).sum(axis=(1,2)))
diffs = diffs / diffs.max()
df = pd.DataFrame()
for method,tup in enumerate(zip(timings,diffs)):
    time,quality = tup
    speedup = timings[refidx]/time
    row = {'method': method,
           f'quality (ref={refidx})': quality,
           'time': time,
           'speedup': speedup,
           'speedup/quality': np.nan if method == refidx else speedup/quality,
           }
    df=df.append(row, ignore_index=True)

print(df)
