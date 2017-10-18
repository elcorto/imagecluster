import numpy as np
import re, pickle, os

##from matplotlib import pyplot as plt
##
##
##def view_image_list(files):
##    for filename in files:
##        fig,ax = plt.subplots()
##        ax.imshow(plt.imread(filename))
##    plt.show()


def read_pk(fn):
    with open(fn, 'rb') as fd:
        ret = pickle.load(fd)
    return ret


def write_pk(obj, fn):
    with open(fn, 'wb') as fd:
        pickle.dump(obj, fd)


def get_files(dr, ext='jpg|jpeg|bmp|png'):
    rex = re.compile(r'^.*\.({})$'.format(ext), re.I)
    return [os.path.join(dr,base) for base in os.listdir(dr) if rex.match(base)]
