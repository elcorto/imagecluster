import itertools, math, os

def makedirs(path):
    """Same as os.makedirs() but silently skips existing dirs."""
    pp = './' if path.strip() == '' else path
    if not os.path.exists(pp):
        os.makedirs(pp)


# http://stackoverflow.com/a/25649669
def idx_square2cond(ii, jj, nn):
    return ii*nn + jj - ii*(ii+1)/2 - ii - 1

# http://stackoverflow.com/a/14839010
def idx_cond2square(kk, nn):
    b = 1 - 2*nn
    x = int(math.floor((-b - math.sqrt(b**2 - 8*kk))/2))
    y = kk + x*(b + x + 2)/2 + 1
    return (x,y)  


def view_lst(lst):
    from pwtools.common import system
    cmd = 'qiv -fm ' + ' '.join(lst)
    system(cmd, wait=True)
