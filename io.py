import h5py

def write_h5(fn, dct, mode='w', **kwds):
    fh = h5py.File(fn, mode=mode, **kwds)
    for key,val in dct.iteritems():
        _key = key if key.startswith('/') else '/'+key
        fh[_key] = val
    fh.close()


def read_h5(fn):
    fh = h5py.File(fn, mode='r')
    dct = {}
    def get(key, obj, dct=dct):
        if isinstance(obj, h5py.Dataset):
            _key = key if key.startswith('/') else '/'+key
            dct[_key] = obj.value
    fh.visititems(get)
    fh.close()
    return dct

def read_db(dbfile):
    db = read_h5(dbfile)
    return db['/files'], db['/fps']
