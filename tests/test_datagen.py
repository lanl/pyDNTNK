'''Generate data first here'''
import sys
from pyDNTNK.TN_data_generator import *
from pyDNMFk.utils import *
import dask.array as da
#@pytest.mark.mpi
def test_datagen():
    args = parse()
    args.dim = [10,10,10,10]
    args.k_grid = [2,2,2,2]
    args.p_grid = [2,1,1,1]
    args.fpath = '../data/array.zarr'
    args.dtype = 'float32'
    '''Lets generate data'''
    data = data_generator(args).fit()
    '''Lets test data'''
    store = zarr.DirectoryStore(args.fpath)
    z_array = zarr.open(store=store, mode='r')
    data = da.from_array(z_array).astype(args.dtype)
    data_dim = data.shape
    chunk_size = z_array.chunks
    assert list(data.shape)==args.dim
    assert list(chunk_size)==[args.dim[i]/args.p_grid[i] for i in range(len(args.dim))]
    assert data.dtype == args.dtype

def main():
    test_datagen()


if __name__ == '__main__':
    main()
