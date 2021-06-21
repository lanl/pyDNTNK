import numpy as np
import zarr
import numpy as np
import shutil
import argparse

def parser():
    r"""
    Reads the input arguments from the user and parses the parameters to the data generator module.
    """
    parser = argparse.ArgumentParser(description='Arguments for Distributed TN data generation')
    parser.add_argument('-d','--dim', type=int, nargs='+',help='Data dimension')
    parser.add_argument('-k','--k_grid',type=int,nargs='+',help='k grid')
    parser.add_argument('-p','--p_grid',type=int,nargs='+',help='p grid')
    parser.add_argument('-f','--fpath',type=str,default='../data/array.zarr',help='Path for storage')
    parser.add_argument('-t','--dtype',type=str,default='float32',help='Data type(float32/float64)')
    args = parser.parse_args()
    return args

class data_generator():
    r"""
    Generates synthetic data for tensor networks. THe factors are generated randomly and then taken product into TT format.
    """
    def __init__(self,args):
        self.dim = args.dim
        self.p_grid = args.p_grid
        self.k_grid = args.k_grid
        self.fpath = args.fpath
        self.dtype = args.dtype
        try:
            shutil.rmtree(self.fpath)
        except:
            pass

    def fit(self):
        '''generates and save factors into Zarr file'''
        print('creating data of size', self.dim, 'k size', self.k_grid)
        factors = {}
        n = len(self.dim)
        for i in range(n - 1, -1, -1):
            if i == 0:
                factors[i] = np.random.rand(self.dim[i], self.k_grid[i]).astype(self.dtype)  # .astype(np.float32)
            elif i == n - 1:
                factors[i] = np.random.rand(self.k_grid[i - 1], self.dim[i]).astype(self.dtype) # .astype(np.float32)
            else:
                factors[i] = np.random.rand(self.k_grid[i - 1], self.dim[i], self.k_grid[i]).astype(self.dtype)  # .astype(np.float32)
            if i == n - 2:
                result = np.tensordot(factors[i], factors[i + 1], axes=1).astype(self.dtype)
            elif i < n - 2:
                result = np.tensordot(factors[i], result, axes=1).astype(self.dtype)

        chunk=tuple(list(int(self.dim[i]/self.p_grid[i]) for i in range(len(self.dim))))
        store = zarr.DirectoryStore(self.fpath)
        z = zarr.zeros(self.dim, chunks=chunk, store=store, overwrite=True)
        z[...]=result


if __name__ == '__main__':
    args = parser()
    data_gen = data_generator(args)
    data_gen.fit()






