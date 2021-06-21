import numpy as np
from copy import deepcopy
import time
import dask
import dask.array as da
from pyDNMFk.utils import *
import zarr

def glob_rank_to_ND_rank(p_grid,glob_RANK):
    r"""Converts global rank to cartesian coordinates i.e  0 -> (0,0,0)"""
    p_grid = list(p_grid)
    p_grid.reverse()
    p_grid = [1]+p_grid
    p_grid_re = list(np.cumprod(p_grid)[:-1])
    p_grid_re.reverse()
    tmp = np.zeros(len(p_grid_re))
    res1=0
    for i in range(len(p_grid_re)):
        if i==0:
            res1 = glob_RANK//p_grid_re[i]
        else:
            res1 = res2//p_grid_re[i]
        res2 = glob_RANK%p_grid_re[i]
        tmp[i] = res1
    return tmp.astype(int)

class tn_data_operations():
    """Perform various data opetations on TN data"""
    def __init__(self,comm,fpath=None,data=None):
        self.fpath = fpath
        self.comms = comm
        self.data = data
        self.size = self.comms.size

    def lazy_read_file(self):
        r"""
        Reads file from the Zarr file system to each MPI rank and then handles the object to dask array

        Parameters
        ----------
            fpath : str
                File path to the Zarr Tensor File
        """
        store = zarr.DirectoryStore(self.fpath)
        z_array = zarr.open(store=store, mode='r')
        self.da_input = da.from_array(z_array)
        self.data = self.da_input
        self.data_dim = self.data.shape
        self.chunk_size = z_array.chunks

    def data_block_idx_range(self, chunk_size):
        r"""Computes the block index range of data for each MPI rank

        Parameters
        ----------
        chunk_size: tuple
            Total data size based on which the index range is determined

        Returns
        ----------
        dtr_blk_idx : list
            List of index range for each MPI rank

        """
        target_shape = chunk_size
        dtr_blk = determine_block_params(self.comms, (1, self.size), (target_shape[0], target_shape[1]))
        dtr_blk_idx = dtr_blk.determine_block_index_range_asymm()
        dtr_blk_shp = dtr_blk.determine_block_shape_asymm()
        return dtr_blk_idx


    def dist_reshape(self, target_shape):
        r"""Computes the Distributed reshape operation with the DASK object

        Parameters
        ----------
        data: DASK object
            Data stored into DASK format
        target_shape : tuple
            Desired data shape

        Returns
        ----------
        data : ndarray
            Reshapes the dask array into desired shape and then computes the chunk of data for each MPI rank

        """
        data = self.data.reshape(target_shape)
        loc_cols_idx = self.data_block_idx_range(data.shape)
        data = data[loc_cols_idx[0][0]:loc_cols_idx[1][0]+1, loc_cols_idx[0][1]:loc_cols_idx[1][1]+1].compute()
        return data

    def lazy_store_file(self):
        r"""Stores the ndarray from each MPI rank into a shared file system

        Parameters
        ----------
        data: ndarray
            Data to be stored from each MPI rank
        target_shape : fpath
            Path to which file to be stored

        """
        store = zarr.DirectoryStore(self.fpath)
        k, n = self.data.shape[0], self.data.shape[1]
        n = self.comms.allreduce(n)
        if self.comms.rank == 0:
            z_array = zarr.open(store, mode='w', shape=[k, n], chunks=self.data.shape)
        self.comms.barrier()
        z_array = zarr.open(store, mode='r+')
        loc_cols_idx = self.data_block_idx_range([k,n])
        z_array[loc_cols_idx[0][0]:loc_cols_idx[1][0]+1, loc_cols_idx[0][1]:loc_cols_idx[1][1]+1] = self.data  # This does not account permutation of  H blocks
        self.comms.barrier()
        self.data = da.from_array(z_array)
        return self.data