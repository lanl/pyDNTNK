from mpi4py import MPI
import numpy
import argparse
import sys
import pandas as pd
from pyDNMFk.utils import *
from pyDNMFk.pyDNMFk import *
import dask
import dask.array as da
from itertools import repeat
import zarr
import os
import pandas as pd
import numpy.linalg as la
from .tt_utils import *
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"



class pyDNTNK():
    r"""
    Performs the distributed Hierrachial Tensor decomposition of given matrix Tensor X into factors/tensor cores

    Parameters
    ----------
        fpath : str
            Path to the tensor file
        model : str (optional)
            Tensor networks model (TT/TK) where TT-Tensor train and TK-Tucker
        params : class
            Class which comprises following attributes
        params.routine : str(optinal)
            Decomposition method (NMF/SVD)
        params.init : str
            NMF initialization(rand/nnsvd)
        params.err  : float
            Error criteria for automating estimating ranks with SVD
        params.ranks : list
            List of ranks for decomposition
        params.comm : object
            Modified communicator object
        params.norm : str
            NMF norm to be minimized
        params.method : str
            NMF optimization method
        params.eps : float
            Epsilon value
        params.verbose : bool
            Flag to enable/disable display results
        params.save_factors : bool
            Flag to enable/disable saving computed factors"""
    def __init__(self, fpath,params, model='tt'):
        self.fpath = fpath
        self.params = params
        self.lazy_read_file()
        self.proc_grid = tuple([self.data_dim[i] // self.chunk_size[i] for i in range(len(self.data_dim))])
        assert list(self.proc_grid)==self.params.p_grid,"Data grid size doesn't match with proc-grid size. Use "+str(self.proc_grid)+" instead"
        self.norm = var_init(self.params,'norm',default='fro')
        self.tt_ranks =var_init(self.params,'tt_ranks',default=None)
        self.init = var_init(self.params,'init',default='rand')
        self.method = var_init(self.params,'method',default='mu')
        self.routine = var_init(self.params,'routine',default='nmf').lower()
        self.params.itr = var_init(self.params,'itr',default=1000)
        self.params.verbose = var_init(self.params,'verbose',default=True)
        self.model = model.lower()
        self.comm = self.params.main_comm
        self.comms = self.params.comm
        self.comm1 = self.params.comm1
        self.p = self.comms.size
        self.global_rank = self.comms.rank
        self.size = self.comms.size
        self.grid_count = len(self.proc_grid)
        self.periods = [1 for i in range(self.grid_count)]
        self.reorder = 0
        self.cart_comm = self.comm.Create_cart(self.proc_grid, self.periods, self.reorder)
        self.coords = self.cart_comm.Get_coords(self.global_rank)
        self.rk_ND = self.coords
        self.grid_2d = [1, self.p]
        self.rank_2d = glob_rank_to_ND_rank(self.grid_2d, self.global_rank)
        self.err = var_init(self.params,'err',default=1e-1)
        self.factors = []
        self.rel_error = []

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

    def compute_svd(self,data,k):
        r"""
        Computes distributed SVD for given data and then returns the factors

        Parameters
        ----------
            data : ndarray
                Data to be decomposed via SVD
            k : int
                Rank for decomposition

        Returns
        ----------
            singularValues: list
                A list of singular values
            U : ndarray
                U matrix
            V : ndarray
                V matrix
            rel_error : float
                Relative error of decomposition
        """
        m, n =data.shape
        n = self.comm1.allreduce(n)
        print(m,n)
        if k==-1:
            k = min(m,n)
        args = parse()
        args.m,args.n,args.k,args.comm = m,n,k,self.comms
        args.eps = np.finfo(data.dtype).eps
        if args.m<args.n: args.p_r,args.p_c = 1,self.size
        dsvd = DistSVD(args, data)
        singularValues, U, V = dsvd.svd()
        rel_error = dsvd.rel_error(U, np.diag(singularValues), V)
        if self.global_rank==0: print('relative error is:', rel_error )
        return singularValues,U,V,rel_error

    def determine_rank(self, X, err):
        r"""
        Automatically estimates the rank for each stage of decomposition with some error criteria

        Parameters
        ----------
            X : ndarray
                Matrix of whose rank is to be estimated
            err : float
                Error for estimation

        Returns
        ----------
            rank : int
                Estimated rank of decomposition
        """
        singularValues,_,_,_ = self.compute_svd(X,k=-1)
        ratio = np.array([np.linalg.norm(singularValues[k:]) / np.linalg.norm(singularValues) for k in
                 range(len(singularValues) - 1, 0, -1)])
        find_idx = numpy.nonzero(ratio <= err)
        rank = find_idx[0]
        if self.global_rank==0: print('Estimated rank=',rank)
        return rank



    def fit(self):
        r"""
        Calls the sub routines to perform distributed Tensor network decomposition with NMF/SVD decomposition for a given TT/TK based method
        Supports Tensor train and Tucker based Tensor networks decomposition.
        THe decomposition can be carried out via SVD/NMF.
        The ranks can be provided by user or can be automatically estimated with SVD.

        """
        N = self.data.shape
        prev_k = 1
        k_lst = []
        k_list = self.tt_ranks.copy()
        if self.model=='tt': depth = len(N)-1
        elif self.model=='tk': depth = len(N)

        for axis in range(depth):
            if self.global_rank==0: print('Decomposing for stage=',axis+1)
            if self.model=='tt':
               self.data = tn_data_operations(self.comm1,self.fpath,self.data).dist_reshape([prev_k * N[axis], -1])
            elif self.model=='tk':
               k_lst.append(prev_k)
               self.data = tn_data_operations(self.comm1,self.fpath,self.data).dist_reshape([N[axis], np.product(k_lst)*np.product(N[axis+1:])])

            if np.any(k_list) != False:
                self.this_k = k_list[axis]
            else:
                if self.global_rank==0: print('Estimating rank now with SVD...')
                self.this_k = self.determine_rank(self.data, self.err)

            if self.routine=='nmf':
                self.params.k = self.this_k
                if self.global_rank==0: print('Performing NMF for TN stage=',axis+1)
                W, H, rel_error = PyNMF(self.data, factors=None, params=self.params).fit()

            elif self.routine=='svd':
                if self.global_rank == 0: print('Performing SVD for TN stage=', axis + 1)
                singularValues, U, V,rel_error = self.compute_svd(self.data,self.this_k)
                S = np.diag(singularValues)
                W = U
                H = S @ V

            if self.model=='tt':
               self.factors.append(np.reshape(W, (prev_k, N[axis], self.this_k)))
            elif self.model=='tk':
                self.factors.append(W)
            if axis==depth-1:
                H_final = np.hstack((self.comm.allgather(H)))
                if self.model=='tk':
                    k_lst.append(self.this_k)
                    H_final = np.reshape(H_final,k_lst[1:])
                # H_final = np.hstack((block_idx_rank_H_tran(H_final,1, self.size)))
                self.factors.append(H_final)
            self.rel_error.append(rel_error)
            prev_k = self.this_k
            fpath = 'data/H_factor.zarr'
            self.data = tn_data_operations(self.comm1,fpath,H).lazy_store_file()

    def return_factors(self):
        r"""Returns the factors after decomposition"""
        if self.global_rank==0:print([i.shape for i in self.factors])
        return self.factors

    def error_compute(self):
        r"""Computes the final decomposition error as a upper bound"""
        self.tt_error = np.linalg.norm(self.rel_error)
        if self.global_rank==0:print('Overall error is::',self.tt_error)
        return {'NMF': self.rel_error, 'tt': self.tt_error}

