import os
import sys
os.environ["OMP_NUM_THREADS"] = "1"
from pyDNTNK import *
from pyDNTNK import pyDNTNK
from pyDNMFk.utils import *
from pyDNMFk.dist_comm import *

'''Test the results here'''
#@pytest.mark.mpi
def test_tt():
    args = parse()
    args.fpath = '../data/array.zarr'
    main_comm = MPI.COMM_WORLD
    rank = main_comm.rank
    size = main_comm.size
    args.p_r, args.p_c = 1, size
    comm = MPI_comm(main_comm, args.p_r, args.p_c)
    args.rank = rank
    args.main_comm = main_comm
    args.comm1 = comm.comm
    args.comm = comm
    args.p_grid = [2,1,1,1]
    args.tt_ranks = [2,2,2,2]
    args.col_comm = comm.cart_1d_column()
    args.row_comm = comm.cart_1d_row()
    args.model,args.routine = 'tt','nmf'
    if main_comm.rank == 0: print('Starting ', args.model, ' Tensor Decomposition with ', args.routine)
    tt = pyDNTNK(args.fpath, args, model=args.model)
    tt.fit()
    tt.error_compute()
    factors = tt.return_factors()
    assert len(factors)==4
    assert([i<1e-2 for i in tt.rel_error])


def main():
    test_tt()


if __name__ == '__main__':
    main()