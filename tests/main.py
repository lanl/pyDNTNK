import os
os.environ["OMP_NUM_THREADS"] = "1"
from pyDNTNK import *
from pyDNMFk.utils import *
from pyDNMFk.dist_comm import *
import argparse
import warnings


def parser_pyNTNK():
    parser = argparse.ArgumentParser(description='Arguments for pyNTNK')
    parser.add_argument('--p_grid', type=int, nargs='+', help='Processor Grid')
    parser.add_argument('--fpath', type=str, default='../data/array.zarr', help='data path to read(eg: ../data/array.zarr)')
    parser.add_argument('--model', type=str, default='TT', help='TN model (TT/TK) for tensor train/Tucker models')
    parser.add_argument('--routine', type=str, default='NMF', help='NMF for nTT/nTK and SVD for TT/TK')
    parser.add_argument('--init', type=str, default='rand', help='NMF initializations: rand/nnsvd')
    parser.add_argument('--itr', type=int, default=1000, help='NMF iterations, default:1000')
    parser.add_argument('--norm', type=str, default='fro', help='Reconstruction Norm for NMF to optimize:KL/FRO')
    parser.add_argument('--method', type=str, default='mu', help='NMF update method:MU/BCD/HALS')
    parser.add_argument('--verbose', type=str2bool, default=True)
    parser.add_argument('--results_path', type=str, default='../results/', help='Path for saving results')
    parser.add_argument('--prune', type=str2bool, default=False, help='Prune zero row/column.')
    parser.add_argument('--precision', type=str, default='float32',
                        help='Precision of the data(float32/float64/float16.')
    parser.add_argument('--err', type=float, default=.1, help='Error for rank estimation at each stage')
    parser.add_argument('--tt_ranks', type=int, nargs='+', help='Ranks for each stage of decomposition')
    parser.add_argument('--save', type=str2bool, default=True, help=' Store TN factors')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parser_pyNTNK()
    main_comm = MPI.COMM_WORLD
    rank = main_comm.rank
    size = main_comm.size
    args.p_r, args.p_c = 1, size
    comm = MPI_comm(main_comm, args.p_r, args.p_c)
    args.rank = rank
    args.main_comm = main_comm
    args.comm1 = comm.comm
    args.comm = comm
    args.col_comm = comm.cart_1d_column()
    args.row_comm = comm.cart_1d_row()
    if main_comm.rank == 0: print('Starting ', args.model, ' Tensor Decomposition with ', args.routine)
    tt = pyNTNK(args.fpath, args, model=args.model)
    tt.fit()
    tt.error_compute()
    factors = tt.return_factors()
    if rank == 0:
        if args.save:
            print('Saving Factors now')
            try:
                os.mkdir(args.results_path)
            except:
                pass
            warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
            np.save(args.results_path + 'factors', factors)
    if main_comm.rank == 0: print('Tensor decomposition done.')
