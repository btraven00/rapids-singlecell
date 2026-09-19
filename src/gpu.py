"""GPU runtime setup shared across rapids-singlecell entrypoints.

Conservative RMM settings (no managed memory, no pool allocator) so that
peak GPU working-set is the *true* working-set, not a pre-warmed pool.
This matters when the GPU memory footprint is itself a benchmark dimension.
"""

import cupy as cp
import rmm
from rmm.allocators.cupy import rmm_cupy_allocator

_initialized = False


def setup_gpu():
    global _initialized
    if _initialized:
        return
    rmm.reinitialize(managed_memory=False, pool_allocator=False)
    cp.cuda.set_allocator(rmm_cupy_allocator)
    _initialized = True


def to_host(array):
    """Return a host copy of a possibly-device array.

    rsc.get.anndata_to_CPU(..., convert_all=True) does not reliably bring
    obsm/varm back: after pca the embedding stays a cupy array for the lanczos
    and randomized solvers, while covariance_eigh yields a host one. cupy then
    refuses the implicit np.asarray conversion, so a run dies at write time
    having already done the compute -- and only for some solvers, which makes
    it look like a solver bug rather than a transfer that did not happen.

    Ask for the transfer explicitly instead of trusting convert_all.
    """
    return array.get() if hasattr(array, "get") else array
