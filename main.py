import os

# Keep BLAS/OpenMP thread counts conservative by default on Windows to reduce
# virtual-memory pressure during worker/process import storms.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from matpub.pipeline import main


if __name__ == "__main__":
    main()
